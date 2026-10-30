"""Offline CPU checks against checked-in Audio8 golden tensors."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from safetensors.torch import load_file

COMFYUI = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(COMFYUI))
sys.path.insert(0, str(PACKAGE))

from comfy.cli_args import args

args.cpu = True

import comfy_kitchen
from audio8.model import Audio8Model, KVCache, filtered_scores, rope_table


MANIFEST = json.loads((FIXTURES / "audio8_tiny_golden.json").read_text(encoding="utf-8"))
CONFIG = MANIFEST["config"]
TENSORS = load_file(str(FIXTURES / MANIFEST["tensor_file"]))


@pytest.fixture(scope="module")
def native_model():
    model = Audio8Model(CONFIG, nn).eval()
    weights = {key.removeprefix("weights."): value for key, value in TENSORS.items() if key.startswith("weights.")}
    model.load_state_dict(weights, strict=True)
    return model


def test_tiny_golden_fixture_is_complete_and_versioned():
    assert MANIFEST["format"] == "audio8-tiny-golden-v1"
    assert len(MANIFEST["upstream"]["revision"]) == 40
    assert MANIFEST["generation"]["weights"].startswith("synthetic")
    assert "expected.prefill_logits" in TENSORS
    assert "expected.fast_codes" in TENSORS


def test_causal_attention_matches_upstream_golden(native_model):
    batch, length = TENSORS["input.attention_hidden"].shape[:2]
    hidden = TENSORS["input.attention_hidden"]
    positions = torch.arange(length)
    rope = rope_table(CONFIG["max_seq_len"], CONFIG["head_dim"], CONFIG["rope_base"], hidden.device)
    rope = rope[positions][None].expand(batch, -1, -1, -1, -1)
    mask = (positions[None, :] <= positions[:, None])[None, None].expand(batch, 1, length, length)
    layer = native_model.layers[0]

    actual = layer.attention(layer.attention_norm(hidden), rope, mask)

    torch.testing.assert_close(actual, TENSORS["expected.attention_output"], atol=2e-5, rtol=2e-5)


def test_prompt_prefill_matches_semantic_slice_golden(native_model):
    prompt = TENSORS["input.prompt"]
    attention_mask = TENSORS["input.attention_mask"]
    batch, _, length = prompt.shape
    positions = attention_mask.cumsum(-1).sub(1).clamp_min(0)
    rope = rope_table(CONFIG["max_seq_len"], CONFIG["head_dim"], CONFIG["rope_base"], prompt.device)[positions]
    rows = torch.arange(length)
    mask = (rows[None, :] <= rows[:, None])[None, None].expand(batch, 1, length, length)
    caches = [
        KVCache.create(batch, CONFIG["n_local_heads"], length, CONFIG["head_dim"], prompt.device, torch.float32)
        for _ in native_model.layers
    ]
    output_weight = torch.cat(
        (
            native_model.embeddings.weight[CONFIG["semantic_begin_id"]:CONFIG["semantic_end_id"] + 1],
            native_model.embeddings.weight[CONFIG["eos_token_id"]:CONFIG["eos_token_id"] + 1],
        ),
        dim=0,
    ).contiguous()

    actual, _ = native_model.slow_step(prompt, rope, mask, caches, 0, length, output_weight)

    torch.testing.assert_close(actual, TENSORS["expected.prefill_logits"], atol=2e-4, rtol=2e-4)


def test_fast_cache_codebooks_match_golden(native_model):
    hidden = TENSORS["input.fast_hidden"]
    caches = [
        KVCache.create(
            hidden.shape[0],
            CONFIG["fast_n_local_heads"],
            CONFIG["num_codebooks"],
            CONFIG["fast_head_dim"],
            hidden.device,
            hidden.dtype,
        )
        for _ in native_model.fast_layers
    ]
    rope = rope_table(CONFIG["num_codebooks"], CONFIG["fast_head_dim"], CONFIG["rope_base"], hidden.device)

    actual = native_model.codebooks(
        hidden,
        TENSORS["input.semantic"] - CONFIG["semantic_begin_id"],
        rope,
        caches,
        top_k=5,
        top_p=0.8,
        temperature=0.7,
        do_sample=False,
        generator=torch.Generator().manual_seed(17),
    )

    torch.testing.assert_close(actual, TENSORS["expected.fast_codes"])


def test_kitchen_rope_matches_upstream_golden(native_model):
    positions = TENSORS["input.rope_positions"]
    query = TENSORS["input.rope_query"]
    key = TENSORS["input.rope_key"]
    table = rope_table(10, CONFIG["head_dim"], CONFIG["rope_base"], query.device)
    frequencies = table[positions][:, :, None]

    actual_query, actual_key = comfy_kitchen.apply_rope(query.clone(), key.clone(), frequencies)

    torch.testing.assert_close(actual_query, TENSORS["expected.rope_query"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_key, TENSORS["expected.rope_key"], atol=1e-6, rtol=1e-6)


def test_top_k_top_p_filtering_matches_upstream_golden():
    actual = filtered_scores(TENSORS["input.filter_logits"], top_k=4, top_p=0.82, temperature=0.65)

    torch.testing.assert_close(actual, TENSORS["expected.filtered_scores"])
    assert torch.isfinite(actual).sum(-1).max().item() <= 4
