import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

COMFYUI = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMFYUI))
sys.path.insert(0, str(PACKAGE))

from comfy.cli_args import args

args.cpu = True

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.nn.utils import weight_norm as legacy_weight_norm
from torch.nn.utils.parametrizations import weight_norm as parametrized_weight_norm

from audio8.codec import (
    ArkttsCausalConv1d,
    ArkttsCodecAttention,
    ArkttsCodecRMSNorm,
    ArkttsCodecTransformerConfig,
    ArkttsCodecWindowTransformer,
    _load_codec_state_dict,
    _normalize_codec_state_dict,
    _remove_weight_norm,
)
from audio8.model import rope_table


def _reference_rope(x, values):
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    values = values.view(1, shaped.shape[1], 1, shaped.shape[3], 2)
    rotated = torch.stack(
        (
            shaped[..., 0] * values[..., 0] - shaped[..., 1] * values[..., 1],
            shaped[..., 1] * values[..., 0] + shaped[..., 0] * values[..., 1],
        ),
        dim=-1,
    )
    return rotated.flatten(3).to(x.dtype)


def test_codec_rms_norm_matches_float32_reference():
    norm = ArkttsCodecRMSNorm(4, eps=1e-5)
    norm.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.25, 0.5, 1.0, 2.0]])

    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight

    torch.testing.assert_close(norm(x), expected)


def test_codec_attention_matches_sdpa_with_local_kv_heads():
    config = ArkttsCodecTransformerConfig(
        n_layer=1,
        n_head=2,
        n_local_heads=1,
        dim=8,
        intermediate_size=16,
        head_dim=4,
        channels_first=False,
    )
    attention = ArkttsCodecAttention(config)
    for index, parameter in enumerate(attention.parameters()):
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        parameter.data.copy_((values + index + 1) / 100)
    x = torch.linspace(-0.5, 0.5, 2 * 5 * 8).reshape(2, 5, 8)
    frequencies = 1.0 / (
        config.rope_base ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
    )
    phases = torch.outer(torch.arange(x.shape[1]), frequencies)
    polar = torch.polar(torch.ones_like(phases), phases)
    rope_values = torch.stack((polar.real, polar.imag), dim=-1).to(torch.bfloat16)
    kitchen_rope = rope_table(x.shape[1], config.head_dim, config.rope_base, x.device)
    positions = torch.arange(x.shape[1])
    mask = (positions[None, :] <= positions[:, None])[None, None]

    actual = attention(x, kitchen_rope, mask)

    query_size = config.n_head * config.head_dim
    kv_size = config.n_local_heads * config.head_dim
    query, key, value = attention.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
    query = _reference_rope(query.view(2, 5, 2, 4), rope_values).transpose(1, 2)
    key = _reference_rope(key.view(2, 5, 1, 4), rope_values).transpose(1, 2)
    value = value.view(2, 5, 1, 4).transpose(1, 2)
    key = key.repeat_interleave(2, dim=1)
    value = value.repeat_interleave(2, dim=1)
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    expected = attention.wo(expected.transpose(1, 2).contiguous().view(2, 5, query_size))

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_small_window_transformer_is_causal_and_preserves_state_keys():
    config = ArkttsCodecTransformerConfig(
        n_layer=1,
        n_head=2,
        n_local_heads=1,
        dim=8,
        intermediate_size=16,
        head_dim=4,
        channels_first=False,
    )
    transformer = ArkttsCodecWindowTransformer(config, 8, window_size=3)
    for index, parameter in enumerate(transformer.parameters()):
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        parameter.data.copy_((values.remainder(11) + index + 1) / 20)
    x = torch.randn(1, 5, 8)
    changed = x.clone()
    changed[:, 4] += 10

    output = transformer(x)
    changed_output = transformer(changed)

    assert output.shape == x.shape
    torch.testing.assert_close(output[:, :4], changed_output[:, :4], atol=1e-6, rtol=1e-6)
    keys = transformer.state_dict()
    assert "layers.0.attention.wqkv.weight" in keys
    assert "layers.0.attention_norm.weight" in keys
    assert "layers.0.feed_forward.w1.weight" in keys


def test_codec_checkpoint_normalization_matches_upstream_loader():
    weights = torch.ones(2)
    cache = torch.ones(1)
    state = {
        "state_dict": {
            "generator.encoder.weight": weights,
            "generator.pre_module.freqs_cis": cache,
            "other.discriminator.weight": weights,
        }
    }

    normalized = _normalize_codec_state_dict(state)

    assert list(normalized) == ["encoder.weight"]
    assert normalized["encoder.weight"] is weights


def test_weight_norm_is_removed_without_changing_convolution_output():
    modern = ArkttsCausalConv1d(2, 3, kernel_size=3).apply_weight_norm()
    modern.conv.parametrizations.weight.original0.data.fill_(0.7)
    modern.conv.parametrizations.weight.original1.data.copy_(
        torch.linspace(-0.5, 0.5, modern.conv.parametrizations.weight.original1.numel()).reshape(
            modern.conv.parametrizations.weight.original1.shape
        )
    )
    legacy = legacy_weight_norm(nn.Conv1d(2, 3, kernel_size=3))
    legacy.weight_g.data.fill_(0.8)
    legacy.weight_v.data.copy_(torch.linspace(-0.25, 0.25, legacy.weight_v.numel()).reshape(legacy.weight_v.shape))
    model = nn.ModuleList([modern, legacy])
    x = torch.randn(1, 2, 8)
    before = (modern(x), legacy(x))

    _remove_weight_norm(model)

    after = (modern(x), legacy(x))
    torch.testing.assert_close(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])
    assert not hasattr(modern.conv, "parametrizations")
    assert not hasattr(legacy, "weight_g")


def test_meta_codec_load_assign_preserves_tiny_checkpoint_state():
    class TinyCodec(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.norm = ArkttsCodecRMSNorm(4, eps=1e-5)
            self.modern = parametrized_weight_norm(nn.Conv1d(2, 2, kernel_size=3, padding=1))
            self.legacy = legacy_weight_norm(nn.Conv1d(2, 2, kernel_size=1))

        def forward(self, x, audio):
            return self.norm(self.linear(x)), self.modern(audio) + self.legacy(audio)

    torch.manual_seed(9)
    original = TinyCodec()
    checkpoint = {
        "state_dict": {
            f"generator.{key}": value.to(torch.bfloat16) if value.is_floating_point() else value
            for key, value in original.state_dict().items()
        }
    }
    expected = TinyCodec()
    expected.load_state_dict(
        {
            key.removeprefix("generator."): value.to(torch.float32) if value.is_floating_point() else value
            for key, value in checkpoint["state_dict"].items()
        },
        strict=True,
    )
    with torch.device("meta"):
        actual = TinyCodec()
    assert all(parameter.is_meta for parameter in actual.parameters())

    _load_codec_state_dict(actual, checkpoint)

    assert not any(parameter.is_meta for parameter in actual.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in actual.parameters())
    assert not parametrize.is_parametrized(actual.modern, "weight")
    assert not hasattr(actual.legacy, "weight_g")
    assert not hasattr(actual.modern, "parametrizations")
    x = torch.randn(2, 4)
    audio = torch.randn(2, 2, 8)
    expected_output = expected(x, audio)
    actual_output = actual(x, audio)
    torch.testing.assert_close(actual_output[0], expected_output[0])
    torch.testing.assert_close(actual_output[1], expected_output[1])


def test_weight_norm_folding_uses_execution_precision():
    expected = nn.ModuleList([
        parametrized_weight_norm(nn.Conv1d(2, 3, 3)),
        legacy_weight_norm(nn.Conv1d(2, 3, 3)),
    ]).to(dtype=torch.bfloat16)
    checkpoint = expected.state_dict()
    with torch.device('meta'):
        actual = nn.ModuleList([
            parametrized_weight_norm(nn.Conv1d(2, 3, 3)),
            legacy_weight_norm(nn.Conv1d(2, 3, 3)),
        ])
    _load_codec_state_dict(actual, checkpoint, dtype=torch.bfloat16)
    x = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(after(x), before(x), atol=0, rtol=0)
        assert after.weight.dtype == torch.bfloat16
