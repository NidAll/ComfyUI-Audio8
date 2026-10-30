import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from audio8.artifact import FORMAT, ASSET, inspect, tokenizer_bytes, read_component, eligible, export


def fixture_bundle(path, token=b'{"version":"1.0"}', codec=True):
    state = {ASSET: torch.tensor(list(token), dtype=torch.uint8),
             'model.layers.0.attention.wo.weight': torch.eye(64),
             'model.norm.weight': torch.ones(64)}
    if codec:
        state['codec.test.weight'] = torch.ones(2)
    meta = {'audio8.format': FORMAT, 'audio8.config': json.dumps({'model_type': 'arktts'}),
            'audio8.tokenizer_sha256': hashlib.sha256(token).hexdigest()}
    save_file(state, str(path), metadata=meta)


def test_format_requires_codec_and_tokenizer(tmp_path):
    file = tmp_path / 'bad.safetensors'
    fixture_bundle(file, codec=False)
    with pytest.raises(ValueError, match='Incomplete'):
        inspect(file)


def test_asset_integrity_and_native_scale_precision(tmp_path):
    file = tmp_path / 'fixture.safetensors'
    fixture_bundle(file)
    assert tokenizer_bytes(file) == b'{"version":"1.0"}'
    state = read_component(file, 'model.', torch.bfloat16)
    assert state['norm.weight'].dtype == torch.bfloat16
    corrupt = tmp_path / 'corrupt.safetensors'
    save_file({ASSET: torch.tensor([1, 2], dtype=torch.uint8)}, str(corrupt),
              metadata={'audio8.tokenizer_sha256': 'invalid'})
    with pytest.raises(ValueError, match='checksum'):
        tokenizer_bytes(corrupt)


def test_protected_layers_are_never_quantized():
    w = torch.ones(64, 64)
    assert eligible('layers.0.attention.wo.weight', w, 'int8')
    assert not eligible('fast_layers.0.attention.wo.weight', w, 'int8')
    assert eligible('fast_layers.0.attention.wo.weight', w, 'int8-all')
    for name in ['embeddings.weight', 'fast_output.weight', 'norm.weight', 'codec.test.weight']:
        assert not eligible(name, w, 'int8-all')
    assert not eligible('layers.0.attention.wo.weight', torch.ones(64, 63), 'int8')


def test_export_is_atomic_and_preserves_source(tmp_path):
    source, target = tmp_path / 'source.safetensors', tmp_path / 'target.safetensors'
    fixture_bundle(source)
    before = source.read_bytes()
    plan = export(source, target, dry_run=True)
    assert plan['quantized_layers'] == 1
    assert not target.exists()
    result = export(source, target)
    assert result['max_weight_relative_l2'] < 0.02
    assert source.read_bytes() == before
    assert read_component(target, 'model.')['layers.0.attention.wo.weight'].dtype == torch.int8
    assert read_component(target, 'model.', torch.bfloat16)['layers.0.attention.wo.weight_scale'].dtype == torch.float32
    assert tokenizer_bytes(target) == b'{"version":"1.0"}'
    with pytest.raises(FileExistsError):
        export(source, target)
    assert not list(tmp_path.glob('*.tmp'))


def test_cpu_fallback_materializes_native_rotation_once(tmp_path):
    from audio8.artifact import dequantize_state
    source, target = tmp_path / 'float.safetensors', tmp_path / 'int8.safetensors'
    fixture_bundle(source)
    export(source, target)
    original = read_component(target, 'model.')
    state = dequantize_state(original, torch.float32)
    assert all(not k.endswith(('.comfy_quant', '.weight_scale')) for k in state)
    torch.testing.assert_close(state['layers.0.attention.wo.weight'], torch.eye(64), atol=0.01, rtol=0.01)
    assert any(k.endswith('.comfy_quant') for k in original)
