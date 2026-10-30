"""Versioned, non-executable single-file format and offline exporter."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor import TensorWiseINT8Layout


FORMAT = 'audio8.singlefile.v1'
REPO = 'Edge0/Audio8-TTS-Realistic-0.6b'
REVISION = '1e857568bfd4f02931aca3988ee773a44d2d26d3'
ASSET = '_assets.tokenizer_json'


def inspect(path):
    with safe_open(str(path), framework='pt', device='cpu') as file:
        meta = file.metadata() or {}
        if meta.get('audio8.format') != FORMAT:
            raise ValueError('Not an Audio8 single-file artifact; run tools/quantize.py first.')
        config = json.loads(meta['audio8.config'])
        if config.get('model_type') != 'arktts':
            raise ValueError('Unsupported Audio8 architecture.')
        if ASSET not in file.keys() or not any(k.startswith('codec.') for k in file.keys()):
            raise ValueError('Incomplete Audio8 artifact: tokenizer and codec are required.')
        return config, meta


def read_component(path, prefix, dtype=None):
    state = {}
    with safe_open(str(path), framework='pt', device='cpu') as file:
        for key in file.keys():
            if key.startswith(prefix):
                value = file.get_tensor(key)
                if dtype is not None and value.is_floating_point() and not key.endswith('.weight_scale'):
                    value = value.to(dtype)
                state[key[len(prefix):]] = value
    return state


def dequantize_state(state, dtype):
    """Materialize CPU fallback weights once, rather than on every token."""
    state = dict(state)
    for key in list(state):
        if not key.endswith('.comfy_quant'):
            continue
        prefix = key.removesuffix('.comfy_quant')
        config = json.loads(state.pop(key).numpy().tobytes())
        if config['format'] != 'int8_tensorwise':
            raise ValueError('Audio8 CPU fallback expects native INT8 weights.')
        weight = state[prefix + '.weight']
        params = TensorWiseINT8Layout.Params(
            scale=state.pop(prefix + '.weight_scale'), orig_dtype=dtype,
            orig_shape=tuple(weight.shape), convrot=config.get('convrot', False),
            convrot_groupsize=config.get('convrot_groupsize', 256),
        )
        state[prefix + '.weight'] = TensorWiseINT8Layout.dequantize(weight, params)
    return state


def tokenizer_bytes(path):
    with safe_open(str(path), framework='pt', device='cpu') as file:
        asset = file.get_tensor(ASSET)
        if asset.dtype != torch.uint8 or asset.ndim != 1 or asset.numel() > 64 * 1024**2:
            raise ValueError('Invalid embedded tokenizer asset.')
        data = asset.numpy().tobytes()
        expected = (file.metadata() or {})['audio8.tokenizer_sha256']
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Embedded tokenizer checksum mismatch.')
        return data


def eligible(key, value, profile):
    branches = ('layers.', 'fast_layers.') if profile == 'int8-all' else ('layers.',)
    return (profile != 'bf16' and key.startswith(branches) and value.ndim == 2
            and key.endswith(('.attention.wqkv.weight', '.attention.wo.weight',
                              '.feed_forward.w1.weight', '.feed_forward.w2.weight',
                              '.feed_forward.w3.weight')) and value.shape[1] % 64 == 0)


def export(source, output, *, profile='int8', codec_dtype='bf16', dry_run=False, progress=None):
    """Convert a local official snapshot, or requantize an existing float bundle.

    Only the explicit converter reads upstream's codec pickle, using the
    restricted weights_only loader. Runtime never reads pickle or Python assets.
    """
    source, output = Path(source), Path(output)
    if profile not in ('bf16', 'int8', 'int8-all') or codec_dtype not in ('bf16', 'fp32'):
        raise ValueError('Unknown precision profile.')
    if output.suffix != '.safetensors':
        raise ValueError('Output must end in .safetensors.')
    if output.exists() or source.resolve() == output.resolve():
        raise FileExistsError(f'Refusing to replace {output}.')
    bundled = source.is_file()
    if bundled:
        config, old_meta = inspect(source)
        tokenizer = tokenizer_bytes(source)
        main = read_component(source, 'model.')
        if any(k.endswith('.comfy_quant') for k in main):
            raise ValueError('Requantization requires a floating-point source bundle.')
    else:
        config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
        if config.get('model_type') != 'arktts':
            raise ValueError('Expected the official arktts model snapshot.')
        tokenizer = (source / 'tokenizer.json').read_bytes()
        main = read_component(source / 'model.safetensors', '')
    selected = [k for k, v in main.items() if eligible(k, v, profile)]
    plan = {'profile': profile, 'codec_dtype': codec_dtype, 'quantized_layers': len(selected),
            'quantized_parameters': sum(main[k].numel() for k in selected), 'layers': selected}
    if dry_run:
        return plan
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    max_relative_error = 0.0
    for index, (key, value) in enumerate(main.items()):
        if key in selected:
            # Native Kitchen rotation and quantization; no second quantizer or
            # custom packing contract. K=896 needs a group dividing 896: 64.
            quantized, params = TensorWiseINT8Layout.quantize(
                value.float(), per_channel=True, convrot=True, convrot_groupsize=64,
            )
            reconstructed = TensorWiseINT8Layout.dequantize(quantized, params)
            error = ((reconstructed.float() - value.float()).square().sum()
                     / value.float().square().sum().clamp_min(1e-30)).sqrt().item()
            max_relative_error = max(max_relative_error, error)
            if not torch.isfinite(reconstructed).all() or error > 0.05:
                raise ValueError(f'Quantization reconstruction failed for {key}: {error:.6f}.')
            prefix = 'model.' + key.removesuffix('.weight')
            state[prefix + '.weight'] = quantized
            state[prefix + '.weight_scale'] = params.scale
            quant = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 64}
            state[prefix + '.comfy_quant'] = torch.tensor(list(json.dumps(quant).encode()), dtype=torch.uint8)
        else:
            state['model.' + key] = value.to(torch.bfloat16) if value.is_floating_point() else value
        if progress is not None:
            progress(index + 1, len(main))
    del main
    if bundled:
        codec = read_component(source, 'codec.')
    else:
        raw = torch.load(source / config['codec_filename'], map_location='cpu', weights_only=True)
        raw = raw.get('state_dict', raw)
        if any('generator.' in k for k in raw):
            raw = {k.replace('generator.', ''): v for k, v in raw.items() if 'generator.' in k}
        codec = {k: v for k, v in raw.items() if not k.endswith(('freqs_cis', 'causal_mask'))}
    dtype = torch.bfloat16 if codec_dtype == 'bf16' else torch.float32
    for key, value in codec.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'Non-tensor codec state: {key}.')
        state['codec.' + key] = value.to(dtype) if value.is_floating_point() else value
    state[ASSET] = torch.frombuffer(bytearray(tokenizer), dtype=torch.uint8)
    plan['max_weight_relative_l2'] = max_relative_error
    metadata = {'audio8.format': FORMAT, 'audio8.config': json.dumps(config),
                'audio8.source': REPO, 'audio8.revision': REVISION,
                'audio8.profile': profile, 'audio8.codec_dtype': codec_dtype,
                'audio8.tokenizer_sha256': hashlib.sha256(tokenizer).hexdigest(),
                'audio8.quantization': json.dumps(plan)}
    fd, temporary = tempfile.mkstemp(prefix='.' + output.name + '.', suffix='.tmp', dir=output.parent)
    os.close(fd)
    try:
        save_file({k: v.contiguous() for k, v in state.items()}, temporary, metadata=metadata)
        inspect(temporary)
        tokenizer_bytes(temporary)
        # Hard-link commit is atomic and never overwrites an existing result,
        # including a file created concurrently while conversion was running.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return plan
