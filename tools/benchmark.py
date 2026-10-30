#!/usr/bin/env python3
"""Synchronized original-versus-bundle benchmark; all inputs must be local."""
import argparse
import importlib.util
import importlib.metadata
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1])]

import soundfile as sf
import torch
from safetensors.torch import load_file


def original_model(source, dtype, device):
    package = types.ModuleType('audio8_original')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    for name in ['configuration_arktts', 'modeling_arktts_codec', 'modeling_arktts']:
        spec = importlib.util.spec_from_file_location('audio8_original.' + name, source / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    c = sys.modules['audio8_original.configuration_arktts'].ArkttsConfig(**json.loads((source / 'config.json').read_text()))
    c._name_or_path = str(source)
    with torch.device('meta'):
        model = sys.modules['audio8_original.modeling_arktts'].ArkttsModel(c)
    model.load_state_dict(load_file(source / 'model.safetensors'), strict=True, assign=True)
    # Non-persistent RoPE buffers were created on meta and are not checkpoint-owned.
    precompute = sys.modules['audio8_original.modeling_arktts']._precompute_rope
    model.freqs_cis = precompute(c.max_seq_len, c.head_dim, c.rope_base)
    model.fast_freqs_cis = precompute(c.num_codebooks, c.fast_head_dim, c.rope_base)
    return model.to(device=device, dtype=dtype).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--original', action='store_true', help='Use reviewed official source from the model directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--text', default='Hello! Welcome to Audio eight. It is wonderful to hear from you.')
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--reference')
    parser.add_argument('--transcript', default='')
    parser.add_argument('--codec-device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--acceleration', choices=['auto', 'eager'], default='auto')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    started = time.perf_counter()
    source = Path(args.model).resolve()
    from audio8.runtime import load_bundle, synthesize, encode_reference
    from audio8.prompt import build_prompt, load_tokenizer
    if args.original:
        model = original_model(source, dtype, device)
        config = model.config.to_dict()
        tokenizer = load_tokenizer((source / 'tokenizer.json').read_bytes())
        bundle = None
    else:
        bundle = load_bundle(str(source), codec_device=args.codec_device)
        model = bundle.patcher.model
    loaded = time.perf_counter() - started
    reference_started = time.perf_counter()
    reference = None
    if args.reference:
        values, sr = sf.read(args.reference, dtype='float32', always_2d=True)
        audio = {'waveform': torch.from_numpy(values.T)[None], 'sample_rate': sr}
        with torch.inference_mode():
            if bundle:
                reference = encode_reference(bundle, audio, args.transcript)
            else:
                from torchaudio.functional import resample
                wave = audio['waveform'].mean(1, keepdim=True)
                wave = resample(wave, sr, config['codec_sample_rate']) if sr != config['codec_sample_rate'] else wave
                codes, length = model.encode_audio(wave.to(device))
                reference = {'codes': codes[0, :, :int(length[0])].cpu(), 'text': args.transcript}
    reference_seconds = time.perf_counter() - reference_started if args.reference else 0.0
    runs = []
    for index in range(args.repeats):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            if bundle:
                audio, stats = synthesize(bundle, [args.text], reference=reference, seed=args.seed,
                                           max_new_tokens=args.frames, do_sample=args.sample,
                                           acceleration=args.acceleration)
            else:
                prompt, mask = build_prompt(tokenizer, config, [args.text],
                                             reference['codes'] if reference else None,
                                             reference['text'] if reference else '')
                generated = model.generate(input_ids=prompt.to(device), attention_mask=mask.to(device),
                                           max_new_tokens=args.frames, do_sample=args.sample,
                                           temperature=0.8, top_p=0.95, top_k=50,
                                           generator=torch.Generator(device=device).manual_seed(args.seed),
                                           return_dict_in_generate=True)
                torch.cuda.synchronize(device) if device.type == 'cuda' else None
                ar_elapsed = time.perf_counter() - started
                waveforms, lengths = model.decode_audio(generated.codes)
                audio = {'waveform': waveforms[:, None].float().cpu(), 'sample_rate': config['codec_sample_rate']}
                stats = {'generation_seconds': ar_elapsed, 'frames': generated.code_lengths.cpu().tolist(),
                         'finished': generated.finished.cpu().tolist(),
                         'audio_seconds': [int(lengths[0]) / audio['sample_rate']]}
        torch.cuda.synchronize(device) if device.type == 'cuda' else None
        stats['wall_seconds'] = time.perf_counter() - started
        stats['rtf'] = stats['wall_seconds'] / max(stats['audio_seconds'][0], 1e-9)
        if device.type == 'cuda':
            stats['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            stats['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        runs.append(stats)
        sf.write(output / f'run-{index}.wav', audio['waveform'][0].T.numpy(), audio['sample_rate'])
        print(json.dumps({'run': index, **stats}), flush=True)
    report = {'original': args.original, 'load_seconds': loaded, 'torch': torch.__version__,
              'reference_encode_seconds': reference_seconds,
              'cold_total_seconds': loaded + reference_seconds + runs[0]['wall_seconds'],
              'dependencies': {name: importlib.metadata.version(name) for name in ['comfy-kitchen', 'tokenizers', 'safetensors']},
              'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
              'settings': vars(args), 'runs': runs,
              'warm_median_seconds': statistics.median(r['wall_seconds'] for r in runs[1:] or runs)}
    digest = hashlib.sha256()
    for file in sorted((ROOT / 'audio8').glob('*.py')):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    report['runtime_source_sha256'] = digest.hexdigest()
    (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.profile and bundle:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
            with torch.inference_mode():
                synthesize(bundle, [args.text], reference=reference, seed=args.seed,
                           max_new_tokens=min(args.frames, 16), do_sample=args.sample, acceleration='eager')
        (output / 'profile.txt').write_text(profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
        profile.export_chrome_trace(str(output / 'trace.json'))


if __name__ == '__main__':
    main()
