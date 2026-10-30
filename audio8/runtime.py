"""ComfyUI residency, audio preparation and synthesis boundaries."""

from dataclasses import dataclass
import threading
import time

import torch
import torchaudio.functional as AF
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops

from . import artifact
from .codec import build_codec
from .model import Audio8Model
from .prompt import build_prompt, load_tokenizer


@dataclass
class Bundle:
    path: str
    config: dict
    patcher: object
    codec_patcher: object
    tokenizer: object
    dtype: torch.dtype
    codec_device: torch.device
    lock: object

    def get_models(self):
        return [self.patcher, self.codec_patcher]


def load_bundle(path, precision='auto', device_name='auto', codec_device='gpu'):
    if precision not in ('auto', 'bf16', 'fp16', 'fp32') or device_name not in ('auto', 'cpu'):
        raise ValueError('Unsupported precision or device.')
    if codec_device not in ('cpu', 'gpu'):
        raise ValueError('Codec device must be cpu or gpu.')
    config, _ = artifact.inspect(path)
    device = torch.device('cpu') if device_name == 'cpu' else mm.get_torch_device()
    dtypes = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    dtype = (torch.bfloat16 if device.type == 'cuda' and mm.should_use_bf16(device) else
             torch.float32) if precision == 'auto' else dtypes[precision]
    operations = comfy.ops.mixed_precision_ops(compute_dtype=dtype, disabled=comfy.ops.get_disabled_quant_formats(device))
    with torch.device('meta'):
        model = Audio8Model(config, operations)
    state = artifact.read_component(path, 'model.', dtype)
    if 'int8_tensorwise' in comfy.ops.get_disabled_quant_formats(device):
        state = artifact.dequantize_state(state, dtype)
    model.load_state_dict(state, strict=True, assign=True)
    model.fuse_projections()
    model.eval()
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=device, offload_device=torch.device('cpu'))
    codec_target = device if codec_device == 'gpu' else torch.device('cpu')
    codec_dtype = torch.float32 if codec_target.type == 'cpu' else dtype
    codec = build_codec(config, artifact.read_component(path, 'codec.'), dtype=codec_dtype)
    codec_patcher = comfy.model_patcher.ModelPatcher(codec, load_device=codec_target, offload_device=torch.device('cpu'))
    return Bundle(str(path), config, patcher, codec_patcher,
                  load_tokenizer(artifact.tokenizer_bytes(path)), dtype, codec_target, threading.RLock())


def codec_for(bundle):
    # Plain codec convolutions cannot be partially offloaded. The manager
    # reserves its full footprint and may evict the transformer between stages.
    mm.load_models_gpu([bundle.codec_patcher], memory_required=384 * 1024**2, force_full_load=True)
    return bundle.codec_patcher.model


def encode_reference(bundle, audio, transcript):
    if not transcript.strip():
        raise ValueError('Enter the exact spoken reference transcript.')
    waveform = audio['waveform']
    if waveform.ndim != 3 or waveform.shape[0] != 1 or not waveform.shape[-1]:
        raise ValueError('Reference AUDIO must contain one nonempty recording [1, channels, samples].')
    if not torch.isfinite(waveform).all():
        raise ValueError('Reference audio contains nonfinite samples.')
    waveform = waveform[0].float().mean(0, keepdim=True).cpu()
    rate = int(audio['sample_rate'])
    if rate <= 0:
        raise ValueError('Reference sample rate must be positive.')
    target = bundle.config['codec_sample_rate']
    if rate != target:
        waveform = AF.resample(waveform, rate, target)
    with bundle.lock:
        codec = codec_for(bundle)
        waveform = waveform.unsqueeze(0).to(device=bundle.codec_device, dtype=next(codec.parameters()).dtype)
        codes, lengths = codec.encode(waveform)
        return {'codes': codes[0, :, :int(lengths[0])].long().cpu().clone(),
                'text': transcript, 'source': artifact.REVISION}


def synthesize(bundle, texts, *, reference=None, seed=42, max_new_tokens=512,
               temperature=0.8, top_p=0.95, top_k=50, do_sample=True, callback=None, acceleration='auto'):
    with bundle.lock:
        device = bundle.patcher.load_device
        prompt, mask = build_prompt(bundle.tokenizer, bundle.config, texts,
                                    reference['codes'] if reference else None,
                                    reference['text'] if reference else '')
        if reference and reference.get('source') != artifact.REVISION:
            raise ValueError('Reference codes were produced for a different Audio8 codec.')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        reserve = 256 * 1024**2
        resident = mm.get_free_memory(device) + bundle.patcher.loaded_size() >= bundle.patcher.model_size() + reserve
        mm.load_models_gpu([bundle.patcher], memory_required=reserve, force_full_load=resident)
        if bundle.patcher.loaded_size() < bundle.patcher.model_size():
            acceleration = 'eager'
        codes, lengths, finished = bundle.patcher.model.generate(
            prompt.to(device), mask.to(device), seed=seed, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k, do_sample=do_sample, callback=callback,
            acceleration=acceleration,
        )
        codes = codes.cpu()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        generated = time.perf_counter()
        codec = codec_for(bundle)
        waves = []
        for index in range(len(texts)):
            mm.throw_exception_if_processing_interrupted()
            count = int(lengths[index])
            if count:
                wave = codec.decode(codes[index:index + 1, :, :count].to(bundle.codec_device))[0].float().cpu()
            else:
                wave = torch.empty(1, 0)
            if not torch.isfinite(wave).all():
                raise RuntimeError('Codec produced nonfinite audio; try FP32 precision.')
            waves.append(wave)
        duration = max((wave.shape[-1] for wave in waves), default=0)
        output = torch.zeros(len(waves), 1, duration)
        for index, wave in enumerate(waves):
            output[index, :, :wave.shape[-1]] = wave
        elapsed = time.perf_counter() - started
        sample_rate = bundle.config['codec_sample_rate']
        stats = {'generation_seconds': generated - started, 'decode_seconds': elapsed - (generated - started),
                 'total_seconds': elapsed, 'frames': lengths.cpu().tolist(), 'finished': finished.cpu().tolist(),
                 'audio_seconds': [wave.shape[-1] / sample_rate for wave in waves],
                 'samples': [wave.shape[-1] for wave in waves], 'seed': seed}
        return {'waveform': output, 'sample_rate': sample_rate}, stats
