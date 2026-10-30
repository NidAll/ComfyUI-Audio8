"""Offline, single-file Whisper transcription with ComfyUI model residency."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import torch
import torchaudio.functional as AF
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer


FORMAT = 'audio8.whisper.v1'
REPO = 'openai/whisper-base'
REVISION = 'e37978b90ca9030d5170a5c07aadb050351a65bb'


def export_asr(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((source / 'config.json').read_text())
    if config.get('model_type') != 'whisper':
        raise ValueError('Expected an official Whisper checkpoint.')
    state = {key: value.half() if value.is_floating_point() else value
             for key, value in load_file(source / 'model.safetensors').items()}
    tokenizer = (source / 'tokenizer.json').read_bytes()
    state['_assets.tokenizer_json'] = torch.tensor(list(tokenizer), dtype=torch.uint8)
    metadata = {'format': FORMAT, 'config': json.dumps(config),
                'generation_config': (source / 'generation_config.json').read_text(),
                'feature_extractor': (source / 'preprocessor_config.json').read_text(),
                'tokenizer_sha256': hashlib.sha256(tokenizer).hexdigest(),
                'source': REPO, 'revision': REVISION}
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.whisper-', suffix='.safetensors', dir=output.parent)
    os.close(fd)
    try:
        save_file(state, temporary, metadata=metadata)
        with safe_open(temporary, framework='pt') as handle:
            if handle.metadata()['format'] != FORMAT or 'model.encoder.conv1.weight' not in handle.keys():
                raise ValueError('Incomplete ASR artifact.')
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'file': output.name, 'bytes': output.stat().st_size, 'source': REPO, 'revision': REVISION}


@dataclass
class ASR:
    patcher: object
    extractor: object
    tokenizer: object
    lock: object

    def get_models(self):
        return [self.patcher]


class WhisperModule(torch.nn.Module):
    # Comfy owns a writable device field; Transformers exposes a read-only one.
    def __init__(self, network):
        super().__init__()
        self.network = network

    def get_dtype(self):
        return next(self.network.parameters()).dtype


def load_asr(path, device_name='auto'):
    # Optional backend: the TTS/quantizer paths do not require Transformers.
    try:
        from transformers import GenerationConfig, WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration
    except ImportError as error:
        raise ImportError('Audio8 ASR needs Transformers: install this pack with its [asr] extra.') from error
    import comfy.model_management as mm
    from comfy.model_patcher import ModelPatcher

    if device_name not in ('auto', 'cpu'):
        raise ValueError('ASR device must be auto or cpu.')
    with safe_open(path, framework='pt') as handle:
        metadata = handle.metadata() or {}
        if metadata.get('format') != FORMAT:
            raise ValueError('Select an Audio8 Whisper bundle exported with tools/prepare_asr.py.')
        data = handle.get_tensor('_assets.tokenizer_json').numpy().tobytes()
        if hashlib.sha256(data).hexdigest() != metadata['tokenizer_sha256']:
            raise ValueError('ASR tokenizer integrity check failed.')
        device = torch.device('cpu') if device_name == 'cpu' else mm.get_torch_device()
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        config = WhisperConfig.from_dict(json.loads(metadata['config']))
        config._attn_implementation = 'sdpa'
        with torch.device('meta'):
            model = WhisperForConditionalGeneration(config)
        state = {key: handle.get_tensor(key).to(dtype=dtype) for key in handle.keys()
                 if not key.startswith('_assets.')}
    if 'proj_out.weight' not in state:
        state['proj_out.weight'] = state['model.decoder.embed_tokens.weight']
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.eval()
    model.generation_config = GenerationConfig.from_dict(json.loads(metadata['generation_config']))
    patcher = ModelPatcher(WhisperModule(model), load_device=device, offload_device=torch.device('cpu'))
    extractor = WhisperFeatureExtractor.from_dict(json.loads(metadata['feature_extractor']))
    return ASR(patcher, extractor, Tokenizer.from_str(data.decode('utf-8')), threading.RLock())


def transcribe(asr, audio, language='auto'):
    import comfy.model_management as mm
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Interrupt(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            mm.throw_exception_if_processing_interrupted()
            return False

    wave = audio['waveform']
    rate = int(audio['sample_rate'])
    if wave.ndim != 3 or wave.shape[0] != 1 or not wave.shape[-1] or rate <= 0:
        raise ValueError('ASR requires one nonempty AUDIO recording with a positive sample rate.')
    wave = wave[0].float().mean(0).cpu()
    if not torch.isfinite(wave).all():
        raise ValueError('Reference audio contains nonfinite samples.')
    if rate != 16000:
        wave = AF.resample(wave, rate, 16000)
    started = time.perf_counter()
    with asr.lock:
        mm.throw_exception_if_processing_interrupted()
        inputs = asr.extractor(wave.numpy(), sampling_rate=16000, return_tensors='pt',
                               return_attention_mask=True, truncation=False)
        model = asr.patcher.model.network
        device = asr.patcher.load_device
        try:
            mm.load_models_gpu([asr.patcher], memory_required=256 * 1024**2, force_full_load=True)
            ids = model.generate(inputs.input_features.to(device=device, dtype=next(model.parameters()).dtype),
                                 attention_mask=inputs.attention_mask.to(device),
                                 language=None if language.strip().lower() == 'auto' else language.strip().lower(),
                                 task='transcribe', do_sample=False, num_beams=1,
                                 return_timestamps=wave.numel() > 30 * 16000,
                                 stopping_criteria=StoppingCriteriaList([Interrupt()]))
            # Whisper timestamp IDs are not marked special in every tokenizer JSON.
            timestamp_begin = model.generation_config.no_timestamps_token_id + 1
            tokens = [token for token in ids[0].cpu().tolist() if token < timestamp_begin]
            text = asr.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        finally:
            # Only this ASR handle is offloaded; leave room for the TTS codec/transformer.
            mm.unload_model_and_clones(asr.patcher)
    if not text:
        raise ValueError('No speech was transcribed. Use a clear reference or enter its transcript manually.')
    return text, {'asr_seconds': time.perf_counter() - started, 'audio_seconds': wave.numel() / 16000,
                  'language': language, 'model': REPO, 'device': str(device)}
