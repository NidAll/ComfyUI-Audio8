import json
import socket
from types import SimpleNamespace
import threading

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models

from audio8.asr import ASR, WhisperModule, export_asr, load_asr, transcribe


@pytest.fixture
def tiny_bundle(tmp_path):
    transformers = pytest.importorskip('transformers')
    source = tmp_path / 'source'
    source.mkdir()
    config = transformers.WhisperConfig(d_model=8, encoder_layers=1, decoder_layers=1,
                                         encoder_attention_heads=2, decoder_attention_heads=2,
                                         encoder_ffn_dim=16, decoder_ffn_dim=16, vocab_size=32,
                                         pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                         decoder_start_token_id=1, max_source_positions=10)
    model = transformers.WhisperForConditionalGeneration(config)
    state = {key: value.clone() for key, value in model.state_dict().items() if key != 'proj_out.weight'}
    save_file(state, source / 'model.safetensors')
    (source / 'config.json').write_text(config.to_json_string())
    (source / 'generation_config.json').write_text(model.generation_config.to_json_string())
    (source / 'preprocessor_config.json').write_text('{}')
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'hello': 1}, unk_token='[UNK]'))
    (source / 'tokenizer.json').write_text(tokenizer.to_str())
    output = tmp_path / 'asr.safetensors'
    export_asr(source, output)
    return source, output


def test_export_and_cpu_load_are_offline_and_do_not_overwrite(tiny_bundle, monkeypatch):
    from comfy.cli_args import args
    monkeypatch.setattr(args, 'cpu', True)
    source, output = tiny_bundle
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_asr(source, output)
    assert output.read_bytes() == before
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('ASR attempted network access'))
    asr = load_asr(str(output), 'cpu')
    model = asr.patcher.model.network
    assert next(model.parameters()).dtype == torch.float32
    assert model.proj_out.weight is model.model.decoder.embed_tokens.weight
    assert asr.tokenizer.decode([1]) == 'hello'
    # Comfy can assign residency fields without touching HF's read-only device property.
    asr.patcher.model.device = torch.device('cpu')
    assert asr.get_models() == [asr.patcher]


def test_transcription_offloads_on_generation_error(monkeypatch):
    pytest.importorskip('transformers')
    import comfy.model_management as mm
    calls = []

    class Failing(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def generate(self, *args, **kwargs):
            assert kwargs['language'] == 'ar'
            assert kwargs['task'] == 'transcribe'
            kwargs['stopping_criteria'][0](None, None)
            raise RuntimeError('decoder failed')

    patcher = SimpleNamespace(model=WhisperModule(Failing()), load_device=torch.device('cpu'))
    extractor = lambda *a, **k: SimpleNamespace(input_features=torch.zeros(1, 80, 3000),
                                               attention_mask=torch.ones(1, 3000))
    asr = ASR(patcher, extractor, None, threading.RLock())
    monkeypatch.setattr(mm, 'load_models_gpu', lambda *a, **k: calls.append('load'))
    monkeypatch.setattr(mm, 'unload_model_and_clones', lambda p: calls.append('unload'))
    monkeypatch.setattr(mm, 'throw_exception_if_processing_interrupted', lambda: calls.append('check'))
    with pytest.raises(RuntimeError, match='decoder failed'):
        transcribe(asr, {'waveform': torch.ones(1, 2, 3200), 'sample_rate': 16000}, ' ar ')
    assert calls == ['check', 'load', 'check', 'unload']


def test_transcription_removes_timestamp_ids_and_offloads(monkeypatch):
    pytest.importorskip('transformers')
    import comfy.model_management as mm

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.generation_config = SimpleNamespace(no_timestamps_token_id=50363)

        def generate(self, *args, **kwargs):
            return torch.tensor([[50364, 3, 50365, 4]])

    patcher = SimpleNamespace(model=WhisperModule(Decoder()), load_device=torch.device('cpu'))
    extractor = lambda *a, **k: SimpleNamespace(input_features=torch.zeros(1, 80, 3000),
                                               attention_mask=torch.ones(1, 3000))
    captured = []
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: captured.append(tokens) or 'hello world')
    asr = ASR(patcher, extractor, tokenizer, threading.RLock())
    monkeypatch.setattr(mm, 'load_models_gpu', lambda *a, **k: None)
    monkeypatch.setattr(mm, 'unload_model_and_clones', lambda p: captured.append('offloaded'))
    text, metrics = transcribe(asr, {'waveform': torch.ones(1, 1, 3200), 'sample_rate': 16000})
    assert text == 'hello world'
    assert captured == [[3, 4], 'offloaded']
    assert metrics['audio_seconds'] == 0.2
