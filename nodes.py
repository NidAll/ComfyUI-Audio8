import json
from functools import partial
from pathlib import Path

import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy_api.latest import io

from .audio8 import artifact
from .audio8.runtime import load_bundle, encode_reference, synthesize
from .audio8.asr import load_asr, transcribe
from .audio8.longform import synthesize_long


MODEL_TYPE = io.Custom('AUDIO8_MODEL')
VOICE_TYPE = io.Custom('AUDIO8_VOICE')
ASR_TYPE = io.Custom('AUDIO8_ASR')
FOLDER = 'audio8'
folder_paths.add_model_folder_path(FOLDER, str(Path(folder_paths.models_dir) / FOLDER))
ASR_FOLDER = 'audio8_asr'
folder_paths.add_model_folder_path(ASR_FOLDER, str(Path(folder_paths.models_dir) / ASR_FOLDER))


def resolve_model(name):
    if name not in folder_paths.get_filename_list(FOLDER) or Path(name).suffix != '.safetensors':
        raise ValueError('Select an Audio8 safetensors file from models/audio8.')
    return folder_paths.get_full_path_or_raise(FOLDER, name)


class Audio8Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = [n for n in folder_paths.get_filename_list(FOLDER) if n.endswith('.safetensors')]
        preferred = 'audio8-realistic-int8-all.safetensors'
        return io.Schema(node_id='Audio8Loader', display_name='Load Audio8', category='audio/Audio8',
                         description='Loads an offline single-file Audio8 model, codec and tokenizer.',
                         inputs=[io.Combo.Input('model_name', options=names,
                                                 default=preferred if preferred in names else (names[0] if names else None)),
                                 io.Combo.Input('precision', options=['auto', 'bf16', 'fp16', 'fp32'], default='auto'),
                                 io.Combo.Input('device', options=['auto', 'cpu'], default='auto'),
                                 io.Combo.Input('codec_device', options=['gpu', 'cpu'], default='gpu',
                                                tooltip='GPU gives faster encoding/decoding; CPU leaves more VRAM for generation.')],
                         outputs=[MODEL_TYPE.Output('model')])

    @classmethod
    def execute(cls, model_name, precision, device, codec_device):
        return io.NodeOutput(load_bundle(resolve_model(model_name), precision, device, codec_device))


class Audio8Voice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Audio8Voice', display_name='Audio8 Reference Voice', category='audio/Audio8',
                         description='Encodes one reference recording. Reuse this output for multiple synthesis nodes.',
                         inputs=[MODEL_TYPE.Input('model'), io.Audio.Input('audio'),
                                 io.String.Input('transcript', multiline=True, default='',
                                                 tooltip='Exact words spoken in the reference, in its original language.')],
                         outputs=[VOICE_TYPE.Output('voice')])

    @classmethod
    def execute(cls, model, audio, transcript):
        return io.NodeOutput(encode_reference(model, audio, transcript))


class Audio8Synthesize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Audio8Synthesize', display_name='Audio8 Speech', category='audio/Audio8',
                         description='Multilingual TTS or voice cloning. Language follows the text; no language token is needed.',
                         inputs=[MODEL_TYPE.Input('model'),
                                 io.String.Input('text', multiline=True, default='Hello! It is wonderful to hear from you.'),
                                 io.Int.Input('seed', default=42, min=0, max=0xffffffffffffffff, control_after_generate=True),
                                 io.Int.Input('max_new_tokens', default=512, min=1, max=2048),
                                 io.Float.Input('temperature', default=0.8, min=0.01, max=2.0, step=0.01),
                                 io.Float.Input('top_p', default=0.95, min=0.01, max=1.0, step=0.01),
                                 io.Int.Input('top_k', default=50, min=1, max=4097),
                                 io.Boolean.Input('do_sample', default=True),
                                 io.Combo.Input('acceleration', options=['auto', 'eager'], default='auto',
                                                tooltip='Auto uses execution-local CUDA graphs for both decoders when the transformer fits in VRAM.'),
                                 io.Boolean.Input('batch_lines', default=False,
                                                  tooltip='Generate each nonempty line as a separate batch item. Outputs are padded to equal length.'),
                                 VOICE_TYPE.Input('voice', optional=True)],
                         outputs=[io.Audio.Output('audio'), io.String.Output('metrics')])

    @classmethod
    def execute(cls, model, text, seed, max_new_tokens, temperature, top_p, top_k, do_sample, acceleration, batch_lines, voice=None):
        texts = [line.strip() for line in text.splitlines() if line.strip()] if batch_lines else [text]
        progress = comfy.utils.ProgressBar(max_new_tokens)

        def update(step, total):
            mm.throw_exception_if_processing_interrupted()
            progress.update_absolute(step, total)

        audio, metrics = synthesize(model, texts, reference=voice, seed=seed, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_p=top_p, top_k=top_k,
                                    do_sample=do_sample, callback=update, acceleration=acceleration)
        return io.NodeOutput(audio, json.dumps(metrics, indent=2))


class Audio8LongSpeech(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = Audio8Synthesize.define_schema()
        schema.node_id = 'Audio8LongSpeech'
        schema.display_name = 'Audio8 Long Speech'
        schema.description = 'Generate long text as complete sentences with the same reference voice, then join them into one recording.'
        schema.inputs = [item for item in schema.inputs if item.id not in ('batch_lines', 'voice')]
        next(item for item in schema.inputs if item.id == 'max_new_tokens').default = 768
        next(item for item in schema.inputs if item.id == 'do_sample').default = False
        schema.inputs.extend([
            io.Int.Input('chunk_chars', default=240, min=1, max=2000,
                         tooltip='Preferred text size per section. Sentence boundaries are preferred; unfinished sections are split and retried.'),
            io.Int.Input('pause_ms', default=100, min=0, max=2000),
            VOICE_TYPE.Input('voice', optional=True),
        ])
        return schema

    @classmethod
    def execute(cls, model, text, seed, max_new_tokens, temperature, top_p, top_k,
                do_sample, acceleration, chunk_chars, pause_ms, voice=None):
        progress = comfy.utils.ProgressBar(max_new_tokens)

        def update(step, total):
            mm.throw_exception_if_processing_interrupted()
            progress.update_absolute(step, total)

        audio, metrics = synthesize_long(partial(synthesize, model), text, chunk_chars=chunk_chars,
                                         pause_ms=pause_ms, seed=seed, max_new_tokens=max_new_tokens,
                                         reference=voice, temperature=temperature, top_p=top_p, top_k=top_k,
                                         do_sample=do_sample, acceleration=acceleration, callback=update)
        return io.NodeOutput(audio, json.dumps(metrics, indent=2))


class Audio8ASRLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = [name for name in folder_paths.get_filename_list(ASR_FOLDER) if name.endswith('.safetensors')]
        return io.Schema(node_id='Audio8ASRLoader', display_name='Load Audio8 ASR', category='audio/Audio8',
                         description='Load a local multilingual Whisper bundle. No automatic downloads.',
                         inputs=[io.Combo.Input('model_name', options=names),
                                 io.Combo.Input('device', options=['auto', 'cpu'], default='auto')],
                         outputs=[ASR_TYPE.Output('asr')])

    @classmethod
    def execute(cls, model_name, device):
        if model_name not in folder_paths.get_filename_list(ASR_FOLDER) or Path(model_name).suffix != '.safetensors':
            raise ValueError('Select an ASR safetensors file from models/audio8_asr.')
        return io.NodeOutput(load_asr(folder_paths.get_full_path_or_raise(ASR_FOLDER, model_name), device))


class Audio8Transcribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Audio8Transcribe', display_name='Audio8 Transcribe Reference', category='audio/Audio8',
                         description='Transcribe the reference in its original language, then offload ASR before TTS. Review the transcript for recognition errors.',
                         inputs=[ASR_TYPE.Input('asr'), io.Audio.Input('audio'),
                                 io.String.Input('language', default='auto',
                                                 tooltip='Auto detects language. An ISO code such as en, fr or ar can improve short-clip recognition.')],
                         outputs=[io.String.Output('transcript'), io.String.Output('metrics')])

    @classmethod
    def execute(cls, asr, audio, language):
        text, metrics = transcribe(asr, audio, language)
        return io.NodeOutput(text, json.dumps(metrics, indent=2))


class Audio8Quantize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='Audio8Quantize', display_name='Export Audio8 INT8', category='audio/Audio8',
                         description='Explicitly quantize a floating-point bundle. Creates a new file; never replaces the source.',
                         is_output_node=True,
                         inputs=[io.Combo.Input('source_model', options=[n for n in folder_paths.get_filename_list(FOLDER) if n.endswith('.safetensors')]),
                                 io.String.Input('output_name', default='audio8-custom-int8.safetensors'),
                                 io.Combo.Input('profile', options=['int8', 'int8-all'], default='int8')],
                         outputs=[io.String.Output('report')])

    @classmethod
    def execute(cls, source_model, output_name, profile):
        if (Path(output_name).name != output_name or '/' in output_name or '\\' in output_name
                or not output_name.endswith('.safetensors')):
            raise ValueError('Output must be a safetensors filename without directories.')
        root = Path(folder_paths.get_folder_paths(FOLDER)[0]).resolve()
        output = root / output_name
        progress = comfy.utils.ProgressBar(1)
        result = artifact.export(resolve_model(source_model), output, profile=profile,
                                  progress=lambda value, total: progress.update_absolute(value, total))
        return io.NodeOutput(json.dumps(result, indent=2))
