# Audio8 for ComfyUI

Offline text-to-speech and zero-shot voice cloning for
[Edge0/Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b).
One `.safetensors` contains the DualAR model, neural codec, tokenizer and
configuration. The node pack supplies the inference code. Loading and synthesis
make no network requests and execute no downloaded or embedded Python.

## Install

Place this directory in `ComfyUI/custom_nodes/ComfyUI-Audio8`. Use the same Python
environment as ComfyUI. Current ComfyUI with its V3 API, PyTorch/torchaudio 2.5+
and Comfy Kitchen 0.2.40+ is required. TTS does not import Transformers. Optional
reference transcription uses Transformers 4.57–5.x; no backend or CUDA library
replacement is required when it is already installed.

```sh
python -m pip install -r requirements.txt
```

Prepare the model explicitly, then restart ComfyUI normally. Nodes never download
models. Run these commands from this pack's directory:

```sh
# Optional download; needs huggingface-hub. Downloads only the pinned Audio8 assets.
python tools/download.py /path/to/audio8-source

# Inspect the exact layer plan without writing a model.
python tools/quantize.py /path/to/audio8-source /path/to/ComfyUI/models/audio8/audio8-realistic-int8-all.safetensors --profile int8-all --dry-run

# Convert entirely on CPU; preserve the source and refuse existing output paths.
python tools/quantize.py /path/to/audio8-source /path/to/ComfyUI/models/audio8/audio8-realistic-int8-all.safetensors --profile int8-all
```

The published single-file bundles are available from
[`NidAll/audio8-tts-realistic-0.6b-int8-convrot`](https://huggingface.co/NidAll/audio8-tts-realistic-0.6b-int8-convrot).
Download them explicitly into the ComfyUI model folders:

```sh
hf download NidAll/audio8-tts-realistic-0.6b-int8-convrot audio8-realistic-int8-all.safetensors --local-dir /path/to/ComfyUI/models/audio8
hf download NidAll/audio8-tts-realistic-0.6b-int8-convrot whisper-base.safetensors --local-dir /path/to/ComfyUI/models/audio8_asr
```

The source directory needs `model.safetensors`, `codec.pth`, `config.json`, and
`tokenizer.json`. Conversion reads the official codec through PyTorch's restricted
`weights_only=True` loader. Runtime accepts only the new single-file format.
Tokenizer integrity and component presence are checked on load. Outputs commit
atomically without replacing an existing model.

## Use

Open a visual workflow from `example_workflows/`. Files ending in `_api.json` are
plain API prompts. The examples use native Load Audio, Save Audio and Preview as
Text nodes alongside these Audio8 nodes:

| Node | Purpose |
| --- | --- |
| Load Audio8 | Select a local bundle, compute precision, device and codec placement. |
| Audio8 Reference Voice | Encode a reference recording and its exact transcript. Reuse its output. |
| Audio8 Speech | Generate speech, optionally using the encoded voice; returns standard AUDIO and metrics. |
| Audio8 Long Speech | Generate long text in complete sections using one shared voice, then join the sections. |
| Load Audio8 ASR | Load a local multilingual Whisper Base model. |
| Audio8 Transcribe Reference | Produce the reference transcript in its original language, then offload ASR. |
| Export Audio8 INT8 | Explicitly export a new quantized artifact from a floating-point bundle. |

Without a voice input, speech uses the model's default voice distribution. For
cloning, load a clean recording with native Load Audio, enter the words it speaks
exactly, and connect Reference Voice to Speech. Stereo references are mixed to
mono and resampled to 44.1 kHz. Reference encoding is cached by the Comfy graph;
changing only target text does not re-encode an unchanged reference node.

For automatic transcription and long **output**, open
`04_long_voice_clone_auto.json`. Load a short, clean recording of the speaker,
choose its language or leave `auto`, and enter the full target text in **Audio8
Long Speech**. The reference goes to both Transcribe Reference and Reference
Voice; the generated transcript connects to Reference Voice. Preview the
transcript and correct recognition errors by disconnecting that link and entering
the exact words manually. The workflow produces one continuous AUDIO output.

Prepare the small ASR bundle once, explicitly:

```sh
# Only needed if Transformers is absent; use the same environment as ComfyUI.
python -m pip install '.[asr]'
python tools/prepare_asr.py /path/to/whisper-source /path/to/ComfyUI/models/audio8_asr/whisper-base.safetensors --download
```

Without `--download`, the export is entirely offline. The approximately 148 MB
FP16 ASR file contains weights, tokenizer, feature-extractor and generation config.
Whisper Base is multilingual; `language=auto` detects the reference language.
Short recordings may benefit from an explicit ISO code such as `en`, `fr`, or
`ar`. ASR transcribes rather than translates. Recognition quality varies, so the
transcript remains visible for review. GPU uses FP16 and PyTorch SDPA; CPU uses
FP32. Only the ASR model is offloaded after transcription, leaving GPU memory for
Audio8. TTS and ASR are separate portable files.

Language follows the input text. There is no artificial language selector or
translation stage. Supported upstream languages are Arabic, Cantonese, Chinese,
Dutch, English, Finnish, French, German, Hindi, Indonesian, Italian, Japanese,
Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai,
Turkish and Vietnamese. Cross-language cloning uses a reference transcript in
its original language and target text in the desired language. Language quality
varies. Native speaker tags in text pass through the upstream prompt format;
this release does not add a separate multi-speaker conversation engine.

Enable **batch_lines** to synthesize each nonempty line as an independent batch
item, optionally sharing one reference voice. AUDIO uses `[batch, 1, samples]`;
short items receive trailing zero padding. Metrics include each original sample
count and EOS status. With batch_lines off, multiline text forms one utterance.

The context holds 2,048 packed text and reference-audio positions. Each generated
frame represents 2,048 samples. `max_new_tokens=512` permits about 23.8 seconds,
subject to remaining context. Text/reference inputs are never silently trimmed.
Generation stops on EOS or the remaining frame budget. `finished=false` means the
budget was reached; increase it or split the text for a complete utterance.
The Comfy node produces complete audio, not incremental streaming packets.

**Audio8 Long Speech** works beyond that per-section limit by splitting target
text at sentence or word boundaries (default `chunk_chars=240`). Each section
reuses the original encoded voice. If a section reaches its frame/context limit,
its incomplete result is discarded and smaller sections are generated instead.
Successful sections are joined on CPU with `pause_ms` of silence between them.
`max_new_tokens` is the limit **per section**, not for the entire output. Metrics
record each section's text, seed, sample range, EOS status and retry count, plus
the final duration. A section that still cannot finish after splitting produces
an actionable error instead of returning cut-off speech. Chunking can change
prosody between sections. Long Speech defaults to greedy decoding (`do_sample=false`)
to reduce the risk of early stopping seen in sampled generation. EOS alone does
not prove that every word was pronounced; review important recordings. It does not extend the model's attention context or
provide streaming. Reference recordings should still be short and clean.

## Precision and acceleration

| Export profile | INT8 ConvRot layers | Preserved BF16 components |
| --- | --- | --- |
| `bf16` | None | Entire model and codec; floating reference bundle. |
| `int8` | 120 slow-transformer linears | Fast transformer, embeddings, norms, output head, codec. |
| `int8-all` | 140 slow/fast-transformer linears | Embeddings, norms, output head, codec. |

The codec is never integer-quantized. `--codec-dtype fp32` preserves its original
FP32 source tensors instead of BF16. Quantized linears use Kitchen's native
INT8 tensorwise format, FP32 row scales and group-64 ConvRot. The exporter checks
every converted matrix against its source. This is weight reconstruction
validation, not a speech-quality guarantee. Export reports include actual layer
coverage and maximum relative L2 reconstruction error.

`acceleration=auto` uses execution-local CUDA graphs for both decoders when at
least 16 frames are permitted and the transformer fits in VRAM. Capture and
warmup costs are included in synthesis timing. Graphs and KV caches are released
after each call; PyTorch's capture stream is reused to avoid accumulating cuBLAS
workspaces. `eager` disables graph capture for backend troubleshooting and very
short requests. CPU execution uses native floating-point fallback and no graphs.

The tied semantic head projects only 4,096 audio tokens plus EOS. Fast decoding
skips the unused output projection on its initial cache step. RoPE, RMSNorm,
attention, quantization and matrix multiplication use shared Kitchen, ComfyUI
and PyTorch operations. No custom CUDA kernels, compilation, external servers or
runtime autotuning are required. Feed-forward gate/up projections share one
native linear operation after loading; quantized bytes and row scales are
concatenated without requantizing. Comfy's selected attention backend and its
small-input dispatch policy are respected.

INT8 is not always faster. Compare `int8-all`, `int8` and `bf16` on the actual
workload and hardware. Graph replay can change their ordering by reducing Python
and kernel-launch overhead. CUDA and CPU paths are covered; other accelerator
backends have not been accepted.

GPU codec placement gives lower latency. The codec and transformer have separate
Comfy model patchers so the manager can evict one between stages. CPU codec
placement uses FP32 and leaves more VRAM for synthesis, at the cost of slower
encoding/decoding. Partially offloaded transformers use eager execution. Batch
size, text length, reference duration and attention backend affect peak memory.

## Reproduce measurements

Run each configuration separately with an idle device. Preserve identical text,
reference, sampling controls and frame budget. Measure cold load, the first call,
and repeated calls; compare complete EOS-terminated speech as well as fixed-frame
throughput. `--profile` writes a CUDA trace and operation table. Results and audio
are written only to the requested output directory.

```sh
python tools/benchmark.py /path/to/ComfyUI/models/audio8/audio8-realistic-int8-all.safetensors --output /path/to/results/int8 --frames 64 --repeats 4
python tools/benchmark.py /path/to/ComfyUI/models/audio8/audio8-realistic-bf16.safetensors --output /path/to/results/bf16 --frames 64 --repeats 4
```

For the original implementation, `--original` requires a local reviewed official
snapshot also containing `configuration_arktts.py`, `modeling_arktts.py` and
`modeling_arktts_codec.py`. This optional benchmark loads that local source;
production nodes do not. Upstream recommends Transformers 4.57.x; use a separate
environment if the official baseline is incompatible with the host installation.

```sh
python tools/benchmark.py /path/to/reviewed-official-snapshot --original --output /path/to/results/original --frames 64 --repeats 4
```

The default benchmark is greedy. Add `--sample` for the node's stochastic mode.
Top-k/top-p filtering precedes temperature, matching upstream. Reduced-vocabulary
sampling consumes fewer random numbers than the original, so equal seeds do not
produce identical sampled audio across implementations. Floating-point kernels
and autoregressive feedback can also change later greedy tokens.

Run the offline tests without model downloads:

```sh
python -m pytest -q
```

Tests include synthetic golden outputs generated with the pinned upstream code,
codec math, native quantization, artifact integrity/atomic export, prompts,
filesystem boundaries and V3 node contracts. Real-model checks and kernel traces
are separate from these synthetic checks. Automated waveform checks do not prove
intelligibility, naturalness or speaker similarity; those need listening and
matched ASR/speaker evaluations. No universal quality-preservation claim is made.

## Provenance

Pinned model/source revision: `1e857568bfd4f02931aca3988ee773a44d2d26d3`.
Architecture and codec adapted from the Apache-2.0
[official Audio8 implementation](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b/tree/1e857568bfd4f02931aca3988ee773a44d2d26d3).
Optional ASR uses [OpenAI Whisper Base](https://huggingface.co/openai/whisper-base)
at revision `e37978b90ca9030d5170a5c07aadb050351a65bb` (Apache-2.0).
See `LICENSE`, `NOTICE` and the fixture provenance. The single-file format is
specific to this pack; it is not a generic core ComfyUI checkpoint.
