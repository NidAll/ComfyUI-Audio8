# Audio8 owner

Keep this pack inside its own directory. No core or sibling changes. Runtime is
local-only: load a single safetensors containing model/codec weights, tokenizer
and config; never execute embedded code or download from a node. Explicit
`tools/download.py` and `tools/prepare_asr.py --download` are the Hub download
entry points. ASR has its own single-file bundle in models/audio8_asr and uses
optional Transformers with Comfy-managed full loading and prompt offloading.

Use V3 nodes, standard AUDIO, Comfy model management and Kitchen/native ops.
Keep KV caches and CUDA graphs local to a synthesis call. Preserve upstream
prompt formatting, sampling order (top-p before temperature), RAS and codec math.
Quantization is opt-in offline or through the explicit export node. Protect
embeddings, norms, fast output and codec by default. Validate actual kernel
execution and generated audio; reconstruction error is not speech quality.
Long output splits target text, reuses the original reference for every section,
and joins only EOS-complete sections. Retry exhausted sections at smaller text
sizes; never silently drop the tail. Preserve the original short/batch node API.

Run CPU tests with the ComfyUI venv and `PYTHONPATH` pointing to ComfyUI. Real GPU
checks need driver access outside the sandbox. Put machine-specific benchmark
results, original weights and generated media under workspace `reports/`, not
in this public-source package. Do not claim production acceptance until long,
short, reference and multilingual real inference checks pass.
