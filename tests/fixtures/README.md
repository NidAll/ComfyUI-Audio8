# Tiny Audio8 golden fixture

`audio8_tiny_golden.safetensors` contains synthetic float32 model weights, small test inputs, and CPU reference outputs. It contains no pretrained weights or private data. Default tests load it directly and do not import Transformers or access an upstream checkout.

The tensors were generated from the official `Edge0/Audio8-TTS-Realistic-0.6b` source at revision `1e857568bfd4f02931aca3988ee773a44d2d26d3`, using `modeling_arktts.py` and `configuration_arktts.py` on CPU. The deterministic model initialization seed is 2026. `audio8_tiny_golden.json` records the tiny model configuration and fixture provenance.
