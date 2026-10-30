import json
import torch
import torch.nn as nn
from comfy.cli_args import args
args.cpu = True
import comfy.ops
from comfy_kitchen.tensor import TensorWiseINT8Layout
from audio8.model import FeedForward


def test_float_gate_up_fusion_preserves_output():
    layer = FeedForward(64, 128, nn)
    x = torch.randn(2, 3, 64)
    expected = layer(x)
    layer.fuse()
    torch.testing.assert_close(layer(x), expected, atol=2e-6, rtol=2e-6)
    assert layer.w1 is None and layer.w3 is None


def test_quantized_gate_up_fusion_keeps_bytes_scales_and_compute():
    operations = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32, disabled=['int8_tensorwise'])
    layer = FeedForward(64, 128, operations)
    state = {}
    for name, shape in [('w1', (128, 64)), ('w2', (64, 128)), ('w3', (128, 64))]:
        q, params = TensorWiseINT8Layout.quantize(torch.randn(shape) * 0.02, per_channel=True,
                                                convrot=True, convrot_groupsize=64)
        state[name + '.weight'] = q
        state[name + '.weight_scale'] = params.scale
        config = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 64}
        state[name + '.comfy_quant'] = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
    layer.load_state_dict(state, strict=True, assign=True)
    x = torch.randn(2, 3, 64)
    expected = layer(x)
    layer.fuse()
    torch.testing.assert_close(layer(x), expected, atol=2e-6, rtol=2e-6)
    fused = layer.gate_up.state_dict()
    assert torch.equal(fused['weight'], torch.cat((state['w1.weight'], state['w3.weight'])))
    assert torch.equal(fused['weight_scale'], torch.cat((state['w1.weight_scale'], state['w3.weight_scale'])))
