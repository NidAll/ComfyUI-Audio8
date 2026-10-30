"""Execution-local CUDA replay for the ten fast-decoder steps per frame."""

import torch


class FastDecoderGraph:
    def __init__(self, function, hidden, semantic, generator):
        self.hidden = torch.empty_like(hidden)
        self.semantic = torch.empty_like(semantic)
        self.hidden.copy_(hidden)
        self.semantic.copy_(semantic)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(generator)
        state = generator.get_state()
        capture = torch.cuda.graph(self.graph)
        # PyTorch reuses this capture stream. Creating a new stream on each
        # call retains a separate cuBLAS workspace for every stream.
        stream = capture.capture_stream
        stream.wait_stream(torch.cuda.current_stream(hidden.device))
        with torch.cuda.stream(stream):
            warmup = function(self.hidden, self.semantic)
            function(self.hidden, self.semantic)
            self.output = torch.empty_like(warmup)
            del warmup
        torch.cuda.current_stream(hidden.device).wait_stream(stream)
        with capture:
            result = function(self.hidden, self.semantic)
            self.output.copy_(result)
            del result
        torch.cuda.current_stream(hidden.device).wait_stream(stream)
        generator.set_state(state)

    def __call__(self, hidden, semantic):
        self.hidden.copy_(hidden)
        self.semantic.copy_(semantic)
        self.graph.replay()
        return self.output

    def close(self):
        self.graph.reset()
        self.graph = None
        self.output = None
        self.hidden = None
        self.semantic = None


class SlowDecoderGraph:
    def __init__(self, function, inputs):
        self.inputs = [value.clone() for value in inputs]
        self.graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.graph(self.graph)
        stream = capture.capture_stream
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream):
            warmup = function(*self.inputs)
            function(*self.inputs)
            self.output = tuple(torch.empty_like(value) for value in warmup)
            del warmup
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        with capture:
            result = function(*self.inputs)
            for destination, source in zip(self.output, result):
                destination.copy_(source)
            del source, result
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)

    def __call__(self, *inputs):
        for destination, source in zip(self.inputs, inputs):
            destination.copy_(source)
        self.graph.replay()
        return self.output

    def close(self):
        self.graph.reset()
        self.graph = None
        self.output = None
        self.inputs.clear()
