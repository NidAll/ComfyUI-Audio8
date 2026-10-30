"""Audio8 DualAR inference, adapted from Edge0's Apache-2.0 implementation.

Weights retain upstream names. Generation projects only permitted semantic
tokens, and uses execution-local KV caches instead of model-owned buffers.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
import comfy_kitchen as kitchen
from comfy.ldm.modules.attention import (
    AttentionTensorContainer, ComfyAttention, optimized_attention_for_device,
)
from .graphs import FastDecoderGraph, SlowDecoderGraph


def rope_table(length, width, base, device):
    frequency = 1.0 / (base ** (torch.arange(0, width, 2, device=device).float() / width))
    phase = torch.outer(torch.arange(length, device=device), frequency)
    # Upstream stores BF16 sin/cos but applies rotations in FP32. Kitchen
    # consumes the equivalent 2x2 matrix, including on its eager backend.
    cosine, sine = phase.cos().to(torch.bfloat16).float(), phase.sin().to(torch.bfloat16).float()
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).unflatten(-1, (2, 2))


@dataclass
class KVCache:
    keys: torch.Tensor
    values: torch.Tensor

    @classmethod
    def create(cls, batch, heads, length, width, device, dtype):
        shape = (batch, heads, length, width)
        return cls(torch.zeros(shape, device=device, dtype=dtype),
                   torch.zeros(shape, device=device, dtype=dtype))


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_heads, width, qkv_bias, out_bias, qk_norm, eps, operations):
        super().__init__()
        self.n_head, self.n_local_heads, self.head_dim = heads, kv_heads, width
        self.wqkv = operations.Linear(dim, (heads + 2 * kv_heads) * width, bias=qkv_bias)
        self.wo = operations.Linear(heads * width, dim, bias=out_bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = operations.RMSNorm(width, eps=eps)
            self.k_norm = operations.RMSNorm(width, eps=eps)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, rope, mask, cache=None, start=0, end=None):
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=x.dtype).masked_fill_(~mask, -float('inf'))
        batch, length, _ = x.shape
        qsize, kvsize = self.n_head * self.head_dim, self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split((qsize, kvsize, kvsize), dim=-1)
        q = q.reshape(batch, length, self.n_head, self.head_dim)
        k = k.reshape(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        frequencies = rope[None, :, None] if rope.ndim == 4 else rope[:, :, None]
        q, k = kitchen.apply_rope(q, k, frequencies)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.reshape(batch, length, self.n_local_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            if isinstance(start, int):
                cache.keys[:, :, start:start + length] = k
                cache.values[:, :, start:start + length] = v
            else:
                cache.keys.index_copy_(2, start, k)
                cache.values.index_copy_(2, start, v)
            k, v = cache.keys[:, :, :end], cache.values[:, :, :end]
        # Comfy's attention dispatcher owns backend selection. Explicit head
        # expansion also supports backends that do not implement GQA.
        repeats = self.n_head // self.n_local_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        attention = optimized_attention_for_device(q.device, mask=mask is not None, small_input=length == 1)
        out = attention(
            AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
            self.n_head, mask=mask, skip_reshape=True,
            preferred_attention=self.comfy_attention,
        )
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim, intermediate, operations):
        super().__init__()
        self.w1 = operations.Linear(dim, intermediate, bias=False)
        self.w2 = operations.Linear(intermediate, dim, bias=False)
        self.w3 = operations.Linear(dim, intermediate, bias=False)
        self.gate_up = None

    def forward(self, x):
        if self.gate_up is not None:
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            return self.w2(F.silu(gate) * up)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def fuse(self):
        first, second = self.w1.state_dict(), self.w3.state_dict()
        if set(first) != set(second):
            raise ValueError('Gate and up projections must have matching quantization.')
        fused = {}
        for key in first:
            if key in ('weight', 'weight_scale'):
                fused[key] = torch.cat((first[key], second[key]), dim=0)
            else:
                if not torch.equal(first[key], second[key]):
                    raise ValueError('Gate and up projection metadata must match.')
                fused[key] = first[key]
        self.gate_up = type(self.w1)(self.w1.in_features, self.w1.out_features * 2, bias=False)
        self.gate_up.load_state_dict(fused, strict=True, assign=True)
        self.w1, self.w3 = None, None


class Block(nn.Module):
    def __init__(self, c, fast, operations):
        super().__init__()
        prefix = 'fast_' if fast else ''
        dim = getattr(c, prefix + 'dim')
        self.attention = Attention(
            dim, getattr(c, prefix + 'n_head'), getattr(c, prefix + 'n_local_heads'),
            getattr(c, prefix + 'head_dim'), getattr(c, prefix + 'attention_qkv_bias'),
            getattr(c, prefix + 'attention_o_bias'), getattr(c, prefix + 'attention_qk_norm'),
            c.norm_eps, operations,
        )
        self.feed_forward = FeedForward(dim, getattr(c, prefix + 'intermediate_size'), operations)
        self.ffn_norm = operations.RMSNorm(dim, eps=c.norm_eps)
        self.attention_norm = operations.RMSNorm(dim, eps=c.norm_eps)

    def forward(self, x, rope, mask, cache=None, start=0, end=None):
        x = x + self.attention(self.attention_norm(x), rope, mask, cache, start, end)
        return x + self.feed_forward(self.ffn_norm(x))


def filtered_scores(logits, top_k, top_p, temperature):
    # Upstream filters nucleus candidates BEFORE applying temperature, and
    # excludes the token that crosses p. Keep that sampling contract.
    values, indices = logits.sort(descending=True, dim=-1)
    cumulative = values.softmax(-1).cumsum(-1)
    positions = torch.arange(values.shape[-1], device=values.device)
    remove = (cumulative > torch.full((), top_p, dtype=cumulative.dtype, device=values.device)) | (positions >= top_k)
    remove[..., 0] = False
    filtered = torch.full_like(logits, -float('inf')).scatter(-1, indices, values.masked_fill(remove, -float('inf')))
    return filtered / torch.full((), temperature, dtype=logits.dtype, device=logits.device).clamp_min(1e-5)


def sample(scores, generator):
    probabilities = scores.softmax(-1)
    noise = -torch.rand(probabilities.shape, dtype=probabilities.dtype,
                        device=probabilities.device, generator=generator).log()
    return (probabilities / noise).argmax(-1)


class Audio8Model(nn.Module):
    def __init__(self, config, operations):
        super().__init__()
        c = self.config = SimpleNamespace(**config)
        self.embeddings = operations.Embedding(c.vocab_size, c.dim)
        self.codebook_embeddings = operations.Embedding(c.codebook_size * c.num_codebooks, c.dim)
        self.layers = nn.ModuleList([Block(c, False, operations) for _ in range(c.n_layer)])
        self.norm = operations.RMSNorm(c.dim, eps=c.norm_eps)
        self.fast_project_in = operations.Linear(c.dim, c.fast_dim) if c.dim != c.fast_dim else nn.Identity()
        self.fast_embeddings = operations.Embedding(c.codebook_size, c.fast_dim)
        self.fast_layers = nn.ModuleList([Block(c, True, operations) for _ in range(c.n_fast_layer)])
        self.fast_norm = operations.RMSNorm(c.fast_dim, eps=c.norm_eps)
        self.fast_output = operations.Linear(c.fast_dim, c.codebook_size, bias=False)

    def fuse_projections(self):
        for layer in [*self.layers, *self.fast_layers]:
            layer.feed_forward.fuse()

    def embed(self, ids):
        c = self.config
        offsets = torch.arange(c.num_codebooks, device=ids.device)[None, :, None] * c.codebook_size
        codebooks = self.codebook_embeddings(ids[:, 1:] + offsets).sum(1)
        semantic = (ids[:, 0] >= c.semantic_begin_id) & (ids[:, 0] <= c.semantic_end_id)
        return self.embeddings(ids[:, 0]) + torch.where(semantic[..., None], codebooks, 0.0)

    def slow_step(self, ids, rope, mask, caches, start, end, output_weight):
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=output_weight.dtype).masked_fill_(~mask, -float('inf'))
        hidden = self.embed(ids)
        for layer, cache in zip(self.layers, caches):
            hidden = layer(hidden, rope, mask, cache, start, end)
        hidden = hidden[:, -1:]
        normalized = self.norm(hidden)
        logits = F.linear(normalized, output_weight)[:, -1]
        return logits, normalized if self.config.norm_fastlayer_input else hidden

    def codebooks(self, slow_hidden, semantic, rope, caches, top_k, top_p, temperature, do_sample, generator):
        c = self.config
        hidden = self.fast_project_in(slow_hidden)
        current = torch.where(semantic < c.codebook_size, semantic, 0)
        codes = [current]
        for position in range(c.num_codebooks):
            mask = None  # Cache slices contain only preceding and current positions.
            for layer, cache in zip(self.fast_layers, caches):
                hidden = layer(hidden, rope[position:position + 1], mask, cache, position, position + 1)
            if position:
                logits = self.fast_output(self.fast_norm(hidden))[:, -1]
                scores = filtered_scores(logits, top_k, top_p, temperature)
                current = sample(scores, generator) if do_sample else scores.argmax(-1)
                codes.append(current)
            hidden = self.fast_embeddings(current)[:, None]
        return torch.stack(codes, dim=1)

    def generate(self, prompt, prompt_mask, *, max_new_tokens=512, temperature=0.8,
                 top_p=0.95, top_k=50, do_sample=True, seed=42, callback=None, acceleration='auto'):
        c = self.config
        batch, _, width = prompt.shape
        if width >= c.max_seq_len:
            raise ValueError(f'Text and reference use {width} positions; the model context is {c.max_seq_len}.')
        if max_new_tokens < 1 or temperature <= 0 or top_k < 1 or not 0 < top_p <= 1:
            raise ValueError('Require max_new_tokens/top_k >= 1, temperature > 0 and 0 < top_p <= 1.')
        limit = min(max_new_tokens, c.max_seq_len - width)
        device, dtype = prompt.device, self.embeddings.weight.dtype
        generator = torch.Generator(device=device).manual_seed(seed)
        slow_cache = [KVCache.create(batch, c.n_local_heads, width + limit, c.head_dim, device, dtype) for _ in self.layers]
        fast_cache = [KVCache.create(batch, c.fast_n_local_heads, c.num_codebooks, c.fast_head_dim, device, dtype) for _ in self.fast_layers]
        rope = rope_table(c.max_seq_len, c.head_dim, c.rope_base, device)
        fast_rope = rope_table(c.num_codebooks, c.fast_head_dim, c.rope_base, device)
        # This is the tied output projection restricted to the model's exact
        # semantic allowlist. Copying avoids retaining an oversized backing view.
        output_ids = torch.cat((torch.arange(c.semantic_begin_id, c.semantic_end_id + 1, device=device),
                                torch.full((1,), c.eos_token_id, device=device, dtype=torch.long)))
        output_weight = self.embeddings(output_ids).contiguous()
        positions = prompt_mask.cumsum(-1).sub(1).clamp_min(0)
        causal = torch.arange(width, device=device)[None, :] <= torch.arange(width, device=device)[:, None]
        mask = causal[None, None] & prompt_mask[:, None, None].bool()
        logits, hidden = self.slow_step(prompt, rope[positions], mask, slow_cache, 0, width, output_weight)
        lengths = prompt_mask.sum(-1)
        valid = torch.zeros(batch, width + limit, device=device, dtype=torch.bool)
        valid[:, :width] = prompt_mask.bool()
        finished = torch.zeros(batch, device=device, dtype=torch.bool)
        code_lengths = torch.zeros(batch, device=device, dtype=torch.long)
        frames = torch.full((batch, c.num_codebooks, limit), -1, device=device, dtype=torch.long)
        previous = None
        fast_graph = None
        slow_graph = None
        use_graph = device.type == 'cuda' and acceleration != 'eager' and limit >= 16
        try:
            for step in range(limit):
                if callback is not None:
                    callback(step, limit)
                scores = filtered_scores(logits, top_k, top_p, temperature)
                semantic = sample(scores, generator) if do_sample else scores.argmax(-1)
                if do_sample:
                    high = sample(filtered_scores(logits, top_k, c.ras_top_p, c.ras_temperature), generator)
                    if previous is not None:
                        repeated = (previous == (semantic + c.semantic_begin_id)[:, None]).any(1)
                        semantic = torch.where(repeated & (semantic < c.codebook_size), high, semantic)
                active = ~finished
                if use_graph and fast_graph is None:
                    fast_graph = FastDecoderGraph(
                        lambda h, s: self.codebooks(h, s, fast_rope, fast_cache, top_k, top_p,
                                                   temperature, do_sample, generator),
                        hidden, semantic, generator,
                    )
                codes = (fast_graph(hidden, semantic) if fast_graph is not None else
                         self.codebooks(hidden, semantic, fast_rope, fast_cache, top_k, top_p,
                                        temperature, do_sample, generator))
                emitted = active & (semantic != c.codebook_size)
                frames[:, :, step] = torch.where(emitted[:, None], codes, -1)
                code_lengths += emitted.long()
                token = torch.where(semantic == c.codebook_size, c.eos_token_id, semantic + c.semantic_begin_id)
                if previous is None:
                    previous = torch.zeros(batch, c.ras_window_size, device=device, dtype=torch.long)
                else:
                    previous = previous.roll(-1, dims=1)
                    previous[:, -1] = token
                finished |= semantic == c.codebook_size
                if finished.all() or step == limit - 1:
                    break
                column = torch.cat((token[:, None], codes), dim=1).unsqueeze(-1)
                valid[:, width + step] = active
                end = width + step + 1
                current_rope = rope[(lengths + step)[:, None]]
                if use_graph:
                    inputs = (column, current_rope, valid[:, None, None],
                              torch.full((1,), width + step, device=device, dtype=torch.long))
                    if slow_graph is None:
                        slow_graph = SlowDecoderGraph(
                            lambda ids, r, m, p: self.slow_step(ids, r, m, slow_cache, p, None, output_weight), inputs,
                        )
                    logits, hidden = slow_graph(*inputs)
                else:
                    mask = valid[:, None, None, :end]
                    logits, hidden = self.slow_step(column, current_rope, mask,
                                                    slow_cache, width + step, end, output_weight)
        finally:
            logits, hidden, codes = None, None, None
            if fast_graph is not None:
                fast_graph.close()
            if slow_graph is not None:
                slow_graph.close()
        return frames[:, :, :int(code_lengths.max())].clone(), code_lengths, finished
