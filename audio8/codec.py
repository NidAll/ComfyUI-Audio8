"""Offline codec adapted from ``Edge0-AI/Audio8_TTS``'s codec module.

The upstream implementation is Apache-2.0 licensed; see its repository LICENSE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import comfy.ops
import comfy_kitchen
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.modules.attention import (
    AttentionTensorContainer,
    ComfyAttention,
    optimized_attention_for_device,
)
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.utils import remove_weight_norm as remove_legacy_weight_norm
from torch.nn.utils import weight_norm as legacy_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from .model import rope_table


class _EmptyLinear(nn.Linear):
    def reset_parameters(self):
        pass


class _EmptyConv1d(nn.Conv1d):
    def reset_parameters(self):
        pass


class _EmptyConvTranspose1d(nn.ConvTranspose1d):
    def reset_parameters(self):
        pass


class _EmptyEmbedding(nn.Embedding):
    def reset_parameters(self):
        pass


class _EmptyLayerNorm(nn.LayerNorm):
    def reset_parameters(self):
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty_like(self.weight))
            if self.bias is not None:
                self.bias = nn.Parameter(torch.empty_like(self.bias))


class ArkttsCodecRMSNorm(comfy.ops.manual_cast.RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__(dim, eps=eps, dtype=torch.float32)


class ArkttsCodecLayerScale(nn.Module):
    def __init__(self, dim: int, inplace: bool = False):
        super().__init__()
        self.inplace = bool(inplace)
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


@dataclass
class ArkttsCodecTransformerConfig:
    n_layer: int
    n_head: int
    dim: int
    intermediate_size: int
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    channels_first: bool = True

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head


class ArkttsCodecAttention(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        total = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = _EmptyLinear(config.dim, total, bias=False)
        self.wo = _EmptyLinear(config.head_dim * config.n_head, config.dim, bias=False)
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.comfy_attention = ComfyAttention()

    def forward(self, x, rope_values, mask):
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        query, key, value = self.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
        query = query.view(batch, length, self.n_head, self.head_dim)
        key = key.view(batch, length, self.n_local_heads, self.head_dim)
        value = value.view(batch, length, self.n_local_heads, self.head_dim)
        frequencies = rope_values[None, :, None]
        query, key = comfy_kitchen.apply_rope(query, key, frequencies)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key, value = comfy.ops.repeat_kv_for_gqa(key, value, self.n_head, 1)
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros(mask.shape, device=query.device, dtype=query.dtype).masked_fill(
                ~mask, torch.finfo(query.dtype).min
            )
        attention = optimized_attention_for_device(query.device, mask=mask is not None)
        output = attention(
            AttentionTensorContainer(query),
            AttentionTensorContainer(key),
            AttentionTensorContainer(value),
            self.n_head,
            mask=mask,
            skip_reshape=True,
            preferred_attention=self.comfy_attention,
        )
        return self.wo(output)


class ArkttsCodecFeedForward(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        self.w1 = _EmptyLinear(config.dim, config.intermediate_size, bias=False)
        self.w3 = _EmptyLinear(config.dim, config.intermediate_size, bias=False)
        self.w2 = _EmptyLinear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Identity()

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class ArkttsCodecTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ArkttsCodecAttention(config)
        self.feed_forward = ArkttsCodecFeedForward(config)
        self.ffn_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_layer_scale = ArkttsCodecLayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = ArkttsCodecLayerScale(config.dim, inplace=True)

    def forward(self, x, rope_values, mask):
        hidden = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), rope_values, mask)
        )
        return hidden + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(hidden)))


class ArkttsCodecWindowTransformer(nn.Module):
    def __init__(self, config, input_dim: int, window_size: int | None, causal: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(
            [ArkttsCodecTransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.input_proj = _EmptyLinear(input_dim, config.dim) if input_dim != config.dim else nn.Identity()
        self.output_proj = _EmptyLinear(config.dim, input_dim) if input_dim != config.dim else nn.Identity()
        self.look_ahead_conv = nn.Identity()
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base

    def forward(self, x, x_lens=None):
        del x_lens
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.look_ahead_conv(self.input_proj(x))
        length = x.shape[1]
        row = torch.arange(length, device=x.device)[:, None]
        column = torch.arange(length, device=x.device)[None, :]
        mask = column <= row
        if self.window_size is not None:
            mask &= column >= (row - self.window_size + 1).clamp_min(0)
        mask = mask[None, None]
        rope_values = rope_table(length, self.head_dim, self.rope_base, x.device)
        for layer in self.layers:
            x = layer(x, rope_values, mask)
        x = self.output_proj(self.norm(x))
        return x.transpose(1, 2) if self.channels_first else x


def _extra_padding(x, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return ideal - length


class ArkttsCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = _EmptyConv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        right = _extra_padding(x, self.kernel_size, self.stride, self.padding)
        return self.conv(F.pad(x, (self.padding, right))).contiguous()

    def apply_weight_norm(self):
        self.conv = weight_norm(self.conv)
        return self


class ArkttsCausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = _EmptyConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        crop = self.kernel_size - self.stride
        return x[..., : x.shape[-1] - crop].contiguous() if crop else x.contiguous()

    def apply_weight_norm(self):
        self.conv = weight_norm(self.conv)
        return self


def _causal_wn_conv(*args, **kwargs):
    return ArkttsCausalConv1d(*args, **kwargs).apply_weight_norm()


def _causal_wn_transpose(*args, **kwargs):
    return ArkttsCausalConvTranspose1d(*args, **kwargs).apply_weight_norm()


def _arktts_snake(x: Tensor, alpha: Tensor) -> Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class ArkttsSnake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x):
        return _arktts_snake(x, self.alpha)


class ArkttsResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            ArkttsSnake1d(dim),
            _causal_wn_conv(dim, dim, kernel_size=7, dilation=dilation),
            ArkttsSnake1d(dim),
            _causal_wn_conv(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        output = self.block(x)
        difference = x.shape[-1] - output.shape[-1]
        if difference > 0:
            x = x[..., :-difference]
        return x + output


class ArkttsEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, transformer_layers: int):
        super().__init__()
        modules = [
            ArkttsResidualUnit(dim // 2, 1),
            ArkttsResidualUnit(dim // 2, 3),
            ArkttsResidualUnit(dim // 2, 9),
            ArkttsSnake1d(dim // 2),
            _causal_wn_conv(dim // 2, dim, kernel_size=2 * stride, stride=stride),
        ]
        if transformer_layers:
            config = ArkttsCodecTransformerConfig(
                n_layer=transformer_layers,
                n_head=dim // 64,
                dim=dim,
                intermediate_size=dim * 3,
            )
            modules.append(ArkttsCodecWindowTransformer(config, dim, window_size=512))
        else:
            modules.append(nn.Identity())
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class ArkttsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 64
        modules = [_causal_wn_conv(1, dim, kernel_size=7)]
        for stride, transformer_layers in zip((2, 4, 8, 8), (0, 0, 0, 4)):
            dim *= 2
            modules.append(ArkttsEncoderBlock(dim, stride, transformer_layers))
        modules.extend((ArkttsSnake1d(dim), _causal_wn_conv(dim, 1024, kernel_size=3)))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class ArkttsDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ArkttsSnake1d(input_dim),
            _causal_wn_transpose(input_dim, output_dim, kernel_size=2 * stride, stride=stride),
            ArkttsResidualUnit(output_dim, 1),
            ArkttsResidualUnit(output_dim, 3),
            ArkttsResidualUnit(output_dim, 9),
        )

    def forward(self, x):
        return self.block(x)


class ArkttsDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 1536
        modules = [_causal_wn_conv(1024, channels, kernel_size=7)]
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            modules.append(ArkttsDecoderBlock(input_dim, output_dim, stride))
        modules.extend((ArkttsSnake1d(output_dim), _causal_wn_conv(output_dim, 1, kernel_size=7), nn.Tanh()))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ArkttsVectorQuantizer(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.in_proj = legacy_weight_norm(_EmptyConv1d(input_dim, codebook_dim, kernel_size=1))
        self.out_proj = legacy_weight_norm(_EmptyConv1d(codebook_dim, input_dim, kernel_size=1))
        self.codebook = _EmptyEmbedding(codebook_size, codebook_dim)

    def decode_code(self, indices):
        return F.embedding(indices, self.codebook.weight).transpose(1, 2)

    def decode_latents(self, latents):
        batch, _, length = latents.shape
        flattened = latents.transpose(1, 2).reshape(batch * length, -1)
        flattened = F.normalize(flattened)
        codebook = F.normalize(self.codebook.weight)
        distances = (
            flattened.pow(2).sum(1, keepdim=True)
            - 2 * flattened @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distances).argmax(1).view(batch, length)
        return self.decode_code(indices), indices

    def forward(self, z):
        projected = self.in_proj(z)
        quantized, indices = self.decode_latents(projected)
        quantized_st = projected + (quantized - projected).detach()
        return self.out_proj(quantized_st), indices, projected


class ArkttsResidualQuantizer(nn.Module):
    def __init__(self, input_dim: int, n_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.quantizers = nn.ModuleList(
            [ArkttsVectorQuantizer(input_dim, codebook_size, codebook_dim) for _ in range(n_codebooks)]
        )

    def forward(self, z):
        quantized_sum = 0.0
        residual = z
        codes = []
        for quantizer in self.quantizers:
            quantized, indices, _ = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            codes.append(indices)
        return quantized_sum, torch.stack(codes, dim=1)

    def from_codes(self, codes):
        output = 0.0
        for index in range(codes.shape[1]):
            projected = self.quantizers[index].decode_code(codes[:, index])
            output = output + self.quantizers[index].out_proj(projected)
        return output


class ArkttsConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = ArkttsCausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = _EmptyLayerNorm(dim, eps=1e-6)
        self.pwconv1 = _EmptyLinear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = _EmptyLinear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 2, 1)
        return residual + x


class ArkttsDownsampleQuantizer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.semantic_quantizer = ArkttsResidualQuantizer(1024, 1, 4096, 8)
        self.quantizer = ArkttsResidualQuantizer(1024, 9, 1024, 8)
        self.downsample = nn.Sequential(
            nn.Sequential(
                ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ),
            nn.Sequential(
                ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ),
        )
        self.upsample = nn.Sequential(
            nn.Sequential(
                ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ),
            nn.Sequential(
                ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(1024),
            ),
        )
        pre_transformer_config = ArkttsCodecTransformerConfig(
            n_layer=8,
            n_head=16,
            dim=1024,
            intermediate_size=3072,
        )
        post_transformer_config = ArkttsCodecTransformerConfig(
            n_layer=int(config.get("codec_post_n_layer", 8)),
            n_head=int(config.get("codec_post_n_head", 16)),
            n_local_heads=int(config.get("codec_post_n_local_heads", 8)),
            dim=1024,
            intermediate_size=int(config.get("codec_post_intermediate_size", 1216)),
        )
        self.pre_module = ArkttsCodecWindowTransformer(
            pre_transformer_config, 1024, window_size=128
        )
        self.post_module = ArkttsCodecWindowTransformer(
            post_transformer_config, 1024, window_size=128
        )
        self.semantic_predictor_module = nn.Identity()

    def forward(self, z):
        original_length = z.shape[-1]
        z = self.pre_module(self.downsample(z))
        semantic, semantic_codes = self.semantic_quantizer(z)
        residual, residual_codes = self.quantizer(z - semantic)
        z = self.upsample(self.post_module(semantic + residual))
        difference = original_length - z.shape[-1]
        if difference > 0:
            z = F.pad(z, (difference, 0))
        elif difference < 0:
            z = z[..., -difference:]
        return z, torch.cat((semantic_codes, residual_codes), dim=1)

    def decode(self, indices):
        indices = indices.clone()
        indices[:, 0].clamp_(0, self.semantic_quantizer.codebook_size - 1)
        indices[:, 1:].clamp_(0, self.quantizer.codebook_size - 1)
        semantic = self.semantic_quantizer.from_codes(indices[:, :1])
        residual = self.quantizer.from_codes(indices[:, 1:])
        return self.upsample(self.post_module(semantic + residual))


class ArkttsCodec(nn.Module):
    sample_rate = 44100
    hop_length = 512
    frame_length = 2048

    def __init__(self, config: dict):
        super().__init__()
        self.sample_rate = int(config.get("codec_sample_rate", self.sample_rate))
        self.frame_length = int(config.get("codec_frame_size", self.frame_length))
        self.encoder = ArkttsEncoder()
        self.quantizer = ArkttsDownsampleQuantizer(config)
        self.decoder = ArkttsDecoder()

    def encode(self, audio, audio_lengths=None):
        if audio.ndim == 2:
            audio = audio[:, None]
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [B, 1, samples]")
        original_length = audio.shape[-1]
        right = math.ceil(original_length / self.frame_length) * self.frame_length - original_length
        if right:
            audio = F.pad(audio, (0, right))
        if audio_lengths is None:
            audio_lengths = torch.full(
                (audio.shape[0],), original_length, device=audio.device, dtype=torch.long
            )
        encoded = self.encoder(audio)
        _, codes = self.quantizer(encoded)
        code_lengths = torch.ceil(audio_lengths.float() / self.frame_length).long()
        max_codes = codes.shape[-1]
        padded = torch.full_like(codes, -1)
        for index, length in enumerate(code_lengths.tolist()):
            padded[index, :, : min(length, max_codes)] = codes[index, :, : min(length, max_codes)]
        return padded, code_lengths.clamp_max(max_codes)

    def decode(self, codes):
        return self.decoder(self.quantizer.decode(codes.long()))


def _normalize_codec_state_dict(state_dict: dict) -> dict:
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator." in key for key in state_dict):
        state_dict = {
            key.replace("generator.", ""): value
            for key, value in state_dict.items()
            if "generator." in key
        }
    return {
        key: value
        for key, value in state_dict.items()
        if not key.endswith(("freqs_cis", "causal_mask"))
    }


def _remove_weight_norm(module: nn.Module) -> None:
    for child in tuple(module.modules()):
        if parametrize.is_parametrized(child, "weight"):
            parametrize.remove_parametrizations(child, "weight", leave_parametrized=True)
        elif hasattr(child, "weight_g") and hasattr(child, "weight_v"):
            remove_legacy_weight_norm(child)


def _load_codec_state_dict(codec: nn.Module, state_dict: dict, dtype=torch.float32) -> None:
    state = _normalize_codec_state_dict(state_dict)
    state = {
        key: value.to(dtype=dtype) if value.is_floating_point() and value.dtype != dtype else value
        for key, value in state.items()
    }
    codec.load_state_dict(state, strict=True, assign=True)
    codec.to(dtype=dtype)
    _remove_weight_norm(codec)


def build_codec(config: dict, state_dict: dict, dtype=torch.float32) -> ArkttsCodec:
    """Strictly load on CPU and fold weight norm in the execution dtype."""
    with torch.device("meta"):
        codec = ArkttsCodec(config)
    _load_codec_state_dict(codec, state_dict, dtype)
    codec.eval()
    return codec


__all__ = ["ArkttsCodec", "build_codec"]
