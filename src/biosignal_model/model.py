"""The network: a transformer encoder over the tokens of packed examples.

Token embedding: a linear map of the token's samples and their observed flags, plus a
linear map of (log length in s, channel scale), plus the embedding of its channel slot.
A hidden token gets a learned vector in place of its samples (its time, length, scale and
slot stay: the model knows where and in which channel to predict).

Position: rotary embedding on continuous time in seconds, not on token index, so tokens
of channels at different rates line up by time. The wavelengths are spread on a log scale
from `min_wavelength_s` to `max_wavelength_s`.

Attention: each token attends to the tokens of its own example only (`document`).
Blocks: pre-norm (RMSNorm), multi-head attention, SwiGLU feed-forward.
Output: the samples of each token, in its channel's scale (the target of masked-token
pretraining).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from torch import nn

from biosignal_model.batch import Inputs


@dataclass(frozen=True)
class ModelConfig:
    samples_per_token: int
    max_channels: int
    width: int
    layers: int
    heads: int
    feed_forward: int
    min_wavelength_s: float
    max_wavelength_s: float
    attention: Literal["flex", "dense"]  # flex: block-sparse kernel (GPU); dense: a full mask


class Encoder(nn.Module):
    frequencies: torch.Tensor  # radians per second, one per pair of head dimensions

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.width % config.heads or (config.width // config.heads) % 2:
            raise ValueError("width / heads must be a whole, even number")
        self.config = config
        p, d = config.samples_per_token, config.width
        self.samples = nn.Linear(2 * p, d)
        self.context = nn.Linear(3, d)
        self.slot = nn.Embedding(config.max_channels, d)
        self.hidden = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, p)
        half = d // config.heads // 2
        wavelengths = torch.logspace(
            math.log10(config.min_wavelength_s), math.log10(config.max_wavelength_s), half
        )
        self.register_buffer("frequencies", 2 * math.pi / wavelengths, persistent=False)

    def forward(self, x: Inputs) -> torch.Tensor:
        """The predicted samples of every token, [R, L, P], in the channel's scale."""
        out: torch.Tensor = self.head(self.features(x))
        return out

    def features(self, x: Inputs) -> torch.Tensor:
        """The representation of every token, [R, L, width], after the last block's norm:
        what a probe or a task head reads."""
        observed = x.observed.to(x.values.dtype)
        h = self.samples(torch.cat([x.values * observed, observed], dim=-1))
        h = torch.where(x.hidden[..., None], self.hidden.to(h.dtype), h)
        h = h + self.context(torch.cat([x.log_length_s[..., None], x.channel_scale], dim=-1))
        h = h + self.slot(x.slot)
        angles = x.time_s[..., None] * self.frequencies  # [R, L, half]
        rotary = (angles.cos()[:, None], angles.sin()[:, None])  # [R, 1, L, half]
        attend = _attention(x.document, self.config.attention)
        for block in self.blocks:
            h = block(h, rotary, attend)
        features: torch.Tensor = self.norm(h)
        return features


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.width
        self.heads = config.heads
        self.norm1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.up = nn.Linear(d, 2 * config.feed_forward, bias=False)
        self.down = nn.Linear(config.feed_forward, d, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        attend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        r, length, d = h.shape
        q, k, v = (
            self.qkv(self.norm1(h)).view(r, length, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )  # fmt: skip
        q, k = _rotate(q, rotary), _rotate(k, rotary)
        a = attend(q, k, v).transpose(1, 2).reshape(r, length, d)
        h = h + self.out(a)
        gate, value = self.up(self.norm2(h)).chunk(2, dim=-1)
        result: torch.Tensor = h + self.down(F.silu(gate) * value)
        return result


def _rotate(x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = (t.to(x.dtype) for t in rotary)
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


def _attention(
    document: torch.Tensor, kind: Literal["flex", "dense"]
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Attention within one example. Padding (document -1) attends to padding only."""
    if kind == "dense":
        mask = (document[:, :, None] == document[:, None, :])[:, None]

        def dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return dense
    from torch.nn.attention.flex_attention import (  # noqa: PLC0415 - GPU only
        create_block_mask,
        flex_attention,
    )

    def same(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return document[b, q] == document[b, kv]

    r, length = document.shape
    block = create_block_mask(same, r, None, length, length, device=document.device)
    kernel = _compiled_flex(flex_attention)

    def flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = kernel(q, k, v, block_mask=block)
        return out

    return flex


_FLEX: dict[str, Callable[..., torch.Tensor]] = {}


def _compiled_flex(flex: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    if "flex" not in _FLEX:
        _FLEX["flex"] = torch.compile(flex, dynamic=False)
    return _FLEX["flex"]


def masked_loss(predicted: torch.Tensor, x: Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean squared error over the observed samples of hidden tokens, and the per-token
    sum and count of squared errors, [R, L, 2], for reports."""
    weight = (x.observed & x.hidden[..., None]).to(predicted.dtype)
    squared = (predicted.float() - x.values) ** 2 * weight
    per_token = torch.stack([squared.sum(-1), weight.sum(-1)], dim=-1)
    loss = squared.sum() / weight.sum().clamp(min=1)
    return loss, per_token.detach()
