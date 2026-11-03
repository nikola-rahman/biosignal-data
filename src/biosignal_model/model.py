"""The network: a transformer encoder over the tokens of packed examples.

Token embedding: a linear map of the token's samples and their observed flags, plus a
linear map of (log length in s, channel scale), plus the embedding of its channel slot.
A hidden token gets a learned vector in place of its samples (its time, length, scale and
slot stay: the model knows where and in which channel to predict).

Position: rotary embedding on continuous time in seconds, not on token index, so tokens
of channels at different rates line up by time. The wavelengths are spread on a log scale
from `min_wavelength_s` to `max_wavelength_s`. The angles (time x frequency) are computed in
float64 and only cos and sin are cast to the model's dtype: in float32 the angle at 1800 s and
a 10 ms wavelength (1.1e6 rad) is off by up to 0.06 rad, and the time itself by 6e-5 s.

Attention: each token attends to the tokens of its own example only (`document`).
Blocks: pre-norm (RMSNorm), multi-head attention, SwiGLU feed-forward.
Output: the samples of each token, in its channel's scale (the target of masked-token
pretraining).

Tokenizers (`ModelConfig.tokenizer`, docs/v3-tokenizer.md): `samples` is the above;
`duration` (D1) embeds frames of a fixed duration with a continuous-time stem; `hierarchical`
(D2) runs the stem on short frames, bidirectional S4D over each channel's frames, attention
pooling into tokens, and adds the local features back to the output.

Benchmark interface: a task reads `features()` and trains with `layer_groups()` (the weights in
order from the input to the output, for layer-wise learning rates); nothing else of the
model is named outside it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from torch import nn

from biosignal_model.batch import (
    DurationTokenizer,
    HierarchicalTokenizer,
    Inputs,
    SamplesTokenizer,
    frame_shape,
)

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


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
    tokenizer: SamplesTokenizer | DurationTokenizer | HierarchicalTokenizer
    # Regularization of the residual branches (global blocks and D2's local blocks): dropout
    # on each branch's output, and stochastic depth (drop path: a whole branch dropped for a
    # row of the batch, or a channel's sequence in the local stage, with this probability).
    # 0 and 0: none (v0-v2).
    dropout: float
    drop_path: float


@dataclass(frozen=True)
class LayerGroups:
    """The weights a task uses, in order from the input to the output, for layer-wise learning
    rates (benchmarks); each weight is in one group. The pretraining heads are not in it."""

    input: list[nn.Parameter]  # samples -> token vectors: tokenizer, context, hidden vectors
    channel: list[nn.Parameter]  # the channel identity (slot embedding)
    layers: list[list[nn.Parameter]]  # each global layer, the lowest first
    output: list[nn.Parameter]  # after the last layer: the final norm (and D2's residual)


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.width % config.heads or (config.width // config.heads) % 2:
            raise ValueError("width / heads must be a whole, even number")
        self.config = config
        p, d = config.samples_per_token, config.width
        t = config.tokenizer
        if isinstance(t, SamplesTokenizer):
            self.samples = nn.Linear(2 * p, d)
        elif isinstance(t, DurationTokenizer):
            self.stem = Stem(t, d)
        else:
            self.stem = Stem(t, t.local_width)
            self.hidden_local = nn.Parameter(torch.zeros(t.local_width))
            self.local = nn.ModuleList(
                S4DBlock(t.local_width, t.local_states, config.dropout, config.drop_path)
                for _ in range(t.local_layers)
            )
            self.pool = AttentionPool(t.local_width, t.pool_heads, d)
            if t.residual_pool == "attention":
                self.residual_pool = AttentionPool(t.local_width, t.pool_heads, t.local_width)
            self.residual = nn.Linear(t.local_width, d)
        self.context = nn.Linear(3, d)
        self.slot = nn.Embedding(config.max_channels, d)
        self.hidden = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(
            Block(config.width, config.heads, config.feed_forward, config.dropout,
                  config.drop_path)
            for _ in range(config.layers)
        )  # fmt: skip
        self.norm = nn.RMSNorm(d)
        if isinstance(t, SamplesTokenizer):
            self.head = nn.Linear(d, p)
        elif isinstance(t, DurationTokenizer):
            self.head_kernel = nn.Linear(self.stem.basis_size, d, bias=False)

    def layer_groups(self) -> LayerGroups:
        """The weights a task uses, from the input to the output (see LayerGroups)."""
        t = self.config.tokenizer
        parts: list[nn.Module]
        if isinstance(t, SamplesTokenizer):
            parts = [self.samples]
        elif isinstance(t, DurationTokenizer):
            parts = [self.stem]
        else:
            parts = [self.stem, self.local, self.pool]
            if t.residual_pool == "attention":
                parts.append(self.residual_pool)
        parts.append(self.context)
        hidden = [self.hidden]  # the learned vectors of hidden tokens / frames
        out: list[nn.Module] = [self.norm]
        if isinstance(t, HierarchicalTokenizer):
            hidden.append(self.hidden_local)
            out.append(self.residual)
        return LayerGroups(
            input=[p for m in parts for p in m.parameters()] + hidden,
            channel=list(self.slot.parameters()),
            layers=[list(b.parameters()) for b in self.blocks],
            output=[p for m in out for p in m.parameters()],
        )

    def rotary(self, time_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) of the rotary angles of tokens at `time_s` [R, L], [R, 1, L, half], in
        the dtype of the weights; the angles in float64 (see the module docstring)."""
        c = self.config
        half = c.width // c.heads // 2
        wavelengths = torch.logspace(
            math.log10(c.min_wavelength_s), math.log10(c.max_wavelength_s), half,
            dtype=torch.float64, device=time_s.device,
        )  # fmt: skip
        angles = time_s.to(torch.float64)[..., None] * (2 * math.pi / wavelengths)
        dtype = self.hidden.dtype
        return angles.cos().to(dtype)[:, None], angles.sin().to(dtype)[:, None]

    def forward(self, x: Inputs) -> torch.Tensor:
        """The predicted samples of every token, [R, L, P] ([R, L, 1, P] for `duration`), in
        the channel's scale."""
        t = self.config.tokenizer
        if isinstance(t, SamplesTokenizer):
            out: torch.Tensor = self.head(self.features(x))
            return out
        if isinstance(t, HierarchicalTokenizer):
            raise NotImplementedError("pretraining with the hierarchical tokenizer: not built yet")
        h = self.features(x)
        assert x.rate_hz is not None  # noqa: S101 - the packer sets it for this tokenizer
        predicted = torch.zeros(x.values.shape, device=h.device, dtype=h.dtype)
        for rate in torch.unique(x.rate_hz[x.document >= 0]).tolist():
            n = frame_shape(t, rate)[0]
            at = x.rate_hz == rate
            basis = self.stem.basis(_offsets(n, rate, h.device), rate)  # [n, B]
            kernel = self.head_kernel(basis.to(h.dtype))  # [n, d]
            predicted[at, 0, :n] = h[at] @ kernel.T
        return predicted

    def features(self, x: Inputs) -> torch.Tensor:
        """The representation of every token, [R, L, width], after the last block's norm:
        what a probe or a task head reads."""
        t = self.config.tokenizer
        local: torch.Tensor | None = None
        if isinstance(t, SamplesTokenizer):
            observed = x.observed.to(x.values.dtype)
            h = self.samples(torch.cat([x.values * observed, observed], dim=-1))
        elif isinstance(t, DurationTokenizer):
            h = self.stem(x)[:, :, 0]
        else:
            h, local = self._hierarchical(x, t)
        h = torch.where(x.hidden[..., None], self.hidden.to(h.dtype), h)
        h = h + self.context(torch.cat([x.log_length_s[..., None], x.channel_scale], dim=-1))
        h = h + self.slot(x.slot)
        rotary = self.rotary(x.time_s)  # [R, 1, L, half]
        attend = _attention(x.document, self.config.attention)
        for block in self.blocks:
            h = block(h, rotary, attend)
        if local is not None:
            h = h + self.residual(local)
        features: torch.Tensor = self.norm(h)
        return features

    def _hierarchical(
        self, x: Inputs, t: HierarchicalTokenizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """D2's local stage: (pooled token vectors [R, L, width], the token's local frame
        features pooled for the output residual [R, L, local_width]: their mean, or attention
        pooling, `t.residual_pool`). Each channel's frames in time order go through the S4D
        blocks; frames that are not observed, padding and other channels never enter a
        sequence."""
        assert x.channel_index is not None and x.position is not None  # noqa: S101
        frames = self.stem(x)  # [R, L, F, C]
        r, length, f, c = frames.shape
        real = x.document >= 0
        ok = x.observed.any(-1) & real[..., None]  # [R, L, F]
        hide = x.hidden[..., None] & real[..., None]
        frames = torch.where(hide[..., None], self.hidden_local.to(frames.dtype), frames)
        ok = ok | hide
        # one sequence per channel of the batch: index (channel, position) of every real token
        per_row = torch.where(real, x.channel_index, -1).amax(dim=1) + 1  # channels per row
        base = torch.cumsum(per_row, 0) - per_row
        rows = torch.arange(r, device=frames.device)[:, None].expand(r, length)
        g = (base[rows] + x.channel_index)[real]
        pos = x.position[real]
        channels, longest = int(per_row.sum()), int(pos.max()) + 1 if pos.numel() else 1
        seq = frames.new_zeros(channels, longest, f, c)
        seq[g, pos] = frames[real]
        mask = ok.new_zeros(channels, longest, f)
        mask[g, pos] = ok[real]
        seq, mask = seq.view(channels, longest * f, c), mask.view(channels, longest * f)
        seq = seq * mask[..., None]
        for block in self.local:
            seq = block(seq, mask)
        out = frames.new_zeros(r, length, f, c)
        out[real] = seq.view(channels, longest, f, c)[g, pos].to(out.dtype)
        if t.residual_pool == "attention":
            return self.pool(out, ok), self.residual_pool(out, ok)
        weight = ok.to(out.dtype)
        mean = (out * weight[..., None]).sum(2) / weight.sum(2, keepdim=True).clamp(min=1)
        return self.pool(out, ok), mean


def _offsets(n: int, rate: float, device: torch.device) -> torch.Tensor:
    """The time of each sample of a frame from its centre, in seconds."""
    return (torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2) / rate


class Stem(nn.Module):
    """A strided convolution with kernels defined in seconds (docs/v3-tokenizer.md): for each
    rate in the batch, the kernels are evaluated at that rate's sample offsets, without the
    sinusoids at or above its Nyquist frequency, so one weight set reads every rate."""

    frequencies: torch.Tensor

    def __init__(self, t: DurationTokenizer | HierarchicalTokenizer, width: int) -> None:
        super().__init__()
        self.tokenizer = t
        frequencies = torch.logspace(
            math.log10(t.kernel_min_hz), math.log10(t.kernel_max_hz), t.kernel_frequencies
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.basis_size = 1 + 2 * t.kernel_frequencies
        self.values = nn.Linear(self.basis_size, t.stem_channels, bias=False)
        self.observed = nn.Linear(self.basis_size, t.stem_channels)
        self.out = nn.Linear(t.stem_windows * t.stem_channels, width)

    def basis(self, tau: torch.Tensor, rate: float) -> torch.Tensor:
        """[*, 1 + 2 frequencies]: 1, cos 2πfτ, sin 2πfτ; zero where f >= rate / 2."""
        keep = (self.frequencies < rate / 2).to(tau.dtype)
        angle = 2 * math.pi * tau[..., None] * self.frequencies
        return torch.cat(
            [torch.ones_like(tau)[..., None], angle.cos() * keep, angle.sin() * keep], dim=-1
        )

    def forward(self, x: Inputs) -> torch.Tensor:
        """[R, L, F, width]: each frame's embedding; zero for padding tokens."""
        assert x.rate_hz is not None  # noqa: S101 - the packer sets it for this tokenizer
        r, length, f, _ = x.values.shape
        k = self.tokenizer.stem_windows
        z = x.values.new_zeros(r, length, f, k, self.values.out_features)
        for rate in torch.unique(x.rate_hz[x.document >= 0]).tolist():
            n = frame_shape(self.tokenizer, rate)[0]
            at = x.rate_hz == rate
            span = n / rate
            half = span / k  # each window's half-width
            centres = -span / 2 + (torch.arange(k, device=x.values.device) + 0.5) * span / k
            tau = _offsets(n, rate, x.values.device)[None, :] - centres[:, None]  # [K, n]
            window = torch.where(tau.abs() < half, 0.5 + 0.5 * torch.cos(math.pi * tau / half), 0.0)
            weight = self.basis(tau, rate) * (window / (rate * half))[..., None]  # [K, n, B]
            u = x.values[at][..., :n] * x.observed[at][..., :n]  # [N, F, n]
            m = x.observed[at][..., :n].to(u.dtype)
            z[at] = torch.einsum("tfn,knb->tfkb", u, weight) @ self.values.weight.T + (
                torch.einsum("tfn,knb->tfkb", m, weight) @ self.observed.weight.T
                + self.observed.bias
            )
        h: torch.Tensor = self.out(F.gelu(z).flatten(-2))
        return h * (x.document >= 0)[..., None, None].to(h.dtype)


class S4DBlock(nn.Module):
    """RMSNorm, bidirectional S4D (arXiv 2206.11893: diagonal state matrix, S4D-Lin
    initialization A_n = -1/2 + iπn, ZOH, the kernel applied as an FFT convolution; one kernel
    per direction), GELU, a linear layer with GLU, and the residual. `mask`: [B, T] frames that
    exist; the others are zero before and after."""

    def __init__(self, width: int, states: int, dropout: float, drop_path: float) -> None:
        super().__init__()
        half = states // 2
        self.dropout, self.drop_path = dropout, drop_path
        self.norm = nn.RMSNorm(width)
        dt_min, dt_max = 0.001, 0.1  # the S4D defaults
        log_dt = torch.rand(2, width) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.c = nn.Parameter(torch.view_as_real(torch.randn(2, width, half, dtype=torch.cfloat)))
        self.log_a_real = nn.Parameter(torch.log(0.5 * torch.ones(2, width, half)))
        self.a_imag = nn.Parameter(
            math.pi * torch.arange(half).float().expand(2, width, half).clone()
        )
        self.d = nn.Parameter(torch.randn(width))
        self.out = nn.Linear(width, 2 * width)

    def kernel(self, length: int) -> torch.Tensor:
        """[2, width, length]: the convolution kernel of each direction."""
        dt = self.log_dt.exp()[..., None]  # [2, H, 1]
        a = -self.log_a_real.exp() + 1j * self.a_imag
        c = torch.view_as_complex(self.c) * (torch.exp(dt * a) - 1.0) / a
        steps = torch.arange(length, device=a.device)
        vandermonde = torch.exp((dt * a)[..., None] * steps)  # [2, H, N, L]
        return 2 * torch.einsum("dhn,dhnl->dhl", c, vandermonde).real

    def forward(self, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        length = u.shape[1]
        keep = mask[..., None].to(u.dtype)
        z = (self.norm(u) * keep).float().transpose(1, 2)  # [B, H, T]
        k = self.kernel(length)
        # Both directions as one kernel over the lags -(T-1)..T-1, one FFT pair: forward
        # y[t] += k0[m] z[t-m], backward y[t] += k1[m] z[t+m] (lag 0 gets both); with circular
        # length 2T, lag -m sits at 2T-m and nothing wraps onto a used output.
        both = F.pad(k[0], (0, length)) + F.pad(k[1, :, 1:].flip(-1), (length + 1, 0))
        both[:, 0] += k[1, :, 0]
        n = 2 * length
        y = torch.fft.irfft(torch.fft.rfft(z, n) * torch.fft.rfft(both, n), n)[..., :length]
        y = (y + z * self.d[:, None]).transpose(1, 2).to(u.dtype)
        y = F.glu(self.out(F.gelu(y)), dim=-1)
        y = _drop_path(F.dropout(y, self.dropout, self.training), self.drop_path, self.training)
        return (u + y) * keep


class AttentionPool(nn.Module):
    """`heads` learned queries over the frames of each token ([R, L, F, C], `ok` [R, L, F]);
    a token with no frame gives 0. Out: [R, L, width]."""

    def __init__(self, local: int, heads: int, width: int) -> None:
        super().__init__()
        if local % heads:
            raise ValueError("local_width / pool_heads must be whole")
        self.heads = heads
        self.query = nn.Parameter(torch.randn(heads, local // heads) / math.sqrt(local // heads))
        self.key = nn.Linear(local, local, bias=False)
        self.value = nn.Linear(local, local, bias=False)
        self.out = nn.Linear(local, width)

    def forward(self, frames: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        r, length, f, c = frames.shape
        key = self.key(frames).view(r, length, f, self.heads, c // self.heads)
        value = self.value(frames).view(r, length, f, self.heads, c // self.heads)
        score = (key * self.query.to(key.dtype)).sum(-1) / math.sqrt(c // self.heads)
        score = score.float().masked_fill(~ok[..., None], -torch.inf)
        weight = torch.nan_to_num(score.softmax(dim=2), nan=0.0).to(value.dtype)  # [R, L, F, H]
        pooled = (weight[..., None] * value).sum(2).reshape(r, length, c)
        out: torch.Tensor = self.out(pooled)
        return out * ok.any(-1)[..., None].to(out.dtype)


class Block(nn.Module):
    """A pre-norm transformer block: attention with rotary angles, a SwiGLU feed-forward."""

    def __init__(
        self, width: int, heads: int, feed_forward: int, dropout: float, drop_path: float
    ) -> None:
        super().__init__()
        d = width
        self.heads = heads
        self.dropout, self.drop_path = dropout, drop_path
        self.norm1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.up = nn.Linear(d, 2 * feed_forward, bias=False)
        self.down = nn.Linear(feed_forward, d, bias=False)

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
        h = h + self._branch(self.out(a))
        gate, value = self.up(self.norm2(h)).chunk(2, dim=-1)
        result: torch.Tensor = h + self._branch(self.down(F.silu(gate) * value))
        return result

    def _branch(self, y: torch.Tensor) -> torch.Tensor:
        return _drop_path(F.dropout(y, self.dropout, self.training), self.drop_path, self.training)


def _drop_path(y: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth: the branch `y` [B, ...] set to 0 for each first-axis item with
    probability `p` (in training), the kept ones scaled by 1 / (1 - p)."""
    if not training or p == 0.0:
        return y
    keep = torch.rand(y.shape[0], *([1] * (y.dim() - 1)), device=y.device) >= p
    return y * keep.to(y.dtype) / (1.0 - p)


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
    from torch.nn.attention.flex_attention import flex_attention  # noqa: PLC0415 - GPU only

    block = _block_mask(document)
    kernel = _compiled_flex(flex_attention)

    def flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = kernel(q, k, v, block_mask=block)
        return out

    return flex


def _block_mask(document: torch.Tensor) -> BlockMask:
    """The flex block mask of "same document" for `document` [R, L], made one row at a time
    (create_block_mask of all rows at once holds a dense [R, L, L] mask as int64: gigabytes at
    L = 8192) and joined: the same block mask."""
    from torch.nn.attention.flex_attention import (  # noqa: PLC0415 - GPU only
        BlockMask,
        create_block_mask,
    )

    def same(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return document[b, q] == document[b, kv]

    length = document.shape[1]
    parts = []
    for row in document:

        def same_in_row(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor,
            row: torch.Tensor = row,
        ) -> torch.Tensor:  # fmt: skip
            return row[q] == row[kv]

        parts.append(create_block_mask(same_in_row, 1, None, length, length, row.device))
    full = [(p.full_kv_num_blocks, p.full_kv_indices) for p in parts]  # separate full blocks
    return BlockMask.from_kv_blocks(
        torch.cat([p.kv_num_blocks for p in parts]),
        torch.cat([p.kv_indices for p in parts]),
        torch.cat([n for n, _ in full if n is not None]),
        torch.cat([i for _, i in full if i is not None]),
        BLOCK_SIZE=parts[0].BLOCK_SIZE,
        mask_mod=same,
        seq_lengths=(length, length),
    )


_FLEX: dict[str, Callable[..., torch.Tensor]] = {}


# Compiled flex kernels kept: one per (rows, length, heads, head width). The v4 model has two
# head shapes (encoder, decoder) and power-of-two row lengths (biosignal_model.v4.layout): with
# torch's default of 8, the ninth shape fell back to the eager kernel.
FLEX_SHAPES = 64


def _compiled_flex(flex: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    if "flex" not in _FLEX:
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, FLEX_SHAPES
        )
        _FLEX["flex"] = torch.compile(flex, dynamic=False)
    return _FLEX["flex"]


def masked_loss(predicted: torch.Tensor, x: Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean squared error over the observed samples of hidden tokens, and the per-token
    sum and count of squared errors, [R, L, 2], for reports."""
    hidden = x.hidden.view(*x.hidden.shape, *([1] * (x.observed.dim() - 2)))
    weight = (x.observed & hidden).to(predicted.dtype)
    squared = (predicted.float() - x.values) ** 2 * weight
    inner = tuple(range(2, squared.dim()))
    per_token = torch.stack([squared.sum(inner), weight.sum(inner)], dim=-1)
    loss = squared.sum() / weight.sum().clamp(min=1)
    return loss, per_token.detach()
