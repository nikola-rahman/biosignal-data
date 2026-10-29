"""The local stage (docs/architecture.md section 3): every sample of every channel, at its own
rate, before the global tokens.

`LocalStage` maps a ChannelInputs to one feature vector per sample, [S, width], in the order of
the flat sample buffer. Implementations:

- `S4DLocalStage`: an input map of each sample (its value, observed and hidden flags), then
  `layers` residual blocks, each: RMSNorm, a bidirectional diagonal complex SSM (S4D) along
  each channel, GELU, a linear map with GLU; and in the layers named by `mixing_layers`, after
  it, attention across the co-located channels at each sample time; a final RMSNorm.
- `IdentityLocalStage`: the input map and the norm only (no context): the control, and the
  place a continuous-kernel convolution can take.

The SSM is defined in continuous time, in seconds, so one set of weights reads every rate:

- per direction (forward, backward), feature h and mode n: lambda = a / Delta_h with a complex
  and Delta_h a time scale in seconds; at init a_n = -1/2 + i pi n (S4D-Lin) and Delta_h
  log-uniform in [min_time_s, max_time_s] (arXiv 2206.11893 with its dt read as a time in
  seconds); B = 1 / Delta_h (so a mode's gain is about the same at every time scale);
- discretized with dt = 1/fs of the channel (`discretization`): `zoh` (zero-order hold:
  A = exp(lambda dt), B = (A - 1) / lambda; the poles exact at every rate, the kernel is the
  integral of the continuous one over [l dt, (l + 1) dt]), `zoh_centred` (the same poles; the
  integral over [(l - 1/2) dt, (l + 1/2) dt], lag 0 over its half: no half-sample delay) or
  `bilinear` (the S4 transform; warps the frequency axis, for the comparison);
- a mode with |lambda| dt > pi is masked at that rate (above Nyquist, or decaying within one
  sample), the same rule at any rate, also one never seen in training; `masked_modes`: `drop`
  (the masked mode adds nothing) or `static_gain` (it adds its static gain -C / a at lag 0: its
  response to in-band input, and the ZOH limit of a mode that decays within one sample; so the
  DC gain of every mode is the same at every rate). Measured 2026-10-29
  (scripts/v4_local_checks.py): with time scales from 1 ms, `drop` changes the features by
  ~40 % between 100 and 500 Hz on a signal below 10 Hz, because a fast mode's static gain does
  not depend on its time scale;
- one kernel per distinct rate in the batch, of at most `kernel_time_constants` times the
  longest decay time 1 / |Re lambda| of the unmasked modes (and at most the longest channel),
  applied as an FFT convolution in float32 at a fast size (2, 3, 5, 7-smooth) of the length
  plus the kernel;
- channels of one rate and length are put together once a forward (length buckets: the stage
  works in its own sample order, given back in the buffer's order at the end), so every layer
  works on views, [channels, width, length] for the FFT; the FFT's zero padding is the only
  padding, so a channel's output does not depend on the others; the kernel lengths of all
  layers and rates come with one host sync a forward.

Channel mixing: channels of one example with the same rate and first-sample time share a time
base (ECG leads, EEG electrodes, the EMG grid); at each sample time, multi-head attention
across them (pre-norm, residual), with no channel identity: the stage is equivariant to the
order of the channels. The channels of such a group are adjacent in the stage's order, so the
attention reads a view [groups, channels, length, width]; only a group whose channels differ
in length (at a recording's end) is gathered, with a mask. Channels on another time base are mixed later, by the global model.
For grids, a neighbour convolution needs the grid positions (descriptors): a later step.

Activation checkpointing per layer (`checkpoint`), so the memory is one layer's activations
plus one [S, width] tensor a layer.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.checkpoint import checkpoint

from biosignal_model.channel_batch import ChannelInputs


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class S4DLocalSpec(_Spec):
    kind: Literal["s4d"]
    width: int
    layers: int
    states: int  # complex modes per feature and direction
    min_time_s: float  # the mode time scales at init: log-uniform in [min, max] (s)
    max_time_s: float
    kernel_time_constants: float  # the kernel cap, in decay times of the slowest mode
    discretization: Literal["zoh", "zoh_centred", "bilinear"]
    masked_modes: Literal["drop", "static_gain"]  # what a mode above |lambda| dt > pi adds
    mixing_layers: list[int]  # the layers (from 0) followed by channel mixing
    mixing_heads: int
    dropout: float  # on the residual branches
    checkpoint: bool  # activation checkpointing per layer


class IdentityLocalSpec(_Spec):
    kind: Literal["identity"]
    width: int


LocalStageSpec = Annotated[S4DLocalSpec | IdentityLocalSpec, Field(discriminator="kind")]


class LocalStage(nn.Module):
    """Samples -> one feature vector per sample, [S, width]."""

    width: int

    def forward(self, x: ChannelInputs) -> torch.Tensor:
        raise NotImplementedError


def local_stage(spec: S4DLocalSpec | IdentityLocalSpec) -> LocalStage:
    if isinstance(spec, S4DLocalSpec):
        return S4DLocalStage(spec)
    return IdentityLocalStage(spec)


def _inputs(x: ChannelInputs, dtype: torch.dtype) -> torch.Tensor:
    """[S, 3]: value (0 where hidden or not observed), observed, hidden."""
    observed = x.observed.to(dtype)
    return torch.stack([x.values.to(dtype) * observed, observed, x.hidden.to(dtype)], dim=-1)


class IdentityLocalStage(LocalStage):
    def __init__(self, spec: IdentityLocalSpec) -> None:
        super().__init__()
        self.width = spec.width
        self.embed = nn.Linear(3, spec.width)
        self.norm = nn.RMSNorm(spec.width)

    def forward(self, x: ChannelInputs) -> torch.Tensor:
        out: torch.Tensor = self.norm(self.embed(_inputs(x, self.embed.weight.dtype)))
        return out


@functools.lru_cache(maxsize=4096)
def fast_size(n: int) -> int:
    """The smallest m >= n whose prime factors are 2, 3, 5 and 7 (fast cuFFT / pocketfft)."""
    m = max(1, n)
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


@dataclass(frozen=True)
class _Bucket:
    """Channels of one rate and length, contiguous in the stage's sample order: the rows
    [start, start + channels x length) are a [channels, length, width] view. `runs`: blocks of
    co-located groups of one size, as (first channel in the bucket, groups G, size m), each a
    [G, m, length, width] view."""

    rate: float
    length: int
    start: int
    channels: int
    runs: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _Gathered:
    """Co-located channels of unequal lengths (at the end of a recording), m a time: index
    [P, m] into the stage's sample order (0 where missing) and valid [P, m]."""

    index: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class _Plan:
    order: torch.Tensor  # int64 [S]: the flat-buffer position of each sample in stage order
    inverse: torch.Tensor  # int64 [S]: the stage-order position of each flat-buffer sample
    buckets: tuple[_Bucket, ...]  # in stage order; together they hold every sample
    gathered: tuple[_Gathered, ...]
    rates: tuple[float, ...]
    longest: dict[float, int]  # rate -> its longest channel


def _plan(x: ChannelInputs) -> _Plan:
    """The stage's sample order, from the channel table (small: on the CPU, once a forward).
    Channels of one rate and length are put together (a bucket); in a bucket, the channels of
    one co-located group (one example, rate and first-sample time) are adjacent, the groups of
    one size together, the largest first, single channels last. Every layer then works on
    views: no gather, scatter or boolean mask per layer."""
    c = x.channels
    rates = c.rate_hz.tolist()
    lengths = c.length.tolist()
    offsets = c.offset.tolist()
    groups: dict[tuple[int, int, float, float], list[int]] = {}
    for k, key in enumerate(zip(c.row.tolist(), c.document.tolist(), rates,
                                c.start_s.tolist(), strict=True)):  # fmt: skip
        if lengths[k]:
            groups.setdefault(key, []).append(k)
    # per bucket (rate, length): the groups of equal length (size >= 2), then the rest
    viewed: dict[tuple[float, int], list[list[int]]] = {}
    single: dict[tuple[float, int], list[int]] = {}
    uneven: list[list[int]] = []
    for members in groups.values():
        key = (rates[members[0]], lengths[members[0]])
        if len(members) >= 2 and all(lengths[k] == key[1] for k in members):
            viewed.setdefault(key, []).append(members)
            continue
        if len(members) >= 2:
            uneven.append(members)
        for k in members:
            single.setdefault((rates[k], lengths[k]), []).append(k)
    order, buckets, position, start = [], [], {}, 0
    for key in sorted(set(viewed) | set(single)):
        rate, length = key
        sets = sorted(viewed.get(key, []), key=len, reverse=True)
        channels = [k for g in sets for k in g] + single.get(key, [])
        runs, at = [], 0
        for size in sorted({len(g) for g in sets}, reverse=True):
            count = sum(1 for g in sets if len(g) == size)
            runs.append((at, count, size))
            at += count * size
        for n, k in enumerate(channels):
            position[k] = start + n * length
            order.append(np.arange(offsets[k], offsets[k] + length))
        buckets.append(_Bucket(rate, length, start, len(channels), tuple(runs)))
        start += len(channels) * length
    device = x.values.device
    flat = np.concatenate(order) if order else np.zeros(0, dtype=np.int64)
    inverse = np.empty_like(flat)
    inverse[flat] = np.arange(flat.size)
    gathered = []
    for size in sorted({len(g) for g in uneven}):
        index, valid = [], []
        for members in (g for g in uneven if len(g) == size):
            span = max(lengths[k] for k in members)
            steps = np.arange(span)
            ok = np.stack([steps < lengths[k] for k in members], axis=1)
            index.append(np.where(ok, np.stack([position[k] + steps for k in members], 1), 0))
            valid.append(ok)
        gathered.append(_Gathered(torch.from_numpy(np.concatenate(index)).to(device),
                                  torch.from_numpy(np.concatenate(valid)).to(device)))  # fmt: skip
    longest: dict[float, int] = {}
    for b in buckets:
        longest[b.rate] = max(longest.get(b.rate, 0), b.length)
    return _Plan(
        order=torch.from_numpy(flat).to(device), inverse=torch.from_numpy(inverse).to(device),
        buckets=tuple(buckets), gathered=tuple(gathered), rates=tuple(sorted(longest)),
        longest=longest,
    )  # fmt: skip


# Lags per kernel piece: the Vandermonde matrix [2, width, states, piece] of one piece at a
# time, recomputed in the backward pass (memory only; the kernel is the same).
KERNEL_PIECE = 4096


class S4D(nn.Module):
    """The bidirectional diagonal SSM of one layer, in continuous time (see the module
    docstring)."""

    def __init__(self, spec: S4DLocalSpec) -> None:
        super().__init__()
        h, n = spec.width, spec.states
        self.spec = spec
        low, high = math.log(spec.min_time_s), math.log(spec.max_time_s)
        self.log_time = nn.Parameter(torch.rand(2, h) * (high - low) + low)  # Delta_h (s)
        self.log_neg_real = nn.Parameter(torch.full((2, h, n), math.log(0.5)))
        self.imag = nn.Parameter(math.pi * torch.arange(n).float().expand(2, h, n).clone())
        # C: complex normal with variance 1, as S4D (and the v3 local stage)
        self.c = nn.Parameter(torch.view_as_real(torch.randn(2, h, n, dtype=torch.cfloat)))
        self.d = nn.Parameter(torch.randn(h))

    def modes(self, rate: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(a * dt / Delta [2, H, N] complex, kept [2, H, N] bool) at `rate`."""
        a = torch.complex(-self.log_neg_real.exp(), self.imag)
        step = (1.0 / rate) / self.log_time.exp()[..., None]  # dt / Delta
        z = a * step
        return z, z.abs() <= math.pi

    def slowest(self, rates: tuple[float, ...]) -> torch.Tensor:
        """[rates]: the longest decay time, in samples, of the unmasked modes at each rate
        (0 if every mode is masked); no host sync."""
        with torch.no_grad():
            out = []
            for rate in rates:
                z, kept = self.modes(rate)
                decay = (-z.real).clamp(min=1e-12)
                out.append(torch.where(kept, 1.0 / decay, 0.0).max())
            return torch.stack(out)

    def kernel_length(self, slowest: float, longest: int) -> int:
        """At most `kernel_time_constants` decay times of the slowest unmasked mode, and at
        most the longest channel."""
        return max(1, min(longest, math.ceil(self.spec.kernel_time_constants * slowest)))

    def length_at(self, rate: float, longest: int) -> int:
        """`kernel_length` at one rate (a host sync; for tests and checks)."""
        return self.kernel_length(float(self.slowest((rate,))[0]), longest)

    def kernel(self, rate: float, length: int) -> torch.Tensor:
        """[2, H, length]: the kernel of each direction at `rate` (float32 for float32
        weights)."""
        z, kept = self.modes(rate)
        a = torch.complex(-self.log_neg_real.exp(), self.imag)
        c = torch.view_as_complex(self.c) * kept
        kind = self.spec.discretization
        if kind == "zoh":
            weight, power, first = c * (z.exp() - 1.0) / a, z, None
        elif kind == "zoh_centred":
            weight, power = c * ((z / 2).exp() - (-z / 2).exp()) / a, z
            first = 2 * (c * ((z / 2).exp() - 1.0) / a).sum(-1).real  # lag 0: half a sample
        else:
            ratio = (1 + z / 2) / (1 - z / 2)
            weight, power, first = c * (z / a) / (1 - z / 2), ratio.log(), None
        pieces = []
        for start in range(0, length, KERNEL_PIECE):
            stop = min(length, start + KERNEL_PIECE)
            if torch.is_grad_enabled():
                piece = checkpoint(_kernel_piece, weight, power, start, stop, use_reentrant=False)
            else:
                piece = _kernel_piece(weight, power, start, stop)
            pieces.append(piece)
        k = torch.cat(pieces, dim=-1)
        if first is not None:
            k = torch.cat([first[..., None].to(k.dtype), k[..., 1:]], dim=-1)
        if self.spec.masked_modes == "static_gain":
            masked = torch.view_as_complex(self.c) * ~kept
            static = 2 * (masked * (-1.0 / a)).sum(-1).real  # [2, H]
            k = torch.cat([k[..., :1] + static[..., None].to(k.dtype), k[..., 1:]], dim=-1)
        return k

    def forward(self, z: torch.Tensor, size: int, kernel: torch.Tensor) -> torch.Tensor:
        """`z` [B, H, L] float32 (contiguous) -> [B, H, L]: both directions over the lags
        -(K-1)..K-1 as one circular kernel of `size` >= L + K - 1, so nothing wraps onto a
        used output."""
        length, k = z.shape[-1], kernel.shape[-1]
        both = z.new_zeros(z.shape[1], size)
        both[:, :k] = kernel[0]
        if k > 1:
            both[:, size - k + 1 :] = kernel[1, :, 1:].flip(-1)
        both[:, 0] = both[:, 0] + kernel[1, :, 0]
        y = torch.fft.irfft(torch.fft.rfft(z, size) * torch.fft.rfft(both, size), size)
        out: torch.Tensor = y[..., :length] + z * self.d[:, None]
        return out


def _kernel_piece(weight: torch.Tensor, power: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """2 Re sum_n weight_n exp(power_n l), l in [start, stop): [2, H, stop - start]. The
    phase power.imag * l in float64 (in float32 it is off by ~1e-3 rad at l = 1e4)."""
    lags = torch.arange(start, stop, device=power.device, dtype=torch.float64)
    phase = torch.remainder(power.imag.double()[..., None] * lags, 2 * math.pi)
    magnitude = (power.real[..., None] * lags.to(power.real.dtype)).exp()
    vandermonde = torch.polar(magnitude, phase.to(magnitude.dtype))  # [2, H, N, l]
    return 2 * torch.einsum("dhn,dhnl->dhl", weight, vandermonde).real


class _Layer(nn.Module):
    def __init__(self, spec: S4DLocalSpec, mixing: bool) -> None:
        super().__init__()
        h = spec.width
        self.dropout = spec.dropout
        self.norm = nn.RMSNorm(h)
        self.ssm = S4D(spec)
        self.out = nn.Linear(h, 2 * h)
        self.mixing = ChannelMixing(h, spec.mixing_heads, spec.dropout) if mixing else None

    def forward(self, h: torch.Tensor, plan: _Plan, kernels: dict[float, int]) -> torch.Tensor:
        """`h` [S, width] in the plan's sample order; `kernels`: the kernel length per rate."""
        width = h.shape[-1]
        with torch.autocast(h.device.type, enabled=False):  # the SSM in float32
            u = self.norm(h.to(torch.promote_types(h.dtype, torch.float32)))
            kernel = {r: self.ssm.kernel(r, n) for r, n in kernels.items()}
            parts = []
            for b in plan.buckets:
                rows = u[b.start : b.start + b.channels * b.length]
                z = rows.view(b.channels, b.length, width).transpose(1, 2).contiguous()
                used = min(b.length, kernels[b.rate])
                size = fast_size(b.length + used - 1)
                y = F.gelu(self.ssm(z, size, kernel[b.rate][..., :used]))
                parts.append(y.transpose(1, 2).reshape(-1, width))
            y = torch.cat(parts) if len(parts) > 1 else parts[0]
        y = F.glu(self.out(y), dim=-1)
        h = h + F.dropout(y, self.dropout, self.training)
        if self.mixing is not None:
            h = self.mixing(h, plan)
        return h


# Sample times per attention call of the channel mixing (the batch of one call is this times
# the heads): within the grid limits of the fused attention kernels.
MIXING_CHUNK = 16384


class ChannelMixing(nn.Module):
    """Attention across co-located channels at each sample time (pre-norm, residual)."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width / mixing_heads must be whole")
        self.heads, self.dropout = heads, dropout
        self.norm = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def attend(self, x: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
        """`x` [P, m, width] (P sample times, m channels) -> [P, m, width]; `keep` [P, m]:
        the channels that exist (None: all)."""
        p, m, width = x.shape
        q, k, v = self.qkv(self.norm(x)).view(p, m, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        outs = []
        for at in range(0, p, MIXING_CHUNK):
            part = slice(at, at + MIXING_CHUNK)
            mask = None if keep is None else keep[part, None, None, :]
            outs.append(F.scaled_dot_product_attention(q[part], k[part], v[part], attn_mask=mask))
        a = torch.cat(outs) if len(outs) > 1 else outs[0]
        out: torch.Tensor = self.out(a.transpose(1, 2).reshape(p, m, width))
        return out

    def forward(self, h: torch.Tensor, plan: _Plan) -> torch.Tensor:
        width = h.shape[-1]
        parts, at = [], 0  # the update, in sample order: mixed runs and zeros between them
        for b in plan.buckets:
            for first, count, size in b.runs:
                start = b.start + first * b.length
                if start > at:
                    parts.append(h.new_zeros(start - at, width))
                block = h[start : start + count * size * b.length]
                x = block.view(count, size, b.length, width).transpose(1, 2)  # [G, L, m, H]
                a = self.attend(x.reshape(count * b.length, size, width), None)
                a = a.view(count, b.length, size, width).transpose(1, 2).reshape(-1, width)
                parts.append(a.to(h.dtype))
                at = start + count * size * b.length
        if not parts and not plan.gathered:
            return h
        if at < h.shape[0]:
            parts.append(h.new_zeros(h.shape[0] - at, width))
        update = torch.cat(parts) if len(parts) > 1 else parts[0]
        for g in plan.gathered:
            a = self.attend(h[g.index], g.valid)
            update = update.index_put((g.index[g.valid],), a[g.valid].to(h.dtype))
        return h + F.dropout(update, self.dropout, self.training)


class S4DLocalStage(LocalStage):
    def __init__(self, spec: S4DLocalSpec) -> None:
        super().__init__()
        if spec.min_time_s <= 0 or spec.max_time_s < spec.min_time_s:
            raise ValueError("0 < min_time_s <= max_time_s")
        if not all(0 <= k < spec.layers for k in spec.mixing_layers):
            raise ValueError(f"mixing_layers are in 0..{spec.layers - 1}")
        self.spec = spec
        self.width = spec.width
        self.embed = nn.Linear(3, spec.width)
        self.layers = nn.ModuleList(
            _Layer(spec, k in spec.mixing_layers) for k in range(spec.layers)
        )
        self.norm = nn.RMSNorm(spec.width)

    def kernel_lengths(self, plan: _Plan) -> list[dict[float, int]]:
        """Per layer, the kernel length per rate: one host sync a forward."""
        if not plan.rates:
            return [{} for _ in self.layers]
        slowest = torch.stack([layer.ssm.slowest(plan.rates) for layer in self.layers]).tolist()
        return [
            {r: layer.ssm.kernel_length(s, plan.longest[r])
             for r, s in zip(plan.rates, row, strict=True)}
            for layer, row in zip(self.layers, slowest, strict=True)
        ]  # fmt: skip

    def forward(self, x: ChannelInputs) -> torch.Tensor:
        plan = _plan(x)
        kernels = self.kernel_lengths(plan)
        h = self.embed(_inputs(x, self.embed.weight.dtype)[plan.order])
        for layer, lengths in zip(self.layers, kernels, strict=True):
            if self.spec.checkpoint and torch.is_grad_enabled():
                h = checkpoint(layer, h, plan, lengths, use_reentrant=False)
            else:
                h = layer(h, plan, lengths)
        out: torch.Tensor = self.norm(h)[plan.inverse]
        return out
