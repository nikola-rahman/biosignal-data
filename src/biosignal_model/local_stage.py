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
- channels of one rate are put in buckets of one FFT size (length buckets), each a padded
  [channels, width, size] tensor; padding is zero, so a channel's output does not depend on
  the others.

Channel mixing: channels of one example with the same rate and first-sample time share a time
base (ECG leads, EEG electrodes, the EMG grid); at each sample time, multi-head attention
across them (pre-norm, residual), with no channel identity: the stage is equivariant to the
order of the channels. Channels on another time base are mixed later, by the global model.
For grids, a neighbour convolution needs the grid positions (descriptors): a later step.

Activation checkpointing per layer (`checkpoint`), so the memory is one layer's activations
plus one [S, width] tensor a layer.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Annotated, Literal

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


@dataclass
class _MixGroup:
    """Co-located channel sets of one size m: at each of P sample times, m channels."""

    index: torch.Tensor  # int64 [P, m]
    valid: torch.Tensor  # bool [P, m]


@dataclass
class _Plan:
    # rate -> buckets of (longest length, index [B, L] into the flat buffer, valid [B, L])
    buckets: dict[float, list[tuple[int, torch.Tensor, torch.Tensor]]]
    mixing: list[_MixGroup]


def _plan(x: ChannelInputs) -> _Plan:
    """The buckets of the SSM and the groups of channel mixing, from the channel table (on
    the CPU: a small table)."""
    c = x.channels
    rates = c.rate_hz.tolist()
    lengths = c.length.tolist()
    offsets = c.offset.tolist()
    device = x.values.device
    by_rate: dict[float, list[int]] = {}
    for k, r in enumerate(rates):
        if lengths[k]:
            by_rate.setdefault(r, []).append(k)
    buckets: dict[float, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
    for r, ks in by_rate.items():
        # one bucket per FFT size of the longest kernel a channel can have (its length)
        by_size: dict[int, list[int]] = {}
        for k in ks:
            by_size.setdefault(fast_size(2 * lengths[k] - 1), []).append(k)
        buckets[r] = []
        for group in by_size.values():
            span = max(lengths[k] for k in group)
            index, valid = _padded([offsets[k] for k in group], [lengths[k] for k in group],
                                   device)  # fmt: skip
            buckets[r].append((span, index, valid))
    groups: dict[tuple[int, int, float, float], list[int]] = {}
    for k, key in enumerate(zip(c.row.tolist(), c.document.tolist(), rates,
                                c.start_s.tolist(), strict=True)):  # fmt: skip
        if lengths[k]:
            groups.setdefault(key, []).append(k)
    sets: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        span = max(lengths[k] for k in members)
        steps = torch.arange(span)
        index = torch.stack([offsets[k] + steps for k in members], dim=1)  # [P, m]
        valid = torch.stack([steps < lengths[k] for k in members], dim=1)
        found = sets.setdefault(len(members), ([], []))
        found[0].append(torch.where(valid, index, 0))
        found[1].append(valid)
    mixing = [_MixGroup(torch.cat(i).to(device), torch.cat(v).to(device)) for i, v in sets.values()]
    return _Plan(buckets, mixing)


def _padded(
    offsets: list[int], lengths: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    span = max(lengths)
    steps = torch.arange(span)
    index = torch.tensor(offsets)[:, None] + steps
    valid = steps[None, :] < torch.tensor(lengths)[:, None]
    return torch.where(valid, index, 0).to(device), valid.to(device)


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

    def kernel_length(self, rate: float, longest: int) -> int:
        """At most `kernel_time_constants` decay times of the slowest unmasked mode."""
        with torch.no_grad():
            z, kept = self.modes(rate)
            decay = (-z.real).clamp(min=1e-12)  # per sample
            slowest = float(torch.where(kept, 1.0 / decay, 0.0).max())
        return max(1, min(longest, math.ceil(self.spec.kernel_time_constants * slowest)))

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

    def forward(
        self, z: torch.Tensor, length: int, size: int, kernel: torch.Tensor
    ) -> torch.Tensor:
        """`z` [B, H, L] float32 -> [B, H, L]: both directions over the lags -(K-1)..K-1 as one
        circular kernel of `size` >= L + K - 1, so nothing wraps onto a used output."""
        k = kernel.shape[-1]
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

    def forward(self, h: torch.Tensor, plan: _Plan) -> torch.Tensor:
        with torch.autocast(h.device.type, enabled=False):  # the SSM in float32
            u = self.norm(h.to(torch.promote_types(h.dtype, torch.float32)))
            y = torch.zeros_like(u)
            for rate, buckets in plan.buckets.items():
                longest = max(length for length, _, _ in buckets)
                k = self.ssm.kernel_length(rate, longest)
                kernel = self.ssm.kernel(rate, k)
                for length, index, valid in buckets:
                    used = min(length, k)
                    z = (u[index] * valid[..., None]).transpose(1, 2)  # [B, H, L], 0 padding
                    out = self.ssm(z, length, fast_size(length + used - 1), kernel[..., :used])
                    y[index[valid]] = out.transpose(1, 2)[valid]
            y = F.gelu(y)
        y = F.glu(self.out(y), dim=-1)
        h = h + F.dropout(y, self.dropout, self.training)
        if self.mixing is not None:
            h = self.mixing(h, plan.mixing)
        return h


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

    def forward(self, h: torch.Tensor, groups: list[_MixGroup]) -> torch.Tensor:
        if not groups:
            return h
        update = torch.zeros_like(h)
        for g in groups:
            p, m = g.index.shape
            x = self.norm(h[g.index])  # [P, m, H]
            q, k, v = self.qkv(x).view(p, m, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
            keep = g.valid[:, None, None, :]  # [P, 1, 1, m]: keys that exist
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
            a = self.out(a.transpose(1, 2).reshape(p, m, -1))
            update[g.index[g.valid]] = a[g.valid].to(update.dtype)
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

    def forward(self, x: ChannelInputs) -> torch.Tensor:
        plan = _plan(x)
        h = self.embed(_inputs(x, self.embed.weight.dtype))
        for layer in self.layers:
            if self.spec.checkpoint and torch.is_grad_enabled():
                h = checkpoint(layer, h, plan, use_reentrant=False)
            else:
                h = layer(h, plan)
        out: torch.Tensor = self.norm(h)
        return out
