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
in length (at a recording's end) is gathered, with a mask. Channels on another time base are
mixed later, by the global model. For grids, a neighbour convolution needs the grid positions
(descriptors): a later step.

Activation checkpointing per layer (`checkpoint`), so the memory is one layer's activations
plus one [S, width] tensor a layer.

Speed (`convolution`, `compile`): `scan` computes the SSM from its recurrence on the GPU
(biosignal_model.ssm_scan: one Triton pass a direction in the residual stream's layout, the
mode parameters' gradients from the adjoint pass, no FFT and no kernel in the time domain;
D u is folded into the lag-0 weight); `fft` is the reference and runs anywhere. `compile`
fuses the elementwise chains (norms, GELU, GLU, residuals, the transposes of the mixing).
Measured 2026-10-29 on an L40S (ECG 32 x 12 x 500 Hz x 10 s, width 128, a training step):
fft eager 1.65 s (1.14 M samples/s), scan + compile 0.44 s without checkpointing; outputs and
gradients against float64 as exact as the FFT path (tests/test_local_stage_gpu.py).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
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
    # how the convolution is computed: fft (any device; the kernel capped as above) or scan
    # (the recurrence on the GPU, biosignal_model.ssm_scan; the kernel is not capped: the
    # same convolution as fft wherever the cap does not cut it, and the untruncated one where it
    # would)
    convolution: Literal["fft", "scan"]
    masked_modes: Literal["drop", "static_gain"]  # what a mode above |lambda| dt > pi adds
    mixing_layers: list[int]  # the layers (from 0) followed by channel mixing
    mixing_heads: int
    dropout: float  # on the residual branches
    checkpoint: bool  # activation checkpointing per layer
    # torch.compile of the elementwise chains around the FFT and the attention (the same
    # operations; outputs within float32 rounding of the eager ones)
    compile: bool


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
    for k, base in enumerate(zip(c.row.tolist(), c.document.tolist(), rates,
                                 c.start_s.tolist(), strict=True)):  # fmt: skip
        if lengths[k]:
            groups.setdefault(base, []).append(k)
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

    def _discretized(self, rate: float) -> tuple[Any, ...]:
        """(weight, power, lag 0 or None, kept, a, c): the kernel at rate is, per direction
        and feature, 2 Re sum_n weight_n exp(power_n l), with lag 0 replaced (when not None)."""
        z, kept = self.modes(rate)
        # a masked mode adds nothing below: its z set to 0 keeps exp(-z / 2) finite (at 1 Hz a
        # 1 ms mode has Re z = -500, exp(250) is inf in float32, and inf x 0 is NaN)
        z = torch.where(kept, z, torch.zeros_like(z))
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
        return weight, power, first, kept, a

    def _lag0(self, weight: Any, first: Any, kept: Any, a: Any) -> torch.Tensor:
        """[2, H]: the kernel at lag 0 (each direction): 2 Re sum_n weight_n, or the
        discretization's own lag 0, plus the static gains of the masked modes."""
        k0 = 2 * weight.sum(-1).real if first is None else first
        if self.spec.masked_modes == "static_gain":
            masked = torch.view_as_complex(self.c) * ~kept
            k0 = k0 + 2 * (masked * (-1.0 / a)).sum(-1).real
        out: torch.Tensor = k0
        return out

    def scan_parameters(self, rate: float) -> tuple[torch.Tensor, ...]:
        """(k0 [2, H], w [2, H, N, 2], p [2, H, N, 2], kept [2, H, N]) at `rate`, with
        gradients: the lag-0 weight, and real views of the weights and log-modes of the lags
        from 1 on (biosignal_model.ssm_scan.ScanConv)."""
        weight, power, first, kept, a = self._discretized(rate)
        k0 = self._lag0(weight, first, kept, a)
        return k0, torch.view_as_real(weight), torch.view_as_real(power), kept

    def kernel(self, rate: float, length: int) -> torch.Tensor:
        """[2, H, length]: the kernel of each direction at `rate` (float32 for float32
        weights)."""
        weight, power, first, kept, a = self._discretized(rate)
        pieces = []
        for start in range(0, length, KERNEL_PIECE):
            stop = min(length, start + KERNEL_PIECE)
            if torch.is_grad_enabled():
                piece = checkpoint(_kernel_piece, weight, power, start, stop, use_reentrant=False)
            else:
                piece = _kernel_piece(weight, power, start, stop)
            pieces.append(piece)
        k = torch.cat(pieces, dim=-1)
        if first is not None or self.spec.masked_modes == "static_gain":
            k0 = self._lag0(weight, first, kept, a)
            k = torch.cat([k0[..., None].to(k.dtype), k[..., 1:]], dim=-1)
        return k

    def forward(self, z: torch.Tensor, size: int, kernel: torch.Tensor) -> torch.Tensor:
        """`z` [B, H, L] float32 (contiguous) -> [B, H, L] (a view): the convolution with both
        directions over the lags -(K-1)..K-1 as one circular kernel of `size` >= L + K - 1, so
        nothing wraps onto a used output (the D term is added by the caller)."""
        k = kernel.shape[-1]
        both = z.new_zeros(z.shape[1], size)
        both[:, :k] = kernel[0]
        if k > 1:
            both[:, size - k + 1 :] = kernel[1, :, 1:].flip(-1)
        both[:, 0] = both[:, 0] + kernel[1, :, 0]
        return cast(torch.Tensor, _FFTConv.apply(z, both, size))


class _FFTConv(torch.autograd.Function):
    """y = irfft(rfft(z, n) rfft(c, n), n)[..., :L]: the circular convolution of z [B, H, L]
    with the real circular kernel c [H, n]. The forward is the eager one; the backward is
    written out, so no zero-filled gradient of the slice, no conj kernels: with
    G = rfft(dy, n) (zero-padded), dz = irfft(G conj(C), n)[..., :L] and
    dc = irfft(sum over B of G conj(Z), n)."""

    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, kernel: torch.Tensor, size: int) -> torch.Tensor:
        spectrum = torch.fft.rfft(z, size)
        kernel_f = torch.fft.rfft(kernel, size)
        ctx.save_for_backward(spectrum, kernel_f)
        ctx.size, ctx.length = size, z.shape[-1]
        out: torch.Tensor = torch.fft.irfft(spectrum * kernel_f, size)[..., : z.shape[-1]]
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        spectrum, kernel_f = ctx.saved_tensors
        g = torch.fft.rfft(grad, ctx.size)
        dz = dk = None
        if ctx.needs_input_grad[0]:
            dz = torch.fft.irfft(g * kernel_f.conj(), ctx.size)[..., : ctx.length]
        if ctx.needs_input_grad[1]:
            dk = torch.fft.irfft((g * spectrum.conj()).sum(0), ctx.size)
        return dz, dk, None


def _kernel_piece(weight: torch.Tensor, power: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """2 Re sum_n weight_n exp(power_n l), l in [start, stop): [2, H, stop - start]. The
    phase power.imag * l in float64 (in float32 it is off by ~1e-3 rad at l = 1e4)."""
    lags = torch.arange(start, stop, device=power.device, dtype=torch.float64)
    phase = torch.remainder(power.imag.double()[..., None] * lags, 2 * math.pi)
    magnitude = (power.real[..., None] * lags.to(power.real.dtype)).exp()
    vandermonde = torch.polar(magnitude, phase.to(magnitude.dtype))  # [2, H, N, l]
    return 2 * torch.einsum("dhn,dhnl->dhl", weight, vandermonde).real


def _to_channels(rows: torch.Tensor, weight: torch.Tensor, channels: int,
                 length: int) -> torch.Tensor:  # fmt: skip
    """[channels x length, H] -> RMS-normed float32 [channels, H, length], contiguous."""
    x = F.rms_norm(rows.to(torch.promote_types(rows.dtype, torch.float32)), (rows.shape[-1],),
                   weight)  # fmt: skip
    return x.view(channels, length, -1).transpose(1, 2).contiguous()


def _from_channels(y: torch.Tensor, z: torch.Tensor, d: torch.Tensor,
                   dtype: torch.dtype) -> torch.Tensor:  # fmt: skip
    """GELU(y + D z), [channels, H, length] -> [channels x length, H] in `dtype` (the next
    linear map's)."""
    out = F.gelu(y + z * d[:, None])
    return out.transpose(1, 2).reshape(-1, out.shape[1]).to(dtype)


def _normed(rows: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm of the rows [samples, H] in float32 (the scan's input)."""
    x = rows.to(torch.promote_types(rows.dtype, torch.float32))
    return F.rms_norm(x, (x.shape[-1],), weight).contiguous()


def _activated(y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """GELU(y) of rows [samples, H] (the scan's y holds D u), in `dtype` (the next linear
    map's)."""
    return F.gelu(y).to(dtype)


def _dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """F.dropout, and nothing at p = 0 (F.dropout then still copies)."""
    return F.dropout(x, p, training) if p > 0 and training else x


def _glu_residual(h: torch.Tensor, y: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    return h + _dropout(F.glu(y, dim=-1), p, training)


def _mix_in(block: torch.Tensor, weight: torch.Tensor, groups: int, size: int,
            length: int) -> torch.Tensor:  # fmt: skip
    """[G x m x L, H] (G groups of m channels) -> RMS-normed [G x L, m, H]."""
    x = block.view(groups, size, length, -1).transpose(1, 2).reshape(groups * length, size, -1)
    return F.rms_norm(x, (x.shape[-1],), weight)


def _mix_out(block: torch.Tensor, a: torch.Tensor, groups: int, size: int, length: int,
             p: float, training: bool) -> torch.Tensor:  # fmt: skip
    """The residual: block [G x m x L, H] + the attention output a [G x L, m, H]."""
    a = a.view(groups, length, size, -1).transpose(1, 2).reshape(-1, a.shape[-1])
    return block + _dropout(a.to(block.dtype), p, training)


def _split(h: torch.Tensor, plan: _Plan) -> tuple[torch.Tensor, ...]:
    """The rows of each bucket (views; the backward is one cat, not a zero-filled gradient a
    slice)."""
    sizes = [b.channels * b.length for b in plan.buckets]
    return (h,) if len(sizes) == 1 else tuple(torch.split(h, sizes))


_COMPILED: dict[object, object] = {}


def _fn(fn: object, compiled: bool) -> object:
    """`fn` itself, or its torch.compile (dynamic shapes: the buckets change from batch to
    batch; one compilation for every layer)."""
    if not compiled:
        return fn
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn, dynamic=True, fullgraph=True)  # type: ignore[call-overload]
    return _COMPILED[fn]


class _Layer(nn.Module):
    def __init__(self, spec: S4DLocalSpec, mixing: bool) -> None:
        super().__init__()
        h = spec.width
        self.dropout = spec.dropout
        self.compiled = spec.compile
        self.norm = nn.RMSNorm(h)
        self.ssm = S4D(spec)
        self.out = nn.Linear(h, 2 * h)
        self.mixing = (
            ChannelMixing(h, spec.mixing_heads, spec.dropout, spec.compile) if mixing else None
        )

    def _scan_parameters(self, rate: float) -> tuple[torch.Tensor, ...]:
        """The scan's (k0, w, p, kept) at `rate`, with D in the forward direction's lag 0
        (D u is a lag-0 term)."""
        k0, w, p, kept = self.ssm.scan_parameters(rate)
        d = self.ssm.d.to(k0.dtype)
        return k0 + torch.stack([d, torch.zeros_like(d)]), w, p, kept

    def forward(self, h: torch.Tensor, plan: _Plan, kernels: dict[float, int]) -> torch.Tensor:
        """`h` [S, width] in the plan's sample order; `kernels`: the kernel length per rate."""
        to_channels: Any = _fn(_to_channels, self.compiled)
        normed: Any = _fn(_normed, self.compiled)
        activated: Any = _fn(_activated, self.compiled)
        from_channels: Any = _fn(_from_channels, self.compiled)
        glu_residual: Any = _fn(_glu_residual, self.compiled)
        # the input of the linear map: in the autocast dtype under autocast (as autocast casts)
        dtype = (torch.get_autocast_dtype(h.device.type) if torch.is_autocast_enabled(h.device.type)
                 else torch.promote_types(h.dtype, torch.float32))  # fmt: skip
        with torch.autocast(h.device.type, enabled=False):  # the SSM in float32
            parts = []
            scan = self.ssm.spec.convolution == "scan"
            if scan:  # Triton, on the GPU: imported only where it runs
                from biosignal_model.ssm_scan import ScanConv  # noqa: PLC0415
            kernel = {} if scan else {r: self.ssm.kernel(r, n) for r, n in kernels.items()}
            modes = {r: self._scan_parameters(r) for r in kernels} if scan else {}
            for b, rows in zip(plan.buckets, _split(h, plan), strict=True):
                if scan:
                    u = normed(rows, self.norm.weight)  # [B x L, H] float32
                    y = cast(torch.Tensor, ScanConv.apply(
                        u.view(b.channels, b.length, -1), *modes[b.rate]
                    ))  # fmt: skip
                    parts.append(activated(y.view(-1, y.shape[-1]), dtype))
                    continue
                z = to_channels(rows, self.norm.weight, b.channels, b.length)
                used = min(b.length, kernels[b.rate])
                y = self.ssm(z, fast_size(b.length + used - 1), kernel[b.rate][..., :used])
                parts.append(from_channels(y, z, self.ssm.d, dtype))
            y = torch.cat(parts) if len(parts) > 1 else parts[0]
        h = glu_residual(h, self.out(y), self.dropout, self.training)
        if self.mixing is not None:
            h = self.mixing(h, plan)
        return h


# Sample times per attention call of the channel mixing (the batch of one call is this times
# the heads): within the grid limits of the fused attention kernels.
MIXING_CHUNK = 16384
# The attention kernels of the channel mixing: the memory-efficient (cutlass) kernel on the GPU,
# measured 2026-10-29 on an L40S, forward + backward, bf16, 4 heads of 32: 134 ms at 160 k
# sample times x 12 channels (flash 213, math 192, a batched matmul 129), 61 ms at 32 k x 64
# (flash 79, math 168, matmul 126); the math kernel elsewhere (CPU, float64).
MIXING_KERNELS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class ChannelMixing(nn.Module):
    """Attention across co-located channels at each sample time (pre-norm, residual)."""

    def __init__(self, width: int, heads: int, dropout: float, compiled: bool) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width / mixing_heads must be whole")
        self.heads, self.dropout, self.compiled = heads, dropout, compiled
        self.norm = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def attend(self, x: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
        """`x` [P, m, width] normed (P sample times, m channels) -> [P, m, width]; `keep`
        [P, m]: the channels that exist (None: all)."""
        _, m, width = x.shape
        # chunks split off before the maps (the backward of a split is one cat; slices of one
        # q, k, v would each get a zero-filled gradient of the whole and be added up)
        chunks = torch.split(x, MIXING_CHUNK)
        masks = (None,) * len(chunks) if keep is None else torch.split(keep, MIXING_CHUNK)
        outs = []
        for part, mask in zip(chunks, masks, strict=True):
            p = part.shape[0]
            q, k, v = self.qkv(part).view(p, m, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
            with sdpa_kernel(MIXING_KERNELS):
                a = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None if mask is None else mask[:, None, None, :]
                )
            outs.append(self.out(a.transpose(1, 2).reshape(p, m, width)))
        out: torch.Tensor = torch.cat(outs) if len(outs) > 1 else outs[0]
        return out

    def forward(self, h: torch.Tensor, plan: _Plan) -> torch.Tensor:
        mix_in: Any = _fn(_mix_in, self.compiled)
        mix_out: Any = _fn(_mix_out, self.compiled)
        # the rows in order: (start, stop, run or None for rows mixed only by gathering)
        spans: list[tuple[int, int, tuple[_Bucket, int, int] | None]] = []
        at = 0
        for b in plan.buckets:
            for first, count, size in b.runs:
                start = b.start + first * b.length
                if start > at:
                    spans.append((at, start, None))
                at = start + count * size * b.length
                spans.append((start, at, (b, count, size)))
        if not spans and not plan.gathered:
            return h
        if at < h.shape[0]:
            spans.append((at, h.shape[0], None))
        sizes = [stop - start for start, stop, _ in spans]
        blocks = tuple(torch.split(h, sizes)) if len(spans) > 1 else (h,)
        pieces = []
        for block, (_, _, run) in zip(blocks, spans, strict=True):
            if run is None:
                pieces.append(block)
                continue
            b, count, size = run
            a = self.attend(mix_in(block, self.norm.weight, count, size, b.length), None)
            pieces.append(mix_out(block, a, count, size, b.length, self.dropout, self.training))
        out = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        for g in plan.gathered:
            a = self.attend(self.norm(h[g.index]), g.valid)
            rows = g.index[g.valid]
            mixed = h[rows] + _dropout(a[g.valid].to(h.dtype), self.dropout, self.training)
            out = out.index_put((rows,), mixed)
        return out


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
        if self.spec.convolution == "scan":  # no cap: every lag of the longest channel
            return [dict(plan.longest) for _ in self.layers]
        ssms = [cast(_Layer, layer).ssm for layer in self.layers]
        slowest = torch.stack([ssm.slowest(plan.rates) for ssm in ssms]).tolist()
        return [
            {r: ssm.kernel_length(s, plan.longest[r])
             for r, s in zip(plan.rates, row, strict=True)}
            for ssm, row in zip(ssms, slowest, strict=True)
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
