"""The bidirectional diagonal SSM of the local stage as a scan on the GPU (Triton): the same
convolution as the FFT path, computed from its recurrence, with its exact gradients.

A direction with modes A_n = exp(p_n) (complex), weights w_n and lag-0 weight k0 has the kernel
k[0] = k0, k[l] = 2 Re sum_n w_n A_n^l (l >= 1), so with the state s_n[t] = A_n (s_n[t-1] +
u[t-1]) = sum_{l >= 1} A_n^l u[t - l]:

    y[t] = k0 u[t] + 2 Re sum_n w_n s_n[t]

forward in time for the forward direction, backward in time for the other. One program holds
the states of `BLOCK` features of one channel and walks its samples once: it reads the input
once and writes the output once, in the [channels, length, features] layout of the residual
stream (no transposes, no FFT buffers, no kernel in the time domain). Measured 2026-10-29 on an
L40S (384 channels x 5000 samples x 128 features, 32 modes): 9 ms a direction against 48 ms for
the float32 FFT convolution, the same error against float64 (2.2e-7 relative).

Gradients (`ScanConv`), with g = dL/dy:

- the input: the transposed operator, the same scan with each direction's weights running the
  other way (the adjoint of a convolution is the correlation);
- the mode parameters, without the kernel's gradient dk[l] = sum_t g[t] u[t - l]:
  S_n = sum_l dk[l] A_n^l = sum_t g[t] s_n[t] and R_n = sum_l dk[l] l A_n^l = sum_t g[t] r_n[t],
  with r_n[t] = sum_{l >= 1} l A_n^l u[t - l] = A_n (r_n[t-1] + s_n[t-1] + u[t-1]); then
  dL/dRe w = 2 Re S, dL/dIm w = -2 Im S, dL/dRe p = 2 Re(w R), dL/dIm p = -2 Im(w R). In the
  same pass as the input's gradient: S = sum_t u[t] lam[t] and R = sum_t u[t] rho[t] with the
  adjoint states lam[t] = sum_{l >= 1} A^l g[t + l] and rho[t] = sum_{l >= 1} l A^l g[t + l]
  (one kernel a direction for the whole backward);
- the lag-0 weight: sum_t g[t] u[t].
"""

from __future__ import annotations

from typing import Any

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

# Features a program holds (its states: BLOCK x modes complex numbers), and warps a program,
# for the forward scan and for the backward (more states).
# Measured 2026-10-29 (one ECG step, L40S): flat within 6 % over BLOCK 16-64 and 2-8 warps.
SCAN_LAUNCH = {"BLOCK": 64, "num_warps": 4}
ADJOINT_LAUNCH = {"BLOCK": 32, "num_warps": 4}


@triton.jit  # type: ignore[untyped-decorator]
def _scan(  # type: ignore[no-untyped-def]
    u_ptr, y_ptr, ar_ptr, ai_ptr, wr_ptr, wi_ptr, k0_ptr, length, width,
    N: tl.constexpr, BLOCK: tl.constexpr,  # noqa: N803 - Triton constexpr names
    REVERSE: tl.constexpr, ACCUMULATE: tl.constexpr,  # noqa: N803
):  # fmt: skip
    b = tl.program_id(0)
    hs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ns = tl.arange(0, N)
    mh = hs < width
    p = hs[:, None] * N + ns[None, :]
    ar = tl.load(ar_ptr + p, mask=mh[:, None], other=0.0)
    ai = tl.load(ai_ptr + p, mask=mh[:, None], other=0.0)
    wr = tl.load(wr_ptr + p, mask=mh[:, None], other=0.0)
    wi = tl.load(wi_ptr + p, mask=mh[:, None], other=0.0)
    k0 = tl.load(k0_ptr + hs, mask=mh, other=0.0)
    sr = tl.zeros([BLOCK, N], dtype=tl.float32)
    si = tl.zeros([BLOCK, N], dtype=tl.float32)
    prev = tl.zeros([BLOCK], dtype=tl.float32)
    base = b.to(tl.int64) * length * width
    first = length - 1 if REVERSE else 0
    following = tl.load(u_ptr + base + first * width + hs, mask=mh, other=0.0)
    for i in range(length):
        t = length - 1 - i if REVERSE else i
        u = following  # loaded a step ahead (the load does not wait for the states)
        ahead = tl.maximum(t - 1, 0) if REVERSE else tl.minimum(t + 1, length - 1)
        following = tl.load(u_ptr + base + ahead * width + hs, mask=mh, other=0.0)
        xr = sr + prev[:, None]
        sr, si = ar * xr - ai * si, ar * si + ai * xr
        y = k0 * u + 2.0 * tl.sum(wr * sr - wi * si, axis=1)
        if ACCUMULATE:
            y += tl.load(y_ptr + base + t * width + hs, mask=mh, other=0.0)
        tl.store(y_ptr + base + t * width + hs, y, mask=mh)
        prev = u


@triton.jit  # type: ignore[untyped-decorator]
def _adjoint(  # type: ignore[no-untyped-def]  # noqa: PLR0915 - one fused pass
    u_ptr, g_ptr, du_ptr, ar_ptr, ai_ptr, wr_ptr, wi_ptr, k0_ptr, sums_ptr, k0g_ptr, length,
    width, N: tl.constexpr, BLOCK: tl.constexpr,  # noqa: N803 - Triton constexpr names
    REVERSE: tl.constexpr, ACCUMULATE: tl.constexpr,  # noqa: N803
):  # fmt: skip
    """The backward of one direction, one pass against its time: with the adjoint states
    lam[t] = A (lam[t+1] + g[t+1]) and rho[t] = A (rho[t+1] + lam[t+1] + g[t+1]) (t+1: the
    previous step of this pass), du[t] = k0 g[t] + 2 Re sum_n w_n lam_n[t] (added into du when
    ACCUMULATE), S_n = sum_t u[t] lam_n[t], R_n = sum_t u[t] rho_n[t] and sum_t u[t] g[t];
    written as sums[b, 0..3, h, n] = Re S, Im S, Re R, Im R and k0g[b, h]."""
    b = tl.program_id(0)
    hs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ns = tl.arange(0, N)
    mh = hs < width
    p = hs[:, None] * N + ns[None, :]
    ar = tl.load(ar_ptr + p, mask=mh[:, None], other=0.0)
    ai = tl.load(ai_ptr + p, mask=mh[:, None], other=0.0)
    wr = tl.load(wr_ptr + p, mask=mh[:, None], other=0.0)
    wi = tl.load(wi_ptr + p, mask=mh[:, None], other=0.0)
    k0 = tl.load(k0_ptr + hs, mask=mh, other=0.0)
    lr = tl.zeros([BLOCK, N], dtype=tl.float32)
    li = tl.zeros([BLOCK, N], dtype=tl.float32)
    qr = tl.zeros([BLOCK, N], dtype=tl.float32)
    qi = tl.zeros([BLOCK, N], dtype=tl.float32)
    acc_sr = tl.zeros([BLOCK, N], dtype=tl.float32)
    acc_si = tl.zeros([BLOCK, N], dtype=tl.float32)
    acc_rr = tl.zeros([BLOCK, N], dtype=tl.float32)
    acc_ri = tl.zeros([BLOCK, N], dtype=tl.float32)
    acc_k0 = tl.zeros([BLOCK], dtype=tl.float32)
    prev = tl.zeros([BLOCK], dtype=tl.float32)
    base = b.to(tl.int64) * length * width
    first = length - 1 if REVERSE else 0
    g_next = tl.load(g_ptr + base + first * width + hs, mask=mh, other=0.0)
    u_next = tl.load(u_ptr + base + first * width + hs, mask=mh, other=0.0)
    for i in range(length):
        t = length - 1 - i if REVERSE else i
        g = g_next  # loaded a step ahead
        u = u_next
        ahead = tl.maximum(t - 1, 0) if REVERSE else tl.minimum(t + 1, length - 1)
        g_next = tl.load(g_ptr + base + ahead * width + hs, mask=mh, other=0.0)
        u_next = tl.load(u_ptr + base + ahead * width + hs, mask=mh, other=0.0)
        xr = lr + prev[:, None]  # lam[t+1] + g[t+1]; its imaginary part is li
        yr = qr + xr
        yi = qi + li
        qr, qi = ar * yr - ai * yi, ar * yi + ai * yr
        lr, li = ar * xr - ai * li, ar * li + ai * xr
        du = k0 * g + 2.0 * tl.sum(wr * lr - wi * li, axis=1)
        if ACCUMULATE:
            du += tl.load(du_ptr + base + t * width + hs, mask=mh, other=0.0)
        tl.store(du_ptr + base + t * width + hs, du, mask=mh)
        uc = u[:, None]
        acc_sr += uc * lr
        acc_si += uc * li
        acc_rr += uc * qr
        acc_ri += uc * qi
        acc_k0 += u * g
        prev = g
    o = b.to(tl.int64) * 4 * width * N + p
    tl.store(sums_ptr + o, acc_sr, mask=mh[:, None])
    tl.store(sums_ptr + o + width * N, acc_si, mask=mh[:, None])
    tl.store(sums_ptr + o + 2 * width * N, acc_rr, mask=mh[:, None])
    tl.store(sums_ptr + o + 3 * width * N, acc_ri, mask=mh[:, None])
    tl.store(k0g_ptr + b.to(tl.int64) * width + hs, acc_k0, mask=mh)


def _grid(u: torch.Tensor, block: int) -> tuple[int, int]:
    return (u.shape[0], triton.cdiv(u.shape[2], block))


def _direction(
    u: torch.Tensor, a: torch.Tensor, w: torch.Tensor, k0: torch.Tensor, reverse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor:  # fmt: skip
    """One direction over u [B, L, H] float32 (contiguous); a, w [H, N] complex64, k0 [H];
    added into `out` when given."""
    y = torch.empty_like(u) if out is None else out
    _scan[_grid(u, SCAN_LAUNCH["BLOCK"])](
        u, y, a.real.contiguous(), a.imag.contiguous(), w.real.contiguous(),
        w.imag.contiguous(), k0.contiguous(), u.shape[1], u.shape[2],
        N=a.shape[-1], REVERSE=reverse, ACCUMULATE=out is not None, **SCAN_LAUNCH,
    )  # fmt: skip
    return y


def bidirectional(
    u: torch.Tensor, a: torch.Tensor, w: torch.Tensor, k0: torch.Tensor, adjoint: bool
) -> torch.Tensor:
    """Both directions over u [B, L, H]: a, w [2, H, N] complex64 (forward, backward), k0
    [2, H]. `adjoint`: the transposed operator (each direction's weights run the other way)."""
    first = _direction(u, a[0], w[0], k0[0], adjoint, None)
    return _direction(u, a[1], w[1], k0[1], not adjoint, first)


def adjoint(
    u: torch.Tensor, g: torch.Tensor, a: torch.Tensor, w: torch.Tensor, k0: torch.Tensor,
    reverse: bool, du: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """The backward of one direction (a, w [H, N], k0 [H]) whose forward ran `not reverse`:
    (du [B, L, H] (added into `du` when given), S [H, N], R [H, N] complex64 summed over
    the channels, sum of u g [H])."""
    channels, length, width = u.shape
    out = torch.empty_like(u) if du is None else du
    sums = torch.empty(channels, 4, width, a.shape[-1], device=u.device, dtype=torch.float32)
    k0g = torch.empty(channels, width, device=u.device, dtype=torch.float32)
    _adjoint[_grid(u, ADJOINT_LAUNCH["BLOCK"])](
        u, g, out, a.real.contiguous(), a.imag.contiguous(), w.real.contiguous(),
        w.imag.contiguous(), k0.contiguous(), sums, k0g, length, width,
        N=a.shape[-1], REVERSE=reverse, ACCUMULATE=du is not None, **ADJOINT_LAUNCH,
    )  # fmt: skip
    total = sums.sum(0)
    return out, torch.complex(total[0], total[1]), torch.complex(total[2], total[3]), k0g.sum(0)


class ScanConv(torch.autograd.Function):
    """y [B, L, H] = the bidirectional convolution of u [B, L, H] with the kernel of k0 [2, H]
    (lag 0, counted in each direction), w and p (real views [2, H, N, 2] of complex weights and
    log-modes A = exp(p)); `kept` [2, H, N]: the modes that are not masked (A = 0 for the rest).
    Gradients to u, k0, w and p (see the module docstring)."""

    @staticmethod
    def forward(
        ctx: Any, u: torch.Tensor, k0: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
        kept: torch.Tensor,
    ) -> torch.Tensor:  # fmt: skip
        weight = torch.view_as_complex(w.detach().float().contiguous())
        power = torch.view_as_complex(p.detach().float().contiguous())
        a = torch.where(kept, power.exp(), torch.zeros_like(power))
        k0f = k0.detach().float()
        ctx.save_for_backward(u, a, weight, k0f)
        return bidirectional(u, a, weight, k0f, adjoint=False)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        u, a, weight, k0 = ctx.saved_tensors
        g = grad.contiguous()
        # forward direction: its backward runs against time; the backward direction's with it
        du, s0, r0, ug = adjoint(u, g, a[0], weight[0], k0[0], reverse=True, du=None)
        du, s1, r1, _ = adjoint(u, g, a[1], weight[1], k0[1], reverse=False, du=du)
        s, r = torch.stack([s0, s1]), torch.stack([r0, r1])
        wr = weight * r
        dw = torch.stack([2 * s.real, -2 * s.imag], dim=-1)
        dp = torch.stack([2 * wr.real, -2 * wr.imag], dim=-1)
        return du, ug.expand(2, -1), dw, dp, None
