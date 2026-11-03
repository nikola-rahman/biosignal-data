"""Architecture checks of the v4 local stage at random init (docs/architecture.md section 10,
items 3 and the discretization): no training.

1. Rate invariance: one band-limited signal (3 channels on one time base, sums of sinusoids
   below a set frequency, defined in continuous time) sampled at 100 / 128 / 173 / 250 /
   500 Hz; the stage's features at the times common to a rate and 500 Hz, relative error
   ||F_r - F_500|| / ||F_500||, for each discretization.
2. Frequency response of one layer's SSM (forward kernel, every feature) at 100 and 500 Hz,
   against the response of the continuous system: magnitude ratio and phase error at set
   frequencies (median over the features), for ZOH, centred ZOH and bilinear.

Usage: python scripts/v4_local_checks.py --device cuda   (on the pod: CUDA_VISIBLE_DEVICES=1)
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from biosignal_data.training.loader import Channel, Example, Origin, Signal
from biosignal_model.channel_batch import ChannelPacker
from biosignal_model.local_stage import S4D, S4DLocalSpec, S4DLocalStage

RATES = [100, 128, 173, 250, 500]
KINDS = ["zoh", "zoh_centred", "bilinear"]


def spec(kind: str, min_time_s: float, masked: str = "drop") -> S4DLocalSpec:
    return S4DLocalSpec(
        kind="s4d", width=128, layers=4, states=32, min_time_s=min_time_s, max_time_s=10.0,
        kernel_time_constants=5.0, discretization=kind, masked_modes=masked,  # type: ignore[arg-type]
        mixing_layers=[1, 3], mixing_heads=4, dropout=0.0, checkpoint=False,
        compile=False, convolution="fft",
    )  # fmt: skip


def signal(seconds: float, band_hz: float, seed: int) -> list[tuple[np.ndarray, np.ndarray,
                                                                     np.ndarray]]:  # fmt: skip
    """3 channels: (frequencies, amplitudes, phases) of 40 sinusoids, log-uniform in
    [0.1, band_hz], amplitude 1/sqrt(f), unit RMS."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        f = np.exp(rng.uniform(math.log(0.1), math.log(band_hz), 40))
        a = 1 / np.sqrt(f)
        a = a / np.sqrt((a**2).sum() / 2)
        out.append((f, a, rng.uniform(0, 2 * math.pi, 40)))
    return out


def example(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], rate: int,
            seconds: float) -> Example:  # fmt: skip
    t = np.arange(round(seconds * rate)) / rate
    signals, channels = [], []
    for k, (f, a, p) in enumerate(parts):
        values = (a[:, None] * np.sin(2 * math.pi * f[:, None] * t + p[:, None])).sum(0)
        signals.append(Signal(values.astype(np.float32), float(rate), 0.0))
        channels.append(Channel("a", k, float(rate), 1.0, 0.0, None, f"c{k}", "ecg", "mV"))
    return Example(tuple(signals), Origin("d/x", "r", "s", 0.0, tuple(channels)))


def features(stage: S4DLocalStage, e: Example, device: torch.device) -> np.ndarray:
    packer = ChannelPacker(
        datasets=["d/x"], token_s=0.04, max_tokens=10**6, max_samples=10**7, max_channels=8,
        rows=1, mask=None, seed=0,
    )  # fmt: skip
    batch, _ = packer.pack([e])
    with torch.no_grad():
        out = stage(batch.inputs.to(device)).double().cpu().numpy()
    n = e.signals[0].values.size
    shaped: np.ndarray = out.reshape(3, n, -1)  # [channel, time, width]
    return shaped


def rate_invariance(device: torch.device) -> None:
    seconds = 10.0
    print("\n1. Rate invariance: relative error of the features vs 500 Hz at common times")
    print("   (all common times | interior: 1 s from each end); width 128, 4 layers, 32 modes,")
    print("   mixing in layers 1 and 3, fp32")
    for band in (10.0, 40.0):
        parts = signal(seconds, band, 0)
        for min_time, masked in ((0.001, "drop"), (0.001, "static_gain"), (0.35, "drop")):
            label = ("init 1 ms-10 s" if min_time == 0.001
                     else "init 0.35-10 s (no mode masked at 100 Hz)")  # fmt: skip
            print(f"\n   signal band 0.1-{band:g} Hz, mode time scales {label}, "
                  f"masked modes: {masked}")  # fmt: skip
            print("   " + "discretization".ljust(14) + "".join(f"{r:>18d} Hz" for r in RATES[:-1]))
            for kind in KINDS:
                torch.manual_seed(0)
                stage = S4DLocalStage(spec(kind, min_time, masked)).to(device).eval()
                reference = features(stage, example(parts, 500, seconds), device)
                cells = []
                for r in RATES[:-1]:
                    f = features(stage, example(parts, r, seconds), device)
                    step = r // math.gcd(r, 500)
                    i = np.arange(0, f.shape[1], step)
                    j = i * 500 // r
                    keep = j < reference.shape[1]
                    i, j = i[keep], j[keep]
                    inner = (i / r >= 1.0) & (i / r <= seconds - 1.0)

                    def rel(m: np.ndarray, i: np.ndarray = i, j: np.ndarray = j,
                            f: np.ndarray = f, ref: np.ndarray = reference) -> float:  # fmt: skip
                        d = f[:, i[m]] - ref[:, j[m]]
                        return float(np.linalg.norm(d) / np.linalg.norm(ref[:, j[m]]))

                    all_m = np.ones(i.size, dtype=bool)
                    cells.append(f"{rel(all_m):.4f} | {rel(inner):.4f} ({i.size:>4d})")
                print("   " + kind.ljust(14) + "".join(f"{c:>21s}" for c in cells))


def frequency_response(device: torch.device) -> None:
    print("\n2. Frequency response of one layer's SSM (forward kernel), vs the continuous system")
    print("   median over 128 features of |H_d| / |H_c| and of the phase error (degrees);")
    print("   time scales 0.35-10 s (no mode masked at 100 Hz: the same modes at both rates)")
    frequencies = np.array([1.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    torch.manual_seed(0)
    base = S4D(spec("zoh", 0.35)).double()
    with torch.no_grad():
        a = torch.complex(-base.log_neg_real.exp(), base.imag)[0].numpy()  # [H, N]
        delta = base.log_time.exp()[0].numpy()[:, None]
        c = torch.view_as_complex(base.c)[0].numpy()
    lam, w = a / delta, c / delta  # lambda (1/s) and C B
    omega = 2 * math.pi * frequencies
    continuous = (w[..., None] / (1j * omega - lam[..., None])
                  + w.conj()[..., None] / (1j * omega - lam.conj()[..., None])).sum(1)  # fmt: skip
    print("   " + "kind, rate".ljust(22) + "".join(f"{f:>16g} Hz" for f in frequencies))
    for kind in KINDS:
        ssm = S4D(spec(kind, 0.35)).double()
        ssm.load_state_dict(base.state_dict())
        for rate in (100.0, 500.0):
            with torch.no_grad():
                k = ssm.kernel(rate, ssm.length_at(rate, 10**9))[0].numpy()  # [H, K]
            lags = np.arange(k.shape[1]) / rate
            # the DTFT at the set frequencies, from the kernel itself
            response = k @ np.exp(-1j * omega[None, :] * lags[:, None])
            ratio = np.median(np.abs(response) / np.abs(continuous), axis=0)
            phase = np.median(np.degrees(np.angle(response / continuous)), axis=0)
            cells = [f"{m:.4f} {p:+6.2f}" for m, p in zip(ratio, phase, strict=True)]
            print("   " + f"{kind}, {rate:g} Hz".ljust(22) + "".join(f"{x:>19s}" for x in cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    device = torch.device(parser.parse_args().device)
    start = time.perf_counter()
    rate_invariance(device)
    frequency_response(device)
    print(f"\ndone in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
