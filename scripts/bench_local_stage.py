"""Throughput and memory of the v4 local stage per signal type (docs/architecture.md section
10, item 4), on synthetic examples of real shapes. A training step: forward, a stand-in loss
(the mean square of the features), backward; bf16 autocast, activation checkpointing per
layer. While it runs it prints, every second, the rate of the last second; then one summary
line per case (median rate over the timed seconds, peak memory).

Usage (on the pod, GPU 1 only):
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_local_stage.py --seconds 20 --widths 128 256 \
        --batch-scale 1.0
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from biosignal_data.training.loader import Channel, Example, Origin, Signal
from biosignal_model.channel_batch import ChannelPacker
from biosignal_model.local_stage import S4DLocalSpec, S4DLocalStage


@dataclass(frozen=True)
class Case:
    name: str
    channels: list[tuple[str, int, float]]  # (type, count, rate) of one example
    seconds: float
    examples: int  # a batch


CASES = [
    Case("ECG 12 x 500 Hz x 10 s", [("ecg", 12, 500.0)], 10.0, 32),
    Case("EEG 19 x 256 Hz x 30 s", [("eeg", 19, 256.0)], 30.0, 16),
    Case("EEG 64 x 256 Hz x 30 s", [("eeg", 64, 256.0)], 30.0, 4),
    Case("HD-EMG 64 x 2048 Hz x 2 s", [("emg", 64, 2048.0)], 2.0, 8),
    Case(
        "PSG 30 s (6 EEG + 2 EOG 256 Hz, EMG + ECG 512 Hz, 3 resp 32 Hz, SpO2 + pos 1 Hz)",
        [
            ("eeg", 6, 256.0),
            ("eog", 2, 256.0),
            ("emg", 1, 512.0),
            ("ecg", 1, 512.0),
            ("resp", 3, 32.0),
            ("spo2", 1, 1.0),
            ("position", 1, 1.0),
        ],
        30.0,
        16,
    ),
]


def examples(case: Case, seed: int) -> list[Example]:
    rng = np.random.default_rng(seed)
    out = []
    for e in range(case.examples):
        signals: list[Signal] = []
        channels: list[Channel] = []
        for kind, count, rate in case.channels:
            for _ in range(count):
                values = rng.normal(size=round(case.seconds * rate)).astype(np.float32)
                signals.append(Signal(values, rate, 0.0))
                channels.append(Channel("a", len(channels), rate, 1.0, 0.0, None,
                                        f"c{len(channels)}", kind, "mV"))  # fmt: skip
        out.append(Example(tuple(signals), Origin("d/x", f"r{e}", f"s{e}", 0.0,
                                                  tuple(channels))))  # fmt: skip
    return out


def spec(width: int) -> S4DLocalSpec:
    return S4DLocalSpec(
        kind="s4d", width=width, layers=4, states=32, min_time_s=0.001, max_time_s=10.0,
        kernel_time_constants=5.0, discretization="zoh_centred", masked_modes="static_gain",
        mixing_layers=[1, 3], mixing_heads=4,
        dropout=0.0, checkpoint=True,
    )  # fmt: skip


def run(case: Case, width: int, seconds: float, device: torch.device) -> str:
    packer = ChannelPacker(
        datasets=["d/x"], token_s=0.04, max_tokens=10**7, max_samples=10**8,
        max_channels=128, rows=1, mask=None, seed=0,
    )  # fmt: skip
    batch, _ = packer.pack(examples(case, 0))
    x = batch.inputs.to(device)
    samples = x.values.numel()
    torch.manual_seed(0)
    stage = S4DLocalStage(spec(width)).to(device)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats(device)

    def step() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = stage(x)
        loss = out.float().square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(2):  # warm-up: allocator, cuFFT plans
        step()
    torch.cuda.synchronize(device)
    print(f"--- {case.name}, width {width}: {samples:,} samples a batch "
          f"({case.examples} examples)", flush=True)  # fmt: skip
    rates: list[float] = []
    start = last = time.perf_counter()
    steps_since = 0
    while time.perf_counter() - start < seconds:
        step()
        torch.cuda.synchronize(device)
        steps_since += 1
        now = time.perf_counter()
        if now - last >= 1.0:
            rate = steps_since * samples / (now - last)
            rates.append(rate)
            memory = torch.cuda.max_memory_allocated(device) / 2**30
            print(f"  t={now - start:5.1f} s  {rate / 1e6:7.2f} M samples/s  "
                  f"{rate / samples * case.examples:7.1f} examples/s  "
                  f"peak {memory:5.2f} GiB", flush=True)  # fmt: skip
            last, steps_since = now, 0
    memory = torch.cuda.max_memory_allocated(device) / 2**30
    median = statistics.median(rates) if rates else float("nan")
    return (f"{case.name} | width {width} | {samples:,} samples | "
            f"{median / 1e6:.2f} M samples/s | {median / samples * case.examples:.1f} "
            f"examples/s | {median / samples * case.examples * case.seconds / 3600:.2f} "
            f"h of recording/s | peak {memory:.2f} GiB")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument("--batch-scale", type=float, required=True)  # examples a batch x this
    args = parser.parse_args()
    seconds = args.seconds
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    summary = []
    for width in args.widths:
        for full in CASES:
            case = replace(full, examples=max(1, round(full.examples * args.batch_scale)))
            try:
                summary.append(run(case, width, seconds, device))
            except torch.OutOfMemoryError:
                summary.append(f"{case.name} | width {width} | out of memory")
                torch.cuda.empty_cache()
            print(summary[-1], flush=True)
    print("\nSUMMARY (median of the per-second rates)")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
