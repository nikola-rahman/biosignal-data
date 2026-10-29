"""Runner of the v3 proxy halving bench with WSD cooldown rungs (coordinator, 2026-10-29; the
configs set `fine_tuning_rungs`, the constant schedule and the epochs). On the pod: logs next
to --status (default /workspace/logs), configs in /workspace/proxy-configs.

Usage: proxy_bench_rungs.py <list> --worktree DIR --lrs 3e-5,1e-4,3e-4 --cut 10:0.5 [--cut ...]
       [--lr-rung E] [--budget-gb 40] [--prefix bench] [--status F]
<list>: lines "design config gpu memory_gb" (config relative to the worktree; memory_gb: the
run's peak GPU memory, for the budget). Each design runs once per learning rate (a copy of its
config with only fine_tuning_learning_rate changed, in /workspace/proxy-configs/). Runs start
in list order while the sum of memory_gb on their GPU stays within --budget-gb; each has its
log /workspace/logs/<prefix>-<design>-lr<lr>.log (an older log of that name is renamed, never
overwritten) and its own process group.
Decisions use only the fold-9 macro-AUC at the end of a rung's cooldown branch (rung_* lines):
- at --lr-rung (default: the first rung of the config), when all learning rates of a design
  have it (or have ended): the best one goes on, the others are stopped;
- at each --cut E:keep, when every design still going has its rung E (or has ended): designs
  ranked by it, the best round(keep x n) (at least 1) go on, the others are stopped.
State lines go to the status file; at the end a table per design: its learning rate, each
rung's fold-9 AUC and train-minus-fold-9 loss gap, the rung fold 9 chooses and its fold-10
macro-AUC, and how it ended."""

import argparse
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("list")
ap.add_argument("--worktree", required=True)
ap.add_argument("--lrs", required=True)
ap.add_argument("--cut", action="append", default=[])
ap.add_argument("--lr-rung", type=int, default=None)
ap.add_argument("--budget-gb", type=float, default=40.0)
ap.add_argument("--prefix", default="bench")
ap.add_argument("--status", default="/workspace/logs/bench.status")
ap.add_argument(
    "--command", default="/workspace/biosignal-data/.venv/bin/biosignal-model"
)  # a fake for the runner's test
ap.add_argument("--poll", type=float, default=30.0)
a = ap.parse_args()
logs = Path(a.status).parent
configs = logs.parent / "proxy-configs"
configs.mkdir(exist_ok=True)
status = Path(a.status)
cuts = sorted((int(e), float(k)) for e, k in (c.split(":") for c in a.cut))
lrs = [float(x) for x in a.lrs.split(",")]
RUNG = re.compile(r"\S+/rung_epoch=")
PAIR = re.compile(r"(\S+?)/(?:from_scratch|fine_tuning)/rung_(\S+?)=(\S+)")


def note(line: str) -> None:
    with status.open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


class Run:
    def __init__(self, design: str, config: str, gpu: str, memory: float, lr: float) -> None:
        self.design, self.gpu, self.memory, self.lr = design, gpu, memory, lr
        self.name = f"{design}-lr{lr:g}"
        self.log = logs / f"{a.prefix}-{self.name}.log"
        source = yaml.safe_load((Path(a.worktree) / config).read_text())
        if source["settings"]["fine_tuning_rungs"] is None:
            raise SystemExit(f"{config}: no fine_tuning_rungs")
        self.rung_epochs = source["settings"]["fine_tuning_rungs"]["epochs"]
        source["settings"]["fine_tuning_learning_rate"] = lr
        self.config = configs / f"{Path(config).stem}-lr{lr:g}.yaml"
        self.config.write_text(yaml.safe_dump(source, sort_keys=False))
        self.proc: subprocess.Popen | None = None
        self.ended: str | None = None

    def start(self) -> None:
        if self.log.exists():  # an earlier run's log is kept, never overwritten
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.log.stat().st_mtime))
            self.log.rename(self.log.with_name(f"{self.log.stem}.{stamp}.log"))
        env = dict(os.environ, PYTHONPATH=f"{a.worktree}/src", CUDA_VISIBLE_DEVICES=self.gpu)
        self.proc = subprocess.Popen(
            [a.command, "benchmark", "ptbxl", str(self.config)],
            cwd=a.worktree,
            env=env,
            stdout=self.log.open("w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        note(f"{self.name} running on GPU {self.gpu} ({self.config.name})")

    def stop(self, why: str) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait()
        self.ended = why
        note(f"{self.name} {why}")

    def rungs(self) -> dict[int, dict[str, float]]:
        """Each finished rung's record: epoch -> {validation_macro_auc, ...}."""
        if not self.log.exists():
            return {}
        out = {}
        for line in self.log.read_text(errors="replace").splitlines():
            if RUNG.search(line):
                record = {k: float(v) for _, k, v in PAIR.findall(line) if _ == "all"}
                if "epoch" in record and "validation_macro_auc" in record:
                    out[int(record["epoch"])] = record
        return out


def auc(r: Run, epoch: int) -> float:
    """The fold-9 macro-AUC at the end of the run's cooldown branch at `epoch`."""
    return r.rungs()[epoch]["validation_macro_auc"]


lines = [
    ln.split()
    for ln in Path(a.list).read_text().splitlines()
    if ln.strip() and not ln.startswith("#")
]
runs = [Run(d, c, g, float(m), lr) for d, c, g, m in lines for lr in lrs]
lr_rung = a.lr_rung if a.lr_rung is not None else runs[0].rung_epochs[0]
designs = list(dict.fromkeys(r.design for r in runs))
status.write_text("")
note(
    f"{len(runs)} runs: {len(designs)} designs x learning rates {lrs}; lr choice at rung "
    f"{lr_rung}; cuts {cuts}; budget {a.budget_gb} GB per GPU"
)
lr_done: set[str] = set()
cut_done: set[int] = set()
while True:
    for r in runs:
        if r.proc is not None and r.ended is None and r.proc.poll() is not None:
            r.ended = f"exit {r.proc.returncode}"
            note(f"{r.name} {r.ended}")
    # the learning rate of each design
    for d in designs:
        mine = [r for r in runs if r.design == d]
        if d in lr_done or not all(r.ended is not None or lr_rung in r.rungs() for r in mine):
            continue
        scored = [r for r in mine if lr_rung in r.rungs()]
        if scored:
            best = max(scored, key=lambda r: r.rungs()[lr_rung]["validation_macro_auc"])
            for r in mine:
                if r is not best and r.ended is None:
                    r.stop(
                        f"stopped: learning rate choice at rung {lr_rung} "
                        f"({auc(r, lr_rung):.4f} < {auc(best, lr_rung):.4f})"
                    )
            note(
                f"{d}: learning rate {best.lr:g} goes on (rung {lr_rung} fold-9 "
                + ", ".join(
                    f"{r.lr:g}: {r.rungs()[lr_rung]['validation_macro_auc']:.4f}" for r in scored
                )
                + ")"
            )
        lr_done.add(d)
    # the cuts between designs
    for epoch, keep in cuts:
        if epoch in cut_done or len(lr_done) < len(designs):
            continue
        going = [r for r in runs if r.ended is None or epoch in r.rungs()]
        going = [r for r in going if not (r.ended or "").startswith("stopped")]
        if not all(r.ended is not None or epoch in r.rungs() for r in going):
            continue
        ranked = sorted(
            (r for r in going if epoch in r.rungs()),
            key=lambda r: r.rungs()[epoch]["validation_macro_auc"],
            reverse=True,
        )
        kept = max(1, round(len(ranked) * keep))
        for r in ranked[kept:]:
            r.stop(f"stopped: cut at rung {epoch} (fold-9 {auc(r, epoch):.4f})")
        note(
            f"cut at rung {epoch}: kept "
            + ", ".join(
                f"{r.name} {r.rungs()[epoch]['validation_macro_auc']:.4f}" for r in ranked[:kept]
            )
        )
        cut_done.add(epoch)
    # start what fits
    for r in runs:
        if r.proc is None and r.ended is None:
            used = sum(
                o.memory for o in runs if o.gpu == r.gpu and o.proc is not None and o.ended is None
            )
            if used + r.memory <= a.budget_gb:
                r.start()
    if all(r.ended is not None for r in runs):
        break
    time.sleep(a.poll)
rows = []
for d in designs:
    for r in [r for r in runs if r.design == d]:
        got = r.rungs()
        if not got:
            rows.append(f"{r.name}: no rung | {r.ended}")
            continue
        best = max(got, key=lambda e: got[e]["validation_macro_auc"])
        per = "; ".join(
            f"rung {e}: fold-9 {v['validation_macro_auc']:.4f}, "
            f"gap {v['train_minus_validation_loss']:+.4f}"
            for e, v in sorted(got.items())
        )
        rows.append(
            f"{r.name}: {per} | chosen rung {best}: "
            f"fold-10 {got[best]['test_macro_auc']:.4f} | {r.ended}"
        )
note("FINAL\n" + "\n".join(rows))
note("bench DONE")
