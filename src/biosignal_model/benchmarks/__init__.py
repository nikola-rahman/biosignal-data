"""Benchmarks with the official split and protocol of each task (see docs/model.md).

A benchmark run is one YAML file: the checkpoint, the device, where to log, and the
benchmark's own `settings` (each benchmark module checks them against its settings class).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict

LOG_EVERY_STEPS = 20  # training steps between two records of the training loss


class Log(Protocol):
    """Records metrics while a benchmark runs (Weights & Biases and the console), with keys
    such as `<task>/<method>/train_loss`."""

    def __call__(self, metrics: dict[str, float]) -> None: ...


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    store: str
    checkpoint: str  # its key in the store, e.g. models/<run>/step-00012000.pt
    device: str
    settings: dict[str, Any]
    wandb_project: str
    wandb_entity: str
    wandb_mode: Literal["online", "offline", "disabled"]

    @classmethod
    def load(cls, path: Path) -> BenchmarkRun:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
