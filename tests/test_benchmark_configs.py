from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

pytest.importorskip("torch")

from biosignal_model.benchmarks import BenchmarkRun, apnea, dalia, isruc, ptbxl

RUNS = Path(__file__).resolve().parents[1] / "runs"
SETTINGS: dict[str, type[BaseModel]] = {
    "apnea": apnea.ApneaSettings,
    "ptbxl": ptbxl.PtbxlSettings,
    "isruc": isruc.IsrucSettings,
    "dalia": dalia.DaliaSettings,
}


@pytest.mark.parametrize("path", sorted(RUNS.glob("bench-*.yaml")), ids=lambda p: p.name)
def test_every_benchmark_run_config_is_valid(path: Path) -> None:
    run = BenchmarkRun.load(path)
    settings = SETTINGS[path.name.split("-")[1]].model_validate(run.settings)
    if isinstance(settings, dalia.DaliaSettings):  # each subject a test subject once
        assert sorted(n for f in settings.folds for n in f.test) == sorted(dalia.SUBJECTS)
