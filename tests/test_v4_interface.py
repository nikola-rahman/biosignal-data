"""The benchmark interface (architecture-v2.md, section 11 step 2): a benchmark uses a model
only through `features()` and `layer_groups()` (and its config, train / eval); every run file
still validates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.batch import (  # noqa: E402
    DurationTokenizer,
    HierarchicalTokenizer,
    Inputs,
    SamplesTokenizer,
)
from biosignal_model.benchmarks import BenchmarkRun, apnea, ptbxl  # noqa: E402
from biosignal_model.model import Encoder, LayerGroups, ModelConfig  # noqa: E402
from biosignal_model.train import TrainConfig  # noqa: E402

D1 = DurationTokenizer(
    kind="duration", token_s=0.08, hop_s=0.04, stem_windows=4, stem_channels=8,
    kernel_frequencies=6, kernel_min_hz=1.0, kernel_max_hz=200.0,
)  # fmt: skip
D2 = HierarchicalTokenizer(
    kind="hierarchical", frame_s=0.04, frame_hop_s=0.02, frames_per_token=12, stem_windows=2,
    stem_channels=8, kernel_frequencies=6, kernel_min_hz=1.0, kernel_max_hz=200.0,
    local_width=16, local_layers=2, local_states=8, pool_heads=4, residual_pool="mean",
)  # fmt: skip
ALL = [
    SamplesTokenizer(kind="samples"), D1, D2, D2.model_copy(update={"residual_pool": "attention"})
]
IDS = ["samples", "duration", "hierarchical", "hierarchical-attention"]
RUNS = Path(__file__).resolve().parents[1] / "runs"


def _config(tokenizer: SamplesTokenizer | DurationTokenizer | HierarchicalTokenizer) -> ModelConfig:
    return ModelConfig(
        samples_per_token=25, max_channels=4, width=32, layers=3, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense", tokenizer=tokenizer,
        dropout=0.0, drop_path=0.0,
    )  # fmt: skip


@pytest.mark.parametrize("tokenizer", ALL, ids=IDS)
def test_the_layer_groups_hold_every_weight_a_task_uses_once(
    tokenizer: SamplesTokenizer | DurationTokenizer | HierarchicalTokenizer,
) -> None:
    model = Encoder(_config(tokenizer))
    g = model.layer_groups()
    groups = [g.input, g.channel, *g.layers, g.output]
    ids = [id(p) for group in groups for p in group]
    pretraining_heads = ("head.", "head_kernel.")
    used = {id(p) for n, p in model.named_parameters() if not n.startswith(pretraining_heads)}
    assert len(ids) == len(set(ids)) and set(ids) == used
    assert len(g.layers) == 3 and all(g.layers) and g.input and g.channel and g.output


class _OnlyTheInterface:
    """An Encoder seen through the benchmark interface alone: any other attribute fails."""

    def __init__(self, model: Encoder) -> None:
        self._model = model

    @property
    def config(self) -> ModelConfig:
        return self._model.config

    def features(self, x: Inputs) -> torch.Tensor:
        return self._model.features(x)

    def layer_groups(self) -> LayerGroups:
        return self._model.layer_groups()

    def train(self, mode: bool = True) -> object:
        return self._model.train(mode)

    def eval(self) -> object:
        return self._model.eval()

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the benchmark reads `{name}` of the model")


def _records(n: int, dataset: str) -> list[Example]:
    rng = np.random.default_rng(0)
    channels = tuple(Channel("100Hz", k, 100.0, 1.0, 0.0, None, f"L{k}", "ecg", "mV")
                     for k in range(2))  # fmt: skip
    return [
        Example(tuple(Signal(rng.normal(size=500).astype(np.float32), 100.0, 0.0)
                      for _ in range(2)),
                Origin(dataset, f"r{i}", f"p{i}", 0.0, channels))  # fmt: skip
        for i in range(n)
    ]


@pytest.mark.parametrize("tokenizer", ALL, ids=IDS)
def test_the_benchmarks_read_the_model_through_the_interface_only(
    tokenizer: SamplesTokenizer | DurationTokenizer | HierarchicalTokenizer,
) -> None:
    config = _config(tokenizer)
    model = _OnlyTheInterface(Encoder(config))
    head = torch.nn.Linear(config.width, 3)
    groups = ptbxl.parameter_groups(model, head, 1e-3, 0.1, 1 / 3, "lower")
    assert [len(g["params"]) > 0 for g in groups] == [True, True, True]
    examples = _records(6, ptbxl.DATASET)
    features = ptbxl.embed(
        model, examples, 1024, 2, torch.device("cpu"), lambda _: None, "e",
        ptbxl.lead_slots(examples, fixed=True),
    )  # fmt: skip
    assert features.shape == (6, 32) and np.isfinite(features).all()
    packer = ptbxl._packer(config, 1024, 2, None)
    p = ptbxl._predict(model, head, examples, 2.5, packer, 2, torch.device("cpu"), 0, "s", 1.0,
                       "mean", with_max=False)  # fmt: skip
    assert p.shape == (6, 3)
    minutes = apnea.Minutes(records=["a"], examples=_records(6, apnea.DATASET),
                            labels=np.zeros(6), record_of=np.zeros(6, dtype=np.int64),
                            in_minute=[(0.0, 5.0)] * 6)  # fmt: skip
    pooled = apnea.embed(model, minutes, 1024, 2, torch.device("cpu"), lambda _: None, "e")
    assert pooled.shape == (6, 32)


def test_every_run_file_is_valid() -> None:
    files = sorted(RUNS.glob("*.yaml"))
    kinds = {"train": 0, "ptbxl": 0, "apnea": 0}
    for f in files:
        if f.name.startswith("bench-ptbxl-"):
            ptbxl.PtbxlSettings.model_validate(BenchmarkRun.load(f).settings)
            kinds["ptbxl"] += 1
        elif f.name.startswith("bench-apnea-"):
            apnea.ApneaSettings.model_validate(BenchmarkRun.load(f).settings)
            kinds["apnea"] += 1
        else:
            TrainConfig.load(f).model_settings()
            kinds["train"] += 1
    assert all(kinds.values()), kinds
