"""The v4 batch format (biosignal_model.channel_batch): flat samples, the channel and token
tables, the global grid, the masks on the samples, packing, and the train config."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.channel_batch import (  # noqa: E402
    ChannelPacker,
    MaskSpec,
    concat,
    hide,
    normalize,
    token_cells,
)
from biosignal_model.chunking import FixedChunker, upsample  # noqa: E402

MASK = MaskSpec(
    ratio=0.75, cycles=0.0, time_spans=0.5, channel_groups=0.3, single_tokens=0.2,
    time_span_s=(0.2, 2.0), group_span_s=(0.2, 2.0),
    cycle_period_s=(0.3, 2.0), cycle_tooth=(0.1, 0.25), cycle_strength=0.3,
)  # fmt: skip


def _example(rates: list[float], seconds: float, seed: int, starts: list[float] | None = None
             ) -> Example:  # fmt: skip
    rng = np.random.default_rng(seed)
    signals, channels = [], []
    for i, rate in enumerate(rates):
        values = rng.normal(size=round(seconds * rate)).astype(np.float32)
        values[5] = np.nan  # an invalid sample
        start = 0.0 if starts is None else starts[i]
        signals.append(Signal(values, rate, start))
        channels.append(Channel("a", i, rate, 1.0, 0.0, None, f"c{i}", "ecg", "mV"))
    return Example(tuple(signals), Origin("d/x", f"r{seed}", "s", 0.0, tuple(channels)))


def _packer(mask: MaskSpec | None, rows: int = 1, token_s: float = 0.04,
            max_tokens: int = 4096, max_samples: int = 100_000) -> ChannelPacker:  # fmt: skip
    return ChannelPacker(
        datasets=["d/x"], token_s=token_s, max_tokens=max_tokens, max_samples=max_samples,
        max_channels=8, rows=rows, mask=mask, seed=0,
    )  # fmt: skip


def test_the_tables_describe_every_sample_and_token() -> None:
    # 100 Hz, 173.61 Hz, and 1 Hz (slower than one sample a cell: one token a sample)
    example = _example([100.0, 173.61, 1.0], 6.0, 0, starts=[0.0, 0.003, 0.4])
    batch, where = _packer(None).pack([example])
    x, c, t = batch.inputs, batch.inputs.channels, batch.inputs.tokens
    assert where == [(0, 0)] and x.rows == 1
    assert c.offset.tolist() == [0, 600, 600 + 1042] and c.length.tolist() == [600, 1042, 6]
    assert t.t0.dtype == t.centre.dtype == torch.float64
    torch.testing.assert_close(t.t1 - t.t0, torch.full_like(t.t0, 0.04), rtol=0, atol=1e-12)
    cells = t.t0 / 0.04
    torch.testing.assert_close(cells, cells.round(), rtol=0, atol=1e-9)  # one grid
    assert t.position.tolist() == list(range(t.channel.numel()))
    counts = torch.bincount(t.channel).tolist()
    assert counts == [150, 150, 6]  # 6 s of 0.04 s cells; the 1 Hz channel: its 6 samples
    for k in range(3):
        n = int(c.length[k])
        times = c.start_s[k] + torch.arange(n, dtype=torch.float64) / c.rate_hz[k]
        token = t.sample_token[c.offset[k] : c.offset[k] + n]
        assert (t.channel[token] == k).all()
        assert (t.t0[token] <= times + 1e-9).all() and (times < t.t1[token] - 1e-9).all()
    assert not x.observed[5] and batch.targets.values[5] == 0  # NaN: not observed, value 0
    assert batch.samples == 600 + 1042 + 6 - 3
    assert FixedChunker(0.04).boundaries(x.values[:, None], x.observed, x) is t
    with pytest.raises(ValueError, match="token_s"):
        FixedChunker(0.05).boundaries(x.values[:, None], x.observed, x)
    up = upsample(torch.arange(t.channel.numel()).float()[:, None], t)
    assert torch.equal(up[:, 0].long(), t.sample_token)


def test_a_sample_on_a_cell_boundary_is_in_the_cell_it_starts() -> None:
    cells = token_cells(0.0, 100.0, 13, 0.04)  # 0.12 / 0.04 = 2.9999999999999996
    assert cells.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]


def test_the_mask_hides_its_ratio_on_the_samples_and_keeps_the_values_for_the_loss() -> None:
    packer = _packer(MASK, rows=4, max_tokens=400)  # 375 tokens an example: one a row
    examples = [_example([100.0, 250.0, 500.0], 5.0, s) for s in range(4)]
    batch, _ = packer.pack(examples)
    x, t = batch.inputs, batch.inputs.tokens
    c = x.channels
    for r in range(4):  # one example a row: the ratio of each example, to whole cells
        mine = t.row == r
        n = int(mine.sum())
        goal = min(n - 1, round(0.75 * n))
        assert goal <= int(t.hidden[mine].sum()) <= goal + 2 * 2  # two kinds by cells of 3
    assert torch.equal(x.hidden, t.hidden[t.sample_token])  # the samples of hidden tokens
    assert (x.values[x.hidden] == 0).all() and not x.observed[x.hidden].any()
    valid = batch.targets.observed
    for k in range(c.row.numel()):  # the student's transform uses its visible samples only
        o, n = int(c.offset[k]), int(c.length[k])
        raw = torch.where(valid[o : o + n], batch.targets.values[o : o + n], torch.nan)
        z, _, _ = normalize(raw.numpy(), ~x.hidden[o : o + n].numpy())
        seen = ~x.hidden[o : o + n]
        assert torch.equal(x.values[o : o + n][seen], torch.from_numpy(z)[seen])
    assert (batch.targets.values[x.hidden & valid] != 0).all()  # kept apart for the loss
    assert (batch.targets.normalized[x.hidden & valid] != 0).all()  # and for the teacher
    teacher = batch.teacher_inputs()
    assert not teacher.hidden.any() and not teacher.tokens.hidden.any()
    assert torch.equal(teacher.values, batch.targets.normalized)
    assert torch.equal(teacher.observed, valid)
    assert c.descriptors == {} and c.present == {}  # the placeholder: no fields yet


def _hidden_cells(mask: MaskSpec, channels: int, seed: int) -> list[np.ndarray]:
    cells = [np.arange(250) for _ in range(channels)]  # 10 s of 0.04 s cells, one rate
    return hide(cells, 0.04, mask, np.random.default_rng(seed), groups=None)


def test_time_spans_hide_all_channels_together_and_groups_hide_some() -> None:
    spans = MASK.model_copy(update={"time_spans": 1.0, "channel_groups": 0.0,
                                    "single_tokens": 0.0})  # fmt: skip
    groups = MASK.model_copy(update={"time_spans": 0.0, "channel_groups": 1.0,
                                     "single_tokens": 0.0})  # fmt: skip
    for seed in range(5):
        hidden = np.stack(_hidden_cells(spans, 4, seed))  # [channels, cells]
        assert 750 <= hidden.sum() <= 753  # whole cells of 4 tokens
        assert not (hidden.any(0) & ~hidden.all(0)).any()  # no time partly hidden
        runs = np.flatnonzero(np.diff(hidden[0].astype(int)) == 1).size
        assert runs >= 1
        grouped = np.stack(_hidden_cells(groups, 4, seed))
        assert 750 <= grouped.sum() <= 753
        assert (grouped.any(0) & ~grouped.all(0)).sum() > 10  # subsets of the channels


def test_labelled_groups_are_hidden_together() -> None:
    groups = MASK.model_copy(update={"time_spans": 0.0, "channel_groups": 1.0,
                                     "single_tokens": 0.0, "ratio": 0.3})  # fmt: skip
    cells = [np.arange(250) for _ in range(4)]
    hidden = np.stack(hide(cells, 0.04, groups, np.random.default_rng(0),
                           groups=["a", "a", "b", "b"]))  # fmt: skip
    assert np.array_equal(hidden[0], hidden[1]) and np.array_equal(hidden[2], hidden[3])


def test_rows_keep_both_limits_and_concat_joins_rows() -> None:
    examples = [_example([100.0, 250.0], 2.0, s) for s in range(6)]  # 100 tokens, 700 samples
    packer = _packer(None, rows=3, max_tokens=250, max_samples=1500)
    batches = list(packer.batches(examples))
    assert [b.inputs.rows for b in batches] == [3]
    c = batches[0].inputs.channels
    assert c.row.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]  # two examples a row
    assert c.document.tolist() == [0, 0, 1, 1] * 3
    single = [next(iter(_packer(None, rows=1, max_tokens=250, max_samples=1500).batches([e])))
              for e in examples[:2]]  # fmt: skip
    joined = concat(single)
    together, _ = _packer(None, rows=2, max_tokens=100, max_samples=1500).pack(examples[:2])
    assert joined.inputs.rows == 2
    for name in ("row", "offset", "length", "document"):
        assert torch.equal(getattr(joined.inputs.channels, name),
                           getattr(together.inputs.channels, name))  # fmt: skip
    for name in ("channel", "row", "position", "sample_token", "t0"):
        assert torch.equal(getattr(joined.inputs.tokens, name),
                           getattr(together.inputs.tokens, name))  # fmt: skip
    assert torch.equal(joined.inputs.values, together.inputs.values)
    with pytest.raises(ValueError, match="longer than"):
        _packer(None, max_tokens=50).pack(examples[:1])


def test_the_train_config_takes_the_channel_format() -> None:
    from biosignal_data.training.loader import SampleAndTokenLimit  # noqa: PLC0415
    from biosignal_model.train import TrainConfig  # noqa: PLC0415

    config: dict[str, Any] = {
        "data": {"store": "/s", "index": "i", "train_split": "train",
                 "validation_split": "validation", "signal_types": None, "datasets": None,
                 "cache_directory": "/c", "cache_bytes": 1, "block_s": 1800.0,
                 "shuffle_buffer": 1, "reads_in_flight": 1, "workers": 0,
                 "pieces_per_epoch": 1, "spans": "per_group", "subject_hours_per_epoch": 4.0,
                 "flat_runs": None,
                 "mixture": {"rule": "unimax", "unit": "global_tokens", "budget_tokens": 1e9,
                             "max_epochs": 4, "clusters": {"all": ["d/x"]}}},
        "tokens": {"format": "channels", "token_s": 0.04, "max_tokens": 8192,
                   "max_samples": 400_000, "max_channels": 64, "rows": 8,
                   "mask": MASK.model_dump()},
        "model": {"width": 32, "layers": 2, "heads": 4, "feed_forward": 64,
                  "min_wavelength_s": 0.01, "max_wavelength_s": 1000.0, "attention": "dense",
                  "tokenizer": {"kind": "samples"}, "dropout": 0.0, "drop_path": 0.0},
        "optim": {"learning_rate": 1e-3, "min_learning_rate": 1e-4, "warmup_steps": 1,
                  "steps": 2, "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95,
                  "grad_clip": 1.0, "schedule": "cosine", "decay_steps": None,
                  "decay_shape": None},
        "run": {"name": "t", "seed": 0, "device": "cpu", "precision": "fp32", "log_every": 1,
                "validate_every": 1, "validation_batches": 1, "checkpoint_every": 1,
                "checkpoint_prefix": "m", "wandb_project": "p", "wandb_entity": "e",
                "wandb_mode": "disabled", "max_bad_steps": 1, "bad_batch_directory": "/b",
                "resume": "never"},
    }  # fmt: skip
    parsed = TrainConfig.model_validate(config)
    assert parsed.span_limit() == SampleAndTokenLimit(400_000, 8192, 0.04)
    with pytest.raises(ValueError, match="v0-v3"):  # a v4 run's model is its model section
        parsed.model_settings()
    bad = config | {"tokens": config["tokens"] | {"mask": MASK.model_dump() | {"extra": 1}}}
    with pytest.raises(ValueError, match="extra"):
        TrainConfig.model_validate(bad)
    # a checkpoint's config from before v4 reads with the old values
    old = {k: dict(v) for k, v in config.items()}
    old["tokens"] = {"max_tokens": 128, "samples_per_token": 16, "max_channels": 4, "rows": 2,
                     "mask_ratio": 0.5}  # fmt: skip
    for name in ("spans", "subject_hours_per_epoch"):
        del old["data"][name]
    old["data"]["mixture"] = {k: v for k, v in config["data"]["mixture"].items() if k != "unit"}
    read = TrainConfig.from_checkpoint(old)
    assert read.tokens.format == "tokens" and read.data.spans == "shared"
    assert read.data.subject_hours_per_epoch is None
    assert read.data.mixture is not None and read.data.mixture.unit == "sample_tokens"
