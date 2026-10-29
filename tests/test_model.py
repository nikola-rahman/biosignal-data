from __future__ import annotations

import dataclasses
import json
import math
import os
import socket
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.storage.stores import LocalStore, StoreFactory  # noqa: E402
from biosignal_data.training import Selection  # noqa: E402
from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.batch import (  # noqa: E402
    SIGNAL_TYPES,
    Batch,
    Inputs,
    Packer,
    SamplesTokenizer,
    concat,
)
from biosignal_model.model import Encoder, ModelConfig, masked_loss  # noqa: E402
from biosignal_model.train import (  # noqa: E402
    ADDED_LATER,
    OptimSection,
    TrainConfig,
    _forward_backward,
    _stream,
    _validation_batches,
    train,
)
from tests.test_training_loader import _index  # noqa: E402


def _example(lengths: list[int], rates: list[float], dataset: str = "d/x") -> Example:
    rng = np.random.default_rng(len(lengths))
    signals, channels = [], []
    for i, (n, rate) in enumerate(zip(lengths, rates, strict=True)):
        values = (rng.normal(size=n) * 1e-5 + 3e-6).astype(np.float32)  # EEG in volts
        values[1] = np.nan
        signals.append(Signal(values, rate, 0.25 / rate))
        channels.append(Channel("a", i, rate, 1.0, 0.0, None, f"c{i}", "eeg", "V"))
    return Example(tuple(signals), Origin(dataset, "r", "d/x:s", 0.0, tuple(channels)))


def _packer(**kw: object) -> Packer:
    fields: dict[str, object] = {
        "datasets": ["d/x", "d/y"], "max_tokens": 64, "samples_per_token": 8,
        "max_channels": 4, "rows": 2, "mask_ratio": 0.5, "seed": 3, "slots": None,
        "tokenizer": SamplesTokenizer(kind="samples"),
    }  # fmt: skip
    return Packer(**(fields | kw))  # type: ignore[arg-type]


def test_tokens_keep_every_value_and_its_time() -> None:
    example = _example([20, 9], [100.0, 10.0])
    tokens = _packer().tokens(example)
    assert len(tokens.slot) == math.ceil(20 / 8) + math.ceil(9 / 8)  # 3 + 2
    first = slice(0, 3)
    mean, log_scale = np.sinh(tokens.channel_scale[0, 0]), tokens.channel_scale[0, 1]
    restored = np.sinh(tokens.values[first]) * np.exp(log_scale) + mean
    original = np.full(24, np.nan, dtype=np.float32)
    original[:20] = example.signals[0].values
    original = original.reshape(3, 8)
    observed = tokens.observed[first]
    # The encoding is float32, like the model's input: a value next to the channel mean
    # (cancellation in x - mean) comes back to float32 precision of the channel's scale.
    scale = np.exp(log_scale)
    np.testing.assert_allclose(restored[observed], original[observed], rtol=1e-5, atol=1e-6 * scale)
    assert not observed.reshape(-1)[1] and not observed.reshape(-1)[20:].any()
    # token centres: start + (k + 1/2) * 8 / rate
    np.testing.assert_allclose(tokens.time_s[first], 0.25 / 100 + (np.arange(3) + 0.5) * 0.08)
    assert len(set(tokens.slot[:3])) == 1 and tokens.slot[0] != tokens.slot[3]
    for channel in (slice(0, 3), slice(3, 5)):
        assert tokens.hidden[channel].any() and not tokens.hidden[channel].all()
    assert set(tokens.signal_type) == {SIGNAL_TYPES.index("eeg")}


def test_a_change_in_a_flat_channel_stays_small_enough_to_learn() -> None:
    # capslpdb SpO2: flat for minutes, then a desaturation in a hidden token.
    values = np.full(64, 97.0, dtype=np.float32)
    values[5] = 97.000001  # one tiny step in the visible part: std near 0
    values[40:48] = 85.0
    channel = Channel("a", 0, 1.0, 1.0, 0.0, None, "SpO2", "spo2", "%")
    example = Example((Signal(values, 1.0, 0.0),), Origin("d/x", "r", "d/x:s", 0.0, (channel,)))
    packer = _packer(samples_per_token=8, mask_ratio=0.0)
    packer.mask_ratio = 0.5
    for _ in range(20):  # whichever tokens are hidden
        tokens = packer.tokens(example)
        assert np.abs(tokens.values).max() < 20  # asinh(1e8) is 19.1


def test_fixed_slots_follow_the_channel_name_in_any_order() -> None:
    example = _example([16, 24, 8], [100.0, 100.0, 100.0])  # channels c0, c1, c2
    turned = Example(example.signals[::-1], dataclasses.replace(
        example.origin, channels=example.origin.channels[::-1]))  # fmt: skip
    packer = _packer(slots={"c0": 3, "c1": 0, "c2": 1}, mask_ratio=0.0)
    for e in (example, turned):
        tokens = packer.tokens(e)
        names = [c.name for c, s in zip(e.origin.channels, e.signals, strict=True)
                 for _ in range(-(-s.values.size // 8))]  # fmt: skip
        assert tokens.slot.tolist() == [{"c0": 3, "c1": 0, "c2": 1}[n] for n in names]
    with pytest.raises(ValueError, match="slot"):
        _packer(slots={"c0": 4})  # max_channels is 4


def test_examples_are_packed_whole_into_rows() -> None:
    examples = [_example([200], [100.0]), _example([200, 100], [100.0, 50.0], "d/y"),
                _example([300], [100.0]), _example([80], [10.0])]  # fmt: skip
    batches = list(_packer().batches(examples))
    assert sum(b.examples for b in batches) == 4
    first = batches[0]
    assert first.inputs.values.shape == (2, 64, 8)
    docs = first.inputs.document[0].tolist()
    assert docs[:25] == [0] * 25 and docs[25 : 25 + 38] == [1] * 38  # 200/8, then 25 + 13
    assert (first.analysis.dataset[first.inputs.document < 0] == -1).all()
    with pytest.raises(ValueError, match="longer than max_tokens"):
        list(_packer().batches([_example([600], [100.0])]))


def _config(**model: object) -> ModelConfig:
    fields: dict[str, object] = {
        "samples_per_token": 8, "max_channels": 4, "width": 32, "layers": 2, "heads": 4,
        "feed_forward": 64, "min_wavelength_s": 0.01, "max_wavelength_s": 1000.0,
        "attention": "dense", "tokenizer": SamplesTokenizer(kind="samples"),
        "dropout": 0.0, "drop_path": 0.0,
    }  # fmt: skip
    return ModelConfig(**(fields | model))  # type: ignore[arg-type]


def test_an_example_never_sees_the_other_examples_of_its_row() -> None:
    torch.manual_seed(0)
    model = Encoder(_config()).eval()
    batch = next(_packer(rows=1).batches([_example([200], [100.0]), _example([160], [100.0])]))
    x = batch.inputs
    before = model(x)
    second = x.document == 1
    x.values[second] = torch.randn_like(x.values[second])
    after = model(x)
    first = x.document == 0
    torch.testing.assert_close(after[first], before[first])
    assert not torch.allclose(after[second], before[second])
    loss, per_token = masked_loss(after, x)
    assert loss.item() > 0 and per_token.shape == (1, 64, 2)


def test_the_block_mask_made_row_by_row_is_the_mask_of_all_rows() -> None:
    from torch.nn.attention.flex_attention import create_block_mask  # noqa: PLC0415

    from biosignal_model.model import _block_mask  # noqa: PLC0415

    document = torch.full((3, 384), -1)
    document[0, :200], document[0, 200:330] = 0, 1
    document[1, :384] = 0
    document[2, :50], document[2, 50:300], document[2, 300:301] = 0, 1, 2

    def same(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return document[b, q] == document[b, kv]

    joined, whole = _block_mask(document), create_block_mask(same, 3, None, 384, 384, "cpu")
    for name in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                 "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"):  # fmt: skip
        assert torch.equal(getattr(joined, name), getattr(whole, name)), name
    assert joined.BLOCK_SIZE == whole.BLOCK_SIZE and joined.seq_lengths == whole.seq_lengths
    assert torch.equal(joined.to_dense(), whole.to_dense())


def test_a_short_run_learns_and_logs_without_artifacts(store: LocalStore, tmp_path: Path) -> None:
    _index(store, {f"r{i}": 40.0 for i in range(6)} | {"held-out": 40.0})
    store.put_bytes("index/i1/_manifest.json", b'{"version": "i1"}')
    config = TrainConfig.model_validate(
        {
            "data": {"store": str(store.root), "index": "i1", "train_split": "train",
                     "validation_split": "test", "signal_types": None, "datasets": None,
                     "cache_directory": str(tmp_path / "cache"), "cache_bytes": 10**9,
                     "block_s": 20.0, "shuffle_buffer": 16, "reads_in_flight": 2, "workers": 0,
                     "mixture": None, "pieces_per_epoch": 1},
            "tokens": {"max_tokens": 128, "samples_per_token": 16, "max_channels": 4,
                       "rows": 2, "mask_ratio": 0.4},
            "model": {"width": 32, "layers": 2, "heads": 4, "feed_forward": 64,
                      "min_wavelength_s": 0.01, "max_wavelength_s": 1000.0,
                      "attention": "dense", "tokenizer": {"kind": "samples"},
                      "dropout": 0.0, "drop_path": 0.0},
            "optim": {"learning_rate": 3e-3, "min_learning_rate": 3e-4, "warmup_steps": 5,
                      "steps": 40, "weight_decay": 0.01, "beta1": 0.9, "beta2": 0.95,
                      "grad_clip": 1.0, "schedule": "cosine", "decay_steps": None,
                      "decay_shape": None},
            "run": {"name": "test", "seed": 0, "device": "cpu", "precision": "fp32",
                    "log_every": 10, "validate_every": 20, "validation_batches": 2,
                    "checkpoint_every": 40, "checkpoint_prefix": "models/test",
                    "wandb_project": "p", "wandb_entity": "e", "wandb_mode": "disabled",
                    "max_bad_steps": 3, "bad_batch_directory": str(tmp_path / "bad"),
                    "resume": "never"},
        }
    )  # fmt: skip
    first = train(config.model_copy(update={"optim": config.optim.model_copy(update={"steps": 1})}),
                  StoreFactory(str(store.root), None), "abc")  # fmt: skip
    last = train(config, StoreFactory(str(store.root), None), "abc")
    assert last["train/loss"] < first["train/loss"]
    assert "validation/loss" in last and "train/loss_by_type/eeg" in last
    assert store.exists("models/test/step-00000040.pt")
    assert store.exists("models/test/config.json")
    assert not list(tmp_path.rglob("*.wandb"))  # nothing kept for upload


def _short_config(store: LocalStore, tmp_path: Path, **run: object) -> TrainConfig:
    _index(store, {f"r{i}": 40.0 for i in range(4)} | {"held-out": 40.0})
    store.put_bytes("index/i1/_manifest.json", b'{"version": "i1"}')
    return TrainConfig.model_validate(
        {
            "data": {"store": str(store.root), "index": "i1", "train_split": "train",
                     "validation_split": "test", "signal_types": None, "datasets": None,
                     "cache_directory": str(tmp_path / "cache"), "cache_bytes": 10**9,
                     "block_s": 20.0, "shuffle_buffer": 16, "reads_in_flight": 2, "workers": 0,
                     "mixture": None, "pieces_per_epoch": 1},
            "tokens": {"max_tokens": 128, "samples_per_token": 16, "max_channels": 4,
                       "rows": 2, "mask_ratio": 0.4},
            "model": {"width": 32, "layers": 1, "heads": 4, "feed_forward": 64,
                      "min_wavelength_s": 0.01, "max_wavelength_s": 1000.0,
                      "attention": "dense", "tokenizer": {"kind": "samples"},
                      "dropout": 0.0, "drop_path": 0.0},
            "optim": {"learning_rate": 1e-3, "min_learning_rate": 1e-4, "warmup_steps": 2,
                      "steps": 10, "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95,
                      "grad_clip": 1.0, "schedule": "cosine", "decay_steps": None,
                      "decay_shape": None},
            "run": {"name": "t", "seed": 0, "device": "cpu", "precision": "fp32",
                    "log_every": 5, "validate_every": 10, "validation_batches": 1,
                    "checkpoint_every": 5, "checkpoint_prefix": "models/t",
                    "wandb_project": "p", "wandb_entity": "e", "wandb_mode": "disabled",
                    "max_bad_steps": 3, "bad_batch_directory": str(tmp_path / "bad"),
                    "resume": "never"} | run,
        }
    )  # fmt: skip


def test_a_checkpoint_from_before_a_field_existed_is_read_as_its_run_was(
    store: LocalStore, tmp_path: Path
) -> None:
    config = _short_config(store, tmp_path)
    stored = config.model_dump(mode="json")
    for section, name in ADDED_LATER:
        del stored[section][name]  # as v1-pretrain wrote it
    with pytest.raises(ValueError, match="pieces_per_epoch"):
        TrainConfig.model_validate(stored)
    assert TrainConfig.from_checkpoint(stored) == config
    assert TrainConfig.from_checkpoint(config.model_dump(mode="json")) == config


def test_a_non_finite_step_is_not_applied_and_too_many_stop_the_run(
    store: LocalStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import biosignal_model.train as module  # noqa: PLC0415

    real = module.masked_loss
    calls = {"n": 0}

    def sometimes_nan(predicted: torch.Tensor, x: object) -> tuple[torch.Tensor, torch.Tensor]:
        loss, per_token = real(predicted, x)  # type: ignore[arg-type]
        calls["n"] += 1
        return (loss * float("nan") if calls["n"] == 2 else loss), per_token

    monkeypatch.setattr(module, "masked_loss", sometimes_nan)
    config = _short_config(store, tmp_path)
    train(config, StoreFactory(str(store.root), None), "abc")  # one bad step: the run goes on
    assert [p.name for p in (tmp_path / "bad").iterdir()] == ["bad-step-00000002.pt"]
    weights = torch.load(store._path("models/t/step-00000010.pt"), weights_only=True)["model"]
    assert all(torch.isfinite(t).all() for t in weights.values())

    monkeypatch.setattr(
        module, "masked_loss", lambda p, x: (real(p, x)[0] * float("nan"), real(p, x)[1])
    )
    with pytest.raises(FloatingPointError, match="3 steps in a row"):
        train(config, StoreFactory(str(store.root), None), "abc")


def test_a_run_resumes_from_its_newest_checkpoint(store: LocalStore, tmp_path: Path) -> None:
    config = _short_config(store, tmp_path, resume="newest")
    short = config.model_copy(update={"optim": config.optim.model_copy(update={"steps": 5})})
    train(short, StoreFactory(str(store.root), None), "abc")
    assert store.exists("models/t/step-00000005.pt") and not store.exists(
        "models/t/step-00000010.pt"
    )
    train(config, StoreFactory(str(store.root), None), "abc")  # steps 6 to 10 only
    resumed = torch.load(store._path("models/t/step-00000010.pt"), weights_only=True)
    assert resumed["step"] == 10 and "wandb_id" in resumed


def test_selection_without_recordings_is_refused(store: LocalStore) -> None:
    _index(store, {"r1": 10.0})
    assert Selection.load(store, "i1", split="validation", signal_types=None).recordings == ()


def test_an_infinite_value_is_never_observed() -> None:
    example = _example([20], [100.0])
    example.signals[0].values[4] = np.inf
    tokens = _packer(mask_ratio=0.0).tokens(example)
    assert np.isfinite(tokens.values).all() and np.isfinite(tokens.channel_scale).all()
    assert not tokens.observed.reshape(-1)[4]


def test_the_wsd_schedule_is_constant_until_its_decay_and_extends_without_a_jump() -> None:
    from biosignal_model.train import OptimSection, learning_rate  # noqa: PLC0415

    def wsd(steps: int) -> OptimSection:
        return OptimSection(
            learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10, steps=steps,
            weight_decay=0.0, beta1=0.9, beta2=0.95, grad_clip=1.0, schedule="wsd",
            decay_steps=100, decay_shape="1-sqrt",
        )  # fmt: skip

    short, long = wsd(1000), wsd(5000)
    assert learning_rate(5, short) == pytest.approx(6e-4)  # warmup
    assert learning_rate(899, short) == learning_rate(899, long) == 1e-3  # stable
    assert learning_rate(999, short) == pytest.approx(1e-4, rel=0.2)  # end of its decay
    assert learning_rate(950, short) == pytest.approx(1e-4 + 9e-4 * (1 - 0.5**0.5))
    assert learning_rate(999, long) == 1e-3  # the longer run is still stable there


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _join(rank: int, port: int) -> None:
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))  # fmt: skip


def _two_row_batch() -> Batch:
    examples = [_example([40 + 8 * i, 24], [100.0, 50.0]) for i in range(12)]
    return next(iter(_packer().batches(examples)))


def _rows(x: Inputs, rows: slice) -> Inputs:
    return Inputs(**{k: None if v is None else v[rows] for k, v in vars(x).items()})


_NO_CLIP = OptimSection(
    learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=1, steps=1, schedule="cosine",
    decay_steps=None, decay_shape=None, weight_decay=0.0, beta1=0.9, beta2=0.95, grad_clip=1e9,
)  # fmt: skip


def _step_on_one_row(rank: int, port: int, out: str) -> None:
    import torch.distributed as dist  # noqa: PLC0415
    from torch.nn.parallel import DistributedDataParallel  # noqa: PLC0415

    _join(rank, port)
    dist.init_process_group("gloo")
    torch.manual_seed(0)
    model = Encoder(_config())
    optimizer = torch.optim.AdamW(model.parameters())
    x = _rows(_two_row_batch().inputs, slice(rank, rank + 1))
    loss, _, _ = _forward_backward(
        DistributedDataParallel(model), optimizer, x, torch.autocast("cpu", enabled=False),
        1e-3, _NO_CLIP, 2,
    )  # fmt: skip
    if rank == 0:
        torch.save({"loss": loss, "grads": [p.grad for p in model.parameters()]}, out)
    dist.destroy_process_group()


def test_two_processes_with_one_row_each_step_as_one_process_with_both(tmp_path: Path) -> None:
    out = str(tmp_path / "two.pt")
    torch.multiprocessing.spawn(_step_on_one_row, args=(_free_port(), out), nprocs=2)
    torch.manual_seed(0)
    model = Encoder(_config())
    loss, _, _ = _forward_backward(
        model, torch.optim.AdamW(model.parameters()), _two_row_batch().inputs,
        torch.autocast("cpu", enabled=False), 1e-3, _NO_CLIP, 1,
    )  # fmt: skip
    two = torch.load(out, weights_only=True)
    assert torch.allclose(two["loss"], loss, rtol=1e-5)
    for mine, theirs in zip([p.grad for p in model.parameters()], two["grads"], strict=True):
        assert torch.allclose(mine, theirs, rtol=1e-4, atol=1e-7)


def _train_as_rank(rank: int, port: int, config_json: str, root: str) -> None:
    _join(rank, port)
    train(TrainConfig.model_validate_json(config_json), StoreFactory(root, None), "abc")


def test_a_run_on_two_processes_writes_one_checkpoint_from_the_first(
    store: LocalStore, tmp_path: Path
) -> None:
    config = _short_config(store, tmp_path)  # rows 2: one a process
    torch.multiprocessing.spawn(
        _train_as_rank, args=(_free_port(), config.model_dump_json(), str(store.root)), nprocs=2
    )
    saved = torch.load(store._path("models/t/step-00000010.pt"), weights_only=True)
    assert saved["step"] == 10
    assert all(torch.isfinite(t).all() for t in saved["model"].values())
    assert json.loads(store.get_bytes("models/t/config.json") or b"{}")["gpus"] == 2
    odd = config.model_copy(update={"tokens": config.tokens.model_copy(update={"rows": 3})})
    with pytest.raises(Exception, match="cannot be split over 2 GPUs"):
        torch.multiprocessing.spawn(
            _train_as_rank, args=(_free_port(), odd.model_dump_json(), str(store.root)), nprocs=2
        )


def test_single_rows_joined_are_the_rows_packed_together() -> None:
    examples = [_example([40 + 8 * i, 24], [100.0, 50.0]) for i in range(12)]
    together = next(iter(_packer(rows=2).batches(examples)))
    singles = list(_packer(rows=1).batches(examples))[:2]
    joined = concat(singles)
    for k, v in vars(together.inputs).items():
        other = vars(joined.inputs)[k]
        assert (v is None and other is None) or torch.equal(v, other), k
    for k, v in vars(together.analysis).items():
        assert torch.equal(v, vars(joined.analysis)[k]), k
    assert (joined.examples, joined.samples) == (together.examples, together.samples)


def test_the_validation_batches_are_kept_and_the_same_on_every_start(
    store: LocalStore, tmp_path: Path
) -> None:
    config = _short_config(store, tmp_path)
    factory = StoreFactory(str(store.root), None)
    stream = _stream(config, factory, "test", 0, 1, config.tokens.rows)
    first = _validation_batches(stream, config, store)
    kept = [k for k in store.list("models/validation-sets/i1/") if k.endswith(".pt")]
    assert len(kept) == 1
    store.put_bytes("index/i1/_moved", b"")  # nothing else changes: the same key is read
    again = _validation_batches(_stream(config, factory, "test", 0, 1, config.tokens.rows),
                                config, store)  # fmt: skip
    assert len(first) == len(again) == config.run.validation_batches
    for a, b in zip(first, again, strict=True):
        assert torch.equal(a.inputs.values, b.inputs.values)
        assert torch.equal(a.inputs.hidden, b.inputs.hidden)
