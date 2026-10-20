"""Masked-token pretraining: the run, its settings, its logs.

Every setting is in one YAML file (see `TrainConfig`); nothing has a default. The run
logs metrics to Weights & Biases (no artifacts: no checkpoints, data or code go there) and
writes checkpoints to the store under `run.checkpoint_prefix`.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import yaml
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from biosignal_data.index.build import ROOT as INDEX_ROOT
from biosignal_data.storage.stores import S3Settings, Store, open_store
from biosignal_data.training.loader import ExampleSource, Selection
from biosignal_model.batch import SIGNAL_TYPES, Analysis, Batch, Packer
from biosignal_model.model import Encoder, ModelConfig, masked_loss


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataSection(_Section):
    store: str  # s3://bucket or a local path
    index: str  # the index version, e.g. 2026-10-20
    train_split: str
    validation_split: str
    signal_types: list[str] | None  # None: every channel
    datasets: list[str] | None  # None: every dataset of the index
    cache_directory: Path
    cache_bytes: int
    shuffle_buffer: int
    reads_in_flight: int
    workers: int  # data loader processes


class TokenSection(_Section):
    max_tokens: int
    samples_per_token: int
    max_channels: int
    rows: int  # rows of max_tokens tokens in one batch
    mask_ratio: float


class ModelSection(_Section):
    width: int
    layers: int
    heads: int
    feed_forward: int
    min_wavelength_s: float
    max_wavelength_s: float
    attention: Literal["flex", "dense"]


class OptimSection(_Section):
    learning_rate: float
    min_learning_rate: float  # the end of the cosine decay
    warmup_steps: int
    steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float


class RunSection(_Section):
    name: str
    seed: int
    device: str  # cuda or cpu
    precision: Literal["bf16", "fp32"]
    log_every: int
    validate_every: int
    validation_batches: int
    checkpoint_every: int
    checkpoint_prefix: str  # in the store, e.g. models/<name>
    wandb_project: str
    wandb_entity: str
    wandb_mode: Literal["online", "offline", "disabled"]


class TrainConfig(_Section):
    data: DataSection
    tokens: TokenSection
    model: ModelSection
    optim: OptimSection
    run: RunSection

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def model_settings(self) -> ModelConfig:
        return ModelConfig(
            samples_per_token=self.tokens.samples_per_token,
            max_channels=self.tokens.max_channels,
            **self.model.model_dump(),
        )


class StoreFactory:
    """Opens the store in each process (a store does not cross processes)."""

    def __init__(self, uri: str, s3: tuple[str, str, Literal["path", "virtual"]] | None):
        self.uri = uri
        self.s3 = s3

    def __call__(self) -> Store:
        if self.s3 is None:
            return open_store(self.uri)
        return open_store(self.uri, S3Settings.from_env(*self.s3))


class BatchStream(IterableDataset[Batch]):
    """Batches of one split; each data loader worker of each rank packs its own part."""

    def __init__(self, source: ExampleSource, config: TrainConfig, datasets: list[str]):
        super().__init__()
        self.source = source
        self.config = config
        self.datasets = datasets
        self.epoch = 0

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        rank, ranks = (
            (dist.get_rank(), dist.get_world_size())
            if dist.is_available() and dist.is_initialized()
            else (0, 1)
        )
        part, parts = rank * workers + worker, ranks * workers
        t = self.config.tokens
        packer = Packer(
            datasets=self.datasets, max_tokens=t.max_tokens,
            samples_per_token=t.samples_per_token, max_channels=t.max_channels, rows=t.rows,
            mask_ratio=t.mask_ratio,
            seed=(self.config.run.seed * 1_000_003 + self.epoch) * 1_009 + part,
        )  # fmt: skip
        yield from packer.batches(self.source.examples(self.epoch, part, parts))


def _stream(config: TrainConfig, factory: StoreFactory, split: str) -> BatchStream:
    d, t = config.data, config.tokens
    store = factory()
    selection = Selection.load(
        store, d.index, split=split,
        signal_types=set(d.signal_types) if d.signal_types is not None else None,
        datasets=set(d.datasets) if d.datasets is not None else None,
    )  # fmt: skip
    if not selection.recordings:
        raise ValueError(f"index {d.index}: no recordings in split {split!r} for this selection")
    source = ExampleSource(
        selection, factory, cache_directory=d.cache_directory, cache_bytes=d.cache_bytes,
        max_tokens=t.max_tokens, samples_per_token=t.samples_per_token,
        max_channels=t.max_channels, shuffle_buffer=d.shuffle_buffer,
        reads_in_flight=d.reads_in_flight, seed=config.run.seed,
    )  # fmt: skip
    datasets = sorted({r.dataset_id for r in selection.recordings})
    return BatchStream(source, config, datasets)


def _loader(stream: BatchStream, workers: int, device: torch.device) -> DataLoader[Batch]:
    return DataLoader(
        stream, batch_size=None, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=False, prefetch_factor=2 if workers else None,
    )  # fmt: skip


def learning_rate(step: int, o: OptimSection) -> float:
    if step < o.warmup_steps:
        return o.learning_rate * (step + 1) / o.warmup_steps
    progress = min(1.0, (step - o.warmup_steps) / max(1, o.steps - o.warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return o.min_learning_rate + (o.learning_rate - o.min_learning_rate) * cosine


class _Totals:
    """Squared error and count, split by dataset and by signal type, for reports."""

    def __init__(self, datasets: list[str], device: torch.device) -> None:
        self.datasets = datasets
        self.by_dataset = torch.zeros(len(datasets), 2, device=device)
        self.by_type = torch.zeros(len(SIGNAL_TYPES), 2, device=device)

    def add(self, analysis: Analysis, per_token: torch.Tensor) -> None:
        for index, totals in ((analysis.dataset, self.by_dataset),
                              (analysis.signal_type, self.by_type)):  # fmt: skip
            keep = index >= 0
            totals.index_add_(0, index[keep], per_token[keep])

    def report(self, prefix: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for names, totals, kind in ((self.datasets, self.by_dataset, "dataset"),
                                    (SIGNAL_TYPES, self.by_type, "type")):  # fmt: skip
            for name, (error, count) in zip(names, totals.tolist(), strict=True):
                if count > 0:
                    out[f"{prefix}/loss_by_{kind}/{name}"] = error / count
        return out


class _Window:
    """What happened since the last log line."""

    def __init__(self, datasets: list[str], device: torch.device) -> None:
        self.totals = _Totals(datasets, device)
        self.steps = 0
        self.loss = self.tokens = self.padding = self.hidden = 0.0
        self.samples = self.examples = self.wait = 0.0
        self.start = time.perf_counter()

    def add(self, batch: Batch, loss: float, per_token: torch.Tensor, waited: float) -> None:
        real = batch.inputs.document >= 0
        self.steps += 1
        self.loss += loss
        self.tokens += real.sum().item()
        self.padding += (~real).sum().item()
        self.hidden += batch.inputs.hidden.sum().item()
        self.samples += batch.samples
        self.examples += batch.examples
        self.wait += waited
        self.totals.add(batch.analysis.to(per_token.device), per_token)

    def report(self) -> dict[str, float]:
        seconds = time.perf_counter() - self.start
        return {
            "train/loss": self.loss / self.steps,
            "speed/tokens_per_s": self.tokens / seconds,
            "speed/examples_per_s": self.examples / seconds,
            "speed/samples_per_s": self.samples / seconds,
            "speed/data_wait_fraction": self.wait / seconds,
            "batch/padding_fraction": self.padding / max(1.0, self.tokens + self.padding),
            "batch/hidden_fraction": self.hidden / max(1.0, self.tokens),
            "batch/examples": self.examples / self.steps,
        } | self.totals.report("train")


def _optimizer(model: Encoder, o: OptimSection, device: torch.device) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": o.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=o.learning_rate, betas=(o.beta1, o.beta2), fused=device.type == "cuda",
    )  # fmt: skip


def train(config: TrainConfig, factory: StoreFactory, code_commit: str) -> dict[str, float]:
    """Run the pretraining; returns the last logged metrics."""
    import wandb  # noqa: PLC0415 - the `train` extra

    run_cfg, o = config.run, config.optim
    torch.manual_seed(run_cfg.seed)
    device = torch.device(run_cfg.device)
    store = factory()
    index_manifest = store.get_bytes(f"{INDEX_ROOT}/{config.data.index}/_manifest.json")
    if index_manifest is None:
        raise FileNotFoundError(f"index {config.data.index} has no manifest")
    train_stream = _stream(config, factory, config.data.train_split)
    validation_stream = _stream(config, factory, config.data.validation_split)
    model = Encoder(config.model_settings()).to(device)
    optimizer = _optimizer(model, o, device)
    described = config.model_dump(mode="json") | {"code_commit": code_commit}
    store.put_bytes(
        f"{run_cfg.checkpoint_prefix}/config.json", json.dumps(described, indent=2).encode()
    )
    run = wandb.init(
        project=run_cfg.wandb_project, entity=run_cfg.wandb_entity, name=run_cfg.name,
        mode=run_cfg.wandb_mode, save_code=False,
        config=described | {
            "parameters": sum(p.numel() for p in model.parameters()),
            "index_manifest": json.loads(index_manifest),
            "train_recordings": len(train_stream.source.selection.recordings),
            "validation_recordings": len(validation_stream.source.selection.recordings),
        },
    )  # fmt: skip
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=run_cfg.precision == "bf16"
    )
    window = _Window(train_stream.datasets, device)
    last: dict[str, float] = {}
    batches = _epochs(train_stream, config.data.workers, device)
    for step in range(1, o.steps + 1):
        before = time.perf_counter()
        batch = next(batches)
        x = batch.inputs.to(device)
        waited = time.perf_counter() - before
        rate = learning_rate(step - 1, o)
        for group in optimizer.param_groups:
            group["lr"] = rate
        with autocast:
            predicted = model(x)
        loss, per_token = masked_loss(predicted, x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
        optimizer.step()
        window.add(batch, loss.item(), per_token, waited)
        if step % run_cfg.log_every == 0 or step == o.steps:
            last = window.report() | {"train/grad_norm": float(grad_norm),
                                      "train/learning_rate": rate}  # fmt: skip
            if device.type == "cuda":
                last["system/max_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
            run.log(last, step=step)
            window = _Window(train_stream.datasets, device)
        if step % run_cfg.validate_every == 0 or step == o.steps:
            validation = _validate(model, validation_stream, config, device, autocast)
            run.log(validation, step=step)
            last |= validation
            window.start = time.perf_counter()  # validation time is not training time
        if step % run_cfg.checkpoint_every == 0 or step == o.steps:
            _checkpoint(store, run_cfg.checkpoint_prefix, step, model, optimizer, config)
            window.start = time.perf_counter()
    run.finish()
    return last


def _epochs(stream: BatchStream, workers: int, device: torch.device) -> Iterator[Batch]:
    for epoch in itertools.count():
        stream.epoch = epoch
        yield from _loader(stream, workers, device)


@torch.no_grad()
def _validate(
    model: Encoder,
    stream: BatchStream,
    config: TrainConfig,
    device: torch.device,
    autocast: Any,
) -> dict[str, float]:
    """The same first batches of the validation split every time, so values compare."""
    model.eval()
    stream.epoch = 0
    totals = _Totals(stream.datasets, device)
    error, count = 0.0, 0.0
    batches = itertools.islice(
        _loader(stream, config.data.workers, device), config.run.validation_batches
    )
    for batch in batches:
        x = batch.inputs.to(device)
        with autocast:
            predicted = model(x)
        _, per_token = masked_loss(predicted, x)
        totals.add(batch.analysis.to(device), per_token)
        error += per_token[..., 0].sum().item()
        count += per_token[..., 1].sum().item()
    model.train()
    return {"validation/loss": error / max(1.0, count)} | totals.report("validation")


def _checkpoint(
    store: Store,
    prefix: str,
    step: int,
    model: Encoder,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> None:
    out = io.BytesIO()
    torch.save(
        {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "config": config.model_dump(mode="json")},
        out,
    )  # fmt: skip
    store.put_bytes(f"{prefix}/step-{step:08d}.pt", out.getvalue())
