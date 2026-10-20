"""Command-line interface: `biosignal-model --help`.

    train      masked-token pretraining, with every setting from one YAML file
    benchmark  score a checkpoint on a benchmark (apnea), with its official protocol

S3 settings come from biosignal-data.yaml (`s3` section); keys from the
BIOSIGNAL_ACCESS_KEY_ID / BIOSIGNAL_SECRET_ACCESS_KEY variables; the Weights & Biases key
from WANDB_API_KEY.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer

from biosignal_data.config import ConfigError, load_config, need
from biosignal_data.storage.stores import StoreFactory
from biosignal_model.train import TrainConfig, train

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
REPO = Path(__file__).resolve().parents[2]


@app.callback()
def main() -> None:
    """Model v0 for the biosignal foundation model."""


@app.command("train")
def train_(
    config: Annotated[Path, typer.Argument(help="The run's settings (YAML).")],
    settings: Annotated[
        Path | None, typer.Option(help="biosignal-data.yaml, for the S3 settings.")
    ] = None,
) -> None:
    """Masked-token pretraining. Logs to Weights & Biases (no artifacts); checkpoints go
    to the store."""
    run = TrainConfig.load(config)
    s3 = None
    if run.data.store.startswith("s3://"):
        try:
            loaded, path = load_config(settings)
            section = need(loaded.s3, "s3", path)
        except ConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        s3 = (section.endpoint, section.region, section.addressing)
    last = train(run, StoreFactory(run.data.store, s3), _commit())
    typer.echo(
        " ".join(f"{k}={v:.4g}" for k, v in sorted(last.items()) if "/" in k and "_by_" not in k)
    )


@app.command("benchmark")
def benchmark(
    name: Annotated[str, typer.Argument(help="The benchmark: apnea or ptbxl.")],
    config: Annotated[Path, typer.Argument(help="The benchmark run's settings (YAML).")],
    settings: Annotated[
        Path | None, typer.Option(help="biosignal-data.yaml, for the S3 settings.")
    ] = None,
) -> None:
    """Score a checkpoint: a linear probe on the frozen encoder, the same with random
    weights, and the same architecture trained from zero. Results go to Weights & Biases
    (no artifacts) and to the store, next to the checkpoint."""
    import io  # noqa: PLC0415
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import wandb  # noqa: PLC0415

    from biosignal_model.benchmarks import BenchmarkRun, apnea, ptbxl  # noqa: PLC0415

    modules: dict[str, Any] = {
        "apnea": (apnea, apnea.ApneaSettings), "ptbxl": (ptbxl, ptbxl.PtbxlSettings),
    }  # fmt: skip
    if name not in modules:
        raise typer.BadParameter(f"unknown benchmark {name!r}; known: {', '.join(modules)}")
    module, settings_class = modules[name]
    run = BenchmarkRun.load(config)
    bench_settings = settings_class.model_validate(run.settings)
    s3 = None
    if run.store.startswith("s3://"):
        try:
            loaded, path = load_config(settings)
            section = need(loaded.s3, "s3", path)
        except ConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        s3 = (section.endpoint, section.region, section.addressing)
    store = StoreFactory(run.store, s3)()
    data = store.get_bytes(run.checkpoint)
    if data is None:
        raise typer.BadParameter(f"{run.checkpoint} is not in {run.store}")
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    described = run.model_dump(mode="json") | {"code_commit": _commit(), "step": checkpoint["step"]}
    logged = wandb.init(
        project=run.wandb_project, entity=run.wandb_entity, mode=run.wandb_mode,
        job_type="benchmark", name=f"{name} {run.checkpoint}", save_code=False, config=described,
    )  # fmt: skip

    def log(metrics: dict[str, float]) -> None:
        """Each record of progress to W&B, and the same line to the console."""
        logged.log(metrics)
        line = " ".join(f"{k}={v:.5g}" for k, v in metrics.items())
        typer.echo(f"{time.strftime('%H:%M:%S')} {line}")

    results = module.run(store, checkpoint, bench_settings, torch.device(run.device), log)
    logged.summary.update({f"{name}/{k}": v for k, v in results.items()})
    logged.finish()
    key = (
        run.checkpoint.rsplit("/", 1)[0] + f"/benchmarks/{name}-step-{checkpoint['step']:08d}.json"
    )
    store.put_bytes(key, json.dumps(described | {"results": results}, indent=2).encode())
    for k, v in sorted(results.items()):
        typer.echo(f"{k}: {v:.4f}")


def _commit() -> str:
    """The git commit of this code; a run from a changed tree is refused, so that the
    commit names the code exactly."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        changed = git("status", "--porcelain", "--untracked-files=no")
        commit = git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise typer.BadParameter(f"{REPO} is not a git checkout: {exc}") from exc
    if changed:
        raise typer.BadParameter(f"uncommitted changes in {REPO}; commit them first")
    return commit
