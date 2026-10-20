"""Command-line interface: `biosignal-model --help`.

    train   masked-token pretraining, with every setting from one YAML file

S3 settings come from biosignal-data.yaml (`s3` section); keys from the
BIOSIGNAL_ACCESS_KEY_ID / BIOSIGNAL_SECRET_ACCESS_KEY variables; the Weights & Biases key
from WANDB_API_KEY.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer

from biosignal_data.config import ConfigError, load_config, need
from biosignal_model.train import StoreFactory, TrainConfig, train

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
