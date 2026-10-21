"""Project settings from `biosignal-data.yaml` at the repository root.

Settings that do not change between runs live here, not in command options. Nothing has
a default: a value that a run needs and the file does not give is an error. Secrets are
not stored here; they come from BIOSIGNAL_* environment variables (see docs/access.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_NAME = "biosignal-data.yaml"
T = TypeVar("T")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class S3Config(_Section):
    """The S3-compatible storage used for s3:// destinations."""

    endpoint: str = Field(pattern=r"^https?://")  # used as written, e.g. https://s3.hippius.com
    region: str  # the region name the storage signs with, e.g. "decentralized"
    addressing: Literal["path", "virtual"]  # https://host/bucket/key or https://bucket.host/key


class DownloadConfig(_Section):
    workers: int = Field(ge=1)  # parallel file downloads, shared by all datasets of a run
    parallel_datasets: int = Field(ge=1)  # datasets listed and copied at the same time
    # All requests to physionet.org at the same time, over every dataset of a run.
    physionet_website_connections: int = Field(ge=1)


class StandardizeConfig(_Section):
    workers: int = Field(ge=1)  # batches converted at the same time
    part_bytes: int = Field(ge=1)  # a batch of recordings closes when its raw files reach this
    # recordings read, compressed and checked in this many processes (a raw version on
    # local disk only, e.g. a staged ZIP); 0: in the batch threads
    processes: int = Field(ge=0)


class QualityConfig(_Section):
    """`biosignal-data quality`: what counts as flat, and the sample for percentiles."""

    flat_s: float = Field(gt=0)  # a run of one stored value counts as flat from here on
    percentile_sample: int = Field(ge=100)  # valid values per channel for p01, p50, p99
    seed: int
    workers: int = Field(ge=1)  # recordings measured at the same time


class StagingConfig(_Section):
    """Disk for the sources whose tools must write files before upload (TUH, NSRR, Kaggle)."""

    dir: Path
    keep: bool  # keep the staged files after the upload


class ReadConfig(_Section):
    cache_dir: Path  # where raw files are kept after `RawDataset.local_path` fetches them


class Config(_Section):
    s3: S3Config | None = None
    download: DownloadConfig | None = None
    standardize: StandardizeConfig | None = None
    quality: QualityConfig | None = None
    staging: StagingConfig | None = None
    read: ReadConfig | None = None


class ConfigError(ValueError):
    pass


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / CONFIG_NAME


def load_config(path: Path | None = None) -> tuple[Config, Path]:
    """The settings and the file they came from."""
    path = path or default_config_path()
    if not path.is_file():
        raise ConfigError(f"no config file at {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return Config.model_validate(data), path
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def need(section: T | None, name: str, path: Path) -> T:
    """A section that this run needs; an error naming the file if it is missing."""
    if section is None:
        raise ConfigError(f"{path} has no `{name}` section, which this run needs")
    return section
