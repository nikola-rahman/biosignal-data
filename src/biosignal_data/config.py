"""Project settings from `biosignal-data.yaml` at the repository root.

Only settings that do not change between runs belong here, e.g. how to reach the S3
storage. Nothing has a default: a missing value is an error. Secrets are not stored here;
the S3 keys come from BIOSIGNAL_ACCESS_KEY_ID and BIOSIGNAL_SECRET_ACCESS_KEY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_NAME = "biosignal-data.yaml"


class S3Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str = Field(pattern=r"^https?://")  # used as written, e.g. https://s3.hippius.com
    region: str  # the region name the storage signs with, e.g. "decentralized"
    addressing: Literal["path", "virtual"]  # https://host/bucket/key or https://bucket.host/key


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    s3: S3Config | None = None  # needed only for s3:// destinations


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
