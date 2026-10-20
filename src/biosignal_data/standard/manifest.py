"""The manifest of a standard version: what was made from which raw version, and every raw
file that was not converted, with the reason. Its own module, so that readers of the
standard layer (biosignal_data.read) need not import the converter."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 1


class NotConverted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str


class StandardManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    dataset_id: str
    raw_version: str
    raw_manifest_sha256: str
    license: str | None
    tool: str
    created_at: datetime
    completed_at: datetime
    layout: dict[str, Any]
    recordings: int
    channels: int
    events: int
    files_used: int
    problems: list[str]  # e.g. RECORDS entries without files
    not_converted: list[NotConverted]
