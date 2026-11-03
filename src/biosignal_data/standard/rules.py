"""standard.yaml: per dataset, what only the dataset's own documentation can tell the
standard layer about its files. Each rule names its source and quotes it.

A dataset without an entry is converted by the general rules of discover.py only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]


def default_rules_path() -> Path:
    return REPO_ROOT / "standard.yaml"


class StandardRulesError(ValueError):
    """standard.yaml is invalid; the message names the problem."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EdfAnnotationFiles(_Model):
    """EDF+ files that hold only the annotations of an EDF recording, in the same folder: a
    file ending in `annotation_suffix` belongs to the recording ending in
    `recording_suffix` whose file name starts with the same `same_first_characters`
    characters. Converting checks that both files start at the same time."""

    recording_suffix: str
    annotation_suffix: str
    same_first_characters: int = Field(gt=0)
    source: HttpUrl
    evidence: str  # an exact quote from the source
    notes: str | None = None


class EdfSuffixes(_Model):
    """EDF files that the dataset names with another suffix (ISRUC-Sleep: `.rec`): they are
    recordings like `.edf` files. Converting checks that each is an EDF file (its header)."""

    suffixes: list[str] = Field(min_length=1)  # lowercase, with the dot
    source: HttpUrl
    evidence: str  # an exact quote from the source
    notes: str | None = None


class PickleRecordings(_Model):
    """Recordings that the dataset publishes only as Python pickles of NumPy arrays
    (PPG-DaLiA: `S<n>/S<n>.pkl`, the synchronised signals and the labels). `reader` names the
    reader for the dataset's layout (standard/sources.py); `pattern` is a regex on the raw
    path. The pickles are read with an unpickler that builds NumPy arrays and Python
    containers only, never other objects."""

    reader: Literal["ppg-dalia"]
    pattern: str
    source: HttpUrl
    evidence: str  # an exact quote from the source
    notes: str | None = None
class UnitCorrection(_Model):
    """The physical unit of channels whose header names another one: the dataset's own
    documentation or its values show that the written unit is wrong. Only the `unit` of the
    channels table changes; the unit as written stays in the channel's `source`
    (`unit_as_written`), and the samples, scale and offset stay as stored. A channel that
    `channels` matches but whose header names another unit than `written` fails its
    recording: the rule does not describe that file."""

    channels: str  # a regex on the channel name as the source wrote it
    written: str  # the unit the header gives
    unit: str  # the true unit
    source: HttpUrl
    evidence: str  # exact quotes from the source
    checked: str  # how the values were compared: what shows the written unit is wrong
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> UnitCorrection:
        re.compile(self.channels)
        if self.written == self.unit:
            raise ValueError("`written` and `unit` are the same")
        return self

    def unit_of(self, name: str, written: str) -> str | None:
        """The true unit of channel `name`, or None if the rule is not for it."""
        if re.search(self.channels, name) is None:
            return None
        if written != self.written:
            raise ValueError(
                f"channel {name!r} has the unit {written!r}; the standard.yaml rule for "
                f"{self.channels!r} expects {self.written!r}"
            )
        return self.unit


class DatasetRules(_Model):
    id: str
    edf_annotation_files: EdfAnnotationFiles | None = None
    edf_suffixes: EdfSuffixes | None = None
    pickle_recordings: PickleRecordings | None = None
    units: list[UnitCorrection] = Field(default_factory=list)


class StandardRules(_Model):
    datasets: list[DatasetRules]

    @classmethod
    def load(cls, path: Path | None = None) -> StandardRules:
        path = path or default_rules_path()
        try:
            rules = cls.model_validate(yaml.safe_load(path.read_text()))
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise StandardRulesError(f"{path}: {exc}") from exc
        ids = [d.id for d in rules.datasets]
        if len(ids) != len(set(ids)):
            raise StandardRulesError(f"{path}: a dataset has two entries")
        return rules

    def get(self, dataset_id: str) -> DatasetRules | None:
        return next((d for d in self.datasets if d.id == dataset_id), None)
