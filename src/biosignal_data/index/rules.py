"""index.yaml: per dataset, how recordings map to persons, and channel types that only the
dataset's own documentation can tell. Each rule names its source.

A dataset without an entry gets the default: each recording is its own subject, marked as
not certain, and channel types from the general rules only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

from biosignal_data.index.signals import SignalType

REPO_ROOT = Path(__file__).resolve().parents[3]


def default_rules_path() -> Path:
    return REPO_ROOT / "index.yaml"


class RulesError(ValueError):
    """index.yaml is invalid; the message names the entry and the problem."""


class SubjectRuleKind(StrEnum):
    PATTERN = "pattern"  # a regex on the recording id; group 1 is the subject
    ONE_PER_RECORDING = "one_per_recording"  # the documentation says so
    UNKNOWN = "unknown"  # the documentation does not say: one per recording, not certain


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SubjectRule(_Model):
    rule: SubjectRuleKind
    pattern: str | None = None
    same_person: list[list[str]] = Field(default_factory=list)  # subject ids of one person
    certain: bool
    source: HttpUrl
    evidence: str | None = None  # an exact quote from the source
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SubjectRule:
        if (self.rule is SubjectRuleKind.PATTERN) != (self.pattern is not None):
            raise ValueError("`pattern` is given exactly when rule is pattern")
        if self.pattern is not None and re.compile(self.pattern).groups < 1:
            raise ValueError("`pattern` needs a group: group 1 is the subject id")
        if self.rule is SubjectRuleKind.UNKNOWN and self.certain:
            raise ValueError("an unknown rule cannot be certain")
        if self.certain and not self.evidence:
            raise ValueError("a certain rule needs `evidence`, quoted from the source")
        return self


class ChannelRule(_Model):
    name: str  # a regex on the channel name as the source wrote it
    type: SignalType
    source: HttpUrl | None = None
    notes: str | None = None


class DatasetRules(_Model):
    id: str
    subject: SubjectRule | None = None
    channels: list[ChannelRule] = Field(default_factory=list)


@dataclass(frozen=True)
class IndexRules:
    datasets: dict[str, DatasetRules]
    path: Path

    @classmethod
    def load(cls, path: Path | None = None) -> IndexRules:
        path = path or default_rules_path()
        try:
            raw = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
        except (OSError, yaml.YAMLError) as exc:
            raise RulesError(f"{path}: {exc}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), list):
            raise RulesError(f"{path}: expected a top-level `datasets:` list")
        entries: dict[str, DatasetRules] = {}
        for i, item in enumerate(raw["datasets"]):
            try:
                entry = DatasetRules.model_validate(item)
            except ValidationError as exc:
                name = item.get("id", f"entry {i}") if isinstance(item, dict) else f"entry {i}"
                raise RulesError(f"{path}: {name}: {exc}") from exc
            if entry.id in entries:
                raise RulesError(f"{path}: duplicate id {entry.id!r}")
            entries[entry.id] = entry
        return cls(entries, path)

    def get(self, dataset_id: str) -> DatasetRules:
        return self.datasets.get(dataset_id) or DatasetRules(id=dataset_id)


def channel_overrides(rules: DatasetRules) -> list[tuple[re.Pattern[str], SignalType]]:
    return [(re.compile(c.name), c.type) for c in rules.channels]
