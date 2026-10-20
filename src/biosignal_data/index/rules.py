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
    TABLE = "table"  # a table of the raw dataset; `pattern` group 1 is the key in it
    ONE_PER_RECORDING = "one_per_recording"  # the documentation says so
    UNKNOWN = "unknown"  # the documentation does not say: one per recording, not certain


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SubjectTable(_Model):
    """A delimited text file of the raw dataset (under data/), one row per recording."""

    file: str
    delimiter: str
    recording_column: str  # holds group 1 of the rule's pattern
    subject_column: str


class SubjectRule(_Model):
    rule: SubjectRuleKind
    pattern: str | None = None
    table: SubjectTable | None = None
    same_person: list[list[str]] = Field(default_factory=list)  # subject ids of one person
    certain: bool
    source: HttpUrl
    evidence: str | None = None  # an exact quote from the source
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SubjectRule:
        with_pattern = self.rule in (SubjectRuleKind.PATTERN, SubjectRuleKind.TABLE)
        if with_pattern != (self.pattern is not None):
            raise ValueError("`pattern` is given exactly when rule is pattern or table")
        if (self.rule is SubjectRuleKind.TABLE) != (self.table is not None):
            raise ValueError("`table` is given exactly when rule is table")
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


class ExcludeRule(_Model):
    """Recordings left out of the index: copies of, or extracts from, other recordings of
    the same dataset. Leaving a recording out is a claim about the data, so it needs a
    source and a quote, or `checked`: how the samples were compared."""

    pattern: str  # a regex on the recording id
    reason: str
    source: HttpUrl
    evidence: str | None = None
    checked: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ExcludeRule:
        re.compile(self.pattern)
        if not (self.evidence or self.checked):
            raise ValueError("an exclude rule needs `evidence` or `checked`")
        return self


class HoldOutTable(_Model):
    """A delimited text file of the raw dataset (under data/) that says which recordings
    are in the benchmark's test set: those whose `column` holds one of `values`."""

    file: str
    delimiter: str
    recording_column: str  # holds group 1 of the rule's pattern
    column: str
    values: list[str]


class HoldOutRule(_Model):
    """Recordings kept for a benchmark's test set: they get the split `held_out`, so that
    no pretraining sees them. The source names the official test set. Either every
    recording `pattern` matches, or, with `table`, those the table puts in the test set
    (then group 1 of `pattern` is the recording's key in the table)."""

    pattern: str  # a regex on the recording id
    table: HoldOutTable | None = None
    benchmark: str
    source: HttpUrl
    evidence: str

    @model_validator(mode="after")
    def _check(self) -> HoldOutRule:
        if self.table is not None and re.compile(self.pattern).groups < 1:
            raise ValueError("with `table`, `pattern` needs a group: the key in the table")
        re.compile(self.pattern)
        return self


class DatasetRules(_Model):
    id: str
    subject: SubjectRule | None = None
    exclude: list[ExcludeRule] = Field(default_factory=list)
    held_out: list[HoldOutRule] = Field(default_factory=list)
    channels: list[ChannelRule] = Field(default_factory=list)

    def excluded(self, recording_id: str) -> ExcludeRule | None:
        return next((e for e in self.exclude if re.search(e.pattern, recording_id)), None)

    def held_out_for(
        self, recording_id: str, tables: dict[int, dict[str, str]] | None = None
    ) -> HoldOutRule | None:
        """The rule that holds the recording out, if any. `tables`: for rules with a table,
        rule position -> {recording key: column value}, read from the raw dataset."""
        for i, rule in enumerate(self.held_out):
            match = re.search(rule.pattern, recording_id)
            if match is None:
                continue
            if rule.table is None:
                return rule
            if tables is None or i not in tables:
                raise ValueError(f"{self.id}: a held_out rule with a table needs its table")
            value = tables[i].get(match.group(1))
            if value is None:
                raise ValueError(f"{self.id}: {match.group(1)!r} is not in {rule.table.file}")
            if value in rule.table.values:
                return rule
        return None


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
