"""The subject of a recording, and the split of a subject.

A subject id is qualified by its dataset (`physionet/chbmit:chb01`): persons are matched
across datasets only where the sources say that the datasets share one id for a person
(a rule's `namespace`, e.g. `mimic-iv:10014354` for MIMIC-IV-ECG and the MIMIC-IV
waveforms).

The split of a subject comes from a hash of its qualified id alone, so it never changes,
whatever is added to or removed from the index.
"""

from __future__ import annotations

import hashlib
import re

from biosignal_data.index.rules import SubjectRule, SubjectRuleKind

SPLITS = (("train", 0.8), ("validation", 0.9), ("test", 1.0))  # upper bounds of [0, 1)


class SubjectError(ValueError):
    """A recording id does not match its dataset's subject pattern."""


def subject_of(
    dataset_id: str,
    recording_id: str,
    rule: SubjectRule | None,
    table: dict[str, str] | None = None,
) -> tuple[str, bool]:
    """The qualified subject id of a recording, and whether the source states it. `table`:
    for a table rule, the rule's table read as {recording column: subject column}."""
    if rule is None or rule.pattern is None:
        certain = rule is not None and rule.rule is SubjectRuleKind.ONE_PER_RECORDING
        return f"{dataset_id}:{recording_id}", certain
    match = re.search(rule.pattern, recording_id)
    if match is None:
        raise SubjectError(f"{dataset_id}: {recording_id!r} does not match {rule.pattern!r}")
    subject = match.group(1)
    if rule.rule is SubjectRuleKind.TABLE:
        if table is None:
            raise ValueError(f"{dataset_id}: a table rule needs its table")
        if subject not in table:
            raise SubjectError(f"{dataset_id}: {subject!r} is not in the subject table")
        subject = table[subject]
    for group in rule.same_person:
        if subject in group:
            subject = group[0]  # one id for one person
    return f"{rule.namespace or dataset_id}:{subject}", rule.certain


def split_of(subject_id: str) -> str:
    """train, validation or test (80/10/10), from the subject id alone."""
    digest = hashlib.sha256(subject_id.encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    return next(name for name, upper in SPLITS if position < upper)
