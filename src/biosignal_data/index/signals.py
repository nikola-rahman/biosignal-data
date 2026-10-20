"""The signal type of a channel, from its name as the source wrote it.

General rules first (electrode and lead names, common abbreviations, the Italian terms of
capslpdb); a dataset's `channels` rules in index.yaml go before them, for names that mean
something only in that dataset. A name no rule matches is "unknown", never a guess: the
index lists those names, so that each gets a rule on purpose.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum


class SignalType(StrEnum):
    ECG = "ecg"
    EEG = "eeg"
    EOG = "eog"
    EMG = "emg"
    RESP = "resp"  # breathing effort or airflow
    SPO2 = "spo2"
    PPG = "ppg"
    BP = "bp"
    HR = "hr"  # heart rate as a signal, computed by the device
    TEMP = "temp"
    POSITION = "position"
    ACCEL = "accel"
    SCG = "scg"  # seismocardiogram
    PCG = "pcg"  # heart sounds
    SOUND = "sound"  # microphone, snoring
    OTHER = "other"  # device status, markers
    UNKNOWN = "unknown"


_ELECTRODE = r"(?:FP|AF|FC|FT|CP|TP|PO|F|C|T|P|O|A|M)(?:Z|\d{1,2})"
# Order matters: EOG before EEG (LOC-A1 is EOG); EMG before ECG; RESP before the rest.
_RULES: tuple[tuple[str, SignalType], ...] = (
    (r"^(EOG|LOC|ROC|E1|E2)\b", SignalType.EOG),
    (r"^(EMG|CHIN|TIB|LEG|DELTOIDE|MILO)|^(DX|SX)\d", SignalType.EMG),
    (rf"^(EEG\s*)?{_ELECTRODE}(\s*[-/]?\s*{_ELECTRODE})?$", SignalType.EEG),
    (r"^(ECG|EKG)|^(I|II|III|AVR|AVL|AVF|V[1-6]|MLI{1,3}|ML\s?I{1,3})$", SignalType.ECG),
    (
        r"^(RESP|THORAX|TORACE|TORACICO|ABDO|ADDOME|ADDDOME|FLOW|FLUSSO|AIRFLOW|NASAL|"
        r"CANN?ULA|TERMISTORE|THERMISTOR|FLATTENING)",
        SignalType.RESP,
    ),
    (r"^(SPO2|SAO2|SPO_2)$|^OX$", SignalType.SPO2),
    (r"^(PLETH|PPG)", SignalType.PPG),
    (r"^(NIBP|ABP|ART|BP)$", SignalType.BP),
    (r"^(HR|HEART RATE|PULSE)", SignalType.HR),
    (r"^TEMP", SignalType.TEMP),
    (r"^(POSITION|POSIZIONE|POS)$", SignalType.POSITION),
    (r"^ACC[XYZ]?$", SignalType.ACCEL),
    (r"^SCG$", SignalType.SCG),
    (r"^PCG", SignalType.PCG),
    (r"^(MIC|SOUND|SNORE)", SignalType.SOUND),
    (r"^(STAT|OX STATUS|TAG)$", SignalType.OTHER),
)
_COMPILED = tuple((re.compile(p), t) for p, t in _RULES)


def signal_type(
    name: str, overrides: Sequence[tuple[re.Pattern[str], SignalType]] = ()
) -> tuple[SignalType, str]:
    """The type of a channel and what decided it: "dataset" (an index.yaml rule of the
    dataset), "name" (a general rule) or "none" (unknown)."""
    for pattern, kind in overrides:
        if pattern.search(name):
            return kind, "dataset"
    key = " ".join(name.upper().split())
    for pattern, kind in _COMPILED:
        if pattern.search(key):
            return kind, "name"
    return SignalType.UNKNOWN, "none"
