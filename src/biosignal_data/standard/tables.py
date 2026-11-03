"""The Parquet tables of the standard layer: their schemas, and their rows for one recording.

Format-specific fields are kept as JSON text in the `source` columns, as the source wrote
them, so that nothing in a header is lost.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from biosignal_data.standard.blocks import BLOCKS
from biosignal_data.standard.sources import Channel, Event, Source

if TYPE_CHECKING:
    from biosignal_data.standard.rules import UnitCorrection

RECORDINGS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("format", pa.string()),
        ("start_time", pa.timestamp("us")),  # as the source wrote it, without a time zone
        ("duration_s", pa.float64()),
        ("arrays", pa.list_(pa.string())),  # array names in the Zarr group, e.g. "250Hz"
        ("n_channels", pa.int32()),
        ("signals", pa.bool_()),  # it has samples (in blocks.parquet); false for a recording
        # whose signals the source does not publish (only its annotations)
        ("source_files", pa.list_(pa.string())),  # raw files it was made from
        ("source_sha256", pa.list_(pa.string())),
        ("source_header", pa.string()),  # the header text, as written
    ]
)
CHANNELS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("array", pa.string()),
        ("index", pa.int32()),  # row in the array
        ("name", pa.string()),  # as the source wrote it
        ("standard_name", pa.string()),  # a mapping added later; empty for now
        ("unit", pa.string()),
        ("sampling_rate", pa.float64()),
        ("dtype", pa.string()),
        ("scale", pa.float64()),  # physical = stored * scale + offset
        ("offset", pa.float64()),
        ("invalid_value", pa.int64()),
        ("sha256", pa.string()),  # of the stored samples (little-endian), for exact checks
        ("source", pa.string()),
    ]
)
EVENTS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("file", pa.string()),
        ("annotator", pa.string()),
        ("onset_s", pa.float64()),
        ("duration_s", pa.float64()),
        ("label", pa.string()),
        ("text", pa.string()),
        ("channel", pa.string()),
        ("sample", pa.int64()),
        ("source", pa.string()),
    ]
)
SCHEMAS = {"recordings": RECORDINGS, "channels": CHANNELS, "events": EVENTS, "blocks": BLOCKS}


def recording_table(
    recording_id: str, source: Source, signals: bool, files: dict[str, str]
) -> pa.Table:
    """`files`: every raw file the recording was made from, with its SHA-256."""
    return pa.Table.from_pylist(
        [
            {
                "recording_id": recording_id,
                "format": source.format,
                "start_time": source.start_time,
                "duration_s": source.duration_s,
                "arrays": [g.name for g in source.groups],
                "n_channels": sum(len(g.channels) for g in source.groups),
                "signals": signals,
                "source_files": list(files),
                "source_sha256": list(files.values()),
                "source_header": source.header,
            }
        ],
        schema=RECORDINGS,
    )


def channel_table(
    recording_id: str,
    source: Source,
    sha256: dict[tuple[str, int], str],
    unit_corrections: Sequence[UnitCorrection] = (),
) -> pa.Table:
    """`sha256`: (array, index) -> the SHA-256 of the channel's stored samples.
    `unit_corrections`: the dataset's standard.yaml rules for units its headers give wrong."""
    rows = []
    for g in source.groups:
        for i, c in enumerate(g.channels):
            unit, fields = _unit(c, unit_corrections)
            rows.append(
                {
                    "recording_id": recording_id,
                    "array": g.name,
                    "index": i,
                    "name": c.name,
                    "standard_name": None,
                    "unit": unit,
                    "sampling_rate": g.rate,
                    "dtype": g.dtype.name,
                    "scale": c.scale,
                    "offset": c.offset,
                    "invalid_value": c.invalid_value,
                    "sha256": sha256.get((g.name, i)),
                    "source": _json(fields),
                }
            )
    return pa.Table.from_pylist(rows, schema=CHANNELS)


def _unit(channel: Channel, corrections: Sequence[UnitCorrection]) -> tuple[str, dict[str, Any]]:
    """The channel's unit and its source fields: the first rule for the channel gives the
    unit, and the unit as written goes into the source fields."""
    for rule in corrections:
        unit = rule.unit_of(channel.name, channel.unit)
        if unit is not None:
            return unit, channel.source | {"unit_as_written": channel.unit}
    return channel.unit, channel.source


def event_table(recording_id: str, events: list[Event]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "recording_id": recording_id,
                "file": e.file,
                "annotator": e.annotator,
                "onset_s": e.onset_s,
                "duration_s": e.duration_s,
                "label": e.label,
                "text": e.text,
                "channel": e.channel,
                "sample": e.sample,
                "source": _json(e.source) if e.source else None,
            }
            for e in events
        ],
        schema=EVENTS,
    )


def to_parquet(table: pa.Table) -> bytes:
    out = io.BytesIO()
    pq.write_table(table, out, compression="zstd")
    return out.getvalue()


def from_parquet(data: bytes, table: str) -> pa.Table:
    return pq.read_table(io.BytesIO(data), schema=SCHEMAS[table])


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)
