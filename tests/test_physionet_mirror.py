from __future__ import annotations

import hashlib
import pickle
from collections.abc import Iterator
from pathlib import Path

import boto3
import numpy as np
import pytest
from moto import mock_aws

from biosignal_data.fetchers import physionet_mirror
from biosignal_data.read.standard import StandardDataset
from biosignal_data.standard.convert import convert, plan_standard
from biosignal_data.storage.stores import LocalStore
from tests.conftest import make_dataset

ECG = (np.arange(200) % 50 - 25).astype("<i2")
FILES = {
    "RECORDS": b"r1\n",
    "r1.hea": b"r1 1 100 200\nr1.dat 16 200 16 0 0 0 0 ECG\n",
    "r1.dat": ECG.tobytes(),
}
DATASET = make_dataset(
    id="physionet/icentia11k-continuous-ecg",
    source={"type": "physionet", "slug": "icentia11k-continuous-ecg", "version": "1.0"},
)
PREFIX = "icentia11k-continuous-ecg/1.0/"


@pytest.fixture
def copy(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        monkeypatch.setattr(physionet_mirror, "s3_client", lambda **_: s3)
        s3.create_bucket(Bucket="physionet-open")
        sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in FILES.items())
        s3.put_object(Bucket="physionet-open", Key=PREFIX + "SHA256SUMS.txt", Body=sums)
        for name, body in FILES.items():
            s3.put_object(Bucket="physionet-open", Key=PREFIX + name, Body=body)
        yield s3


def test_a_version_on_the_aws_copy_converts_exactly_without_a_raw_version(
    copy: object, tmp_path: Path
) -> None:
    copy.put_object(Bucket="physionet-open", Key=PREFIX + "notes.txt", Body=b"x")  # type: ignore[attr-defined]
    raw = physionet_mirror.mirror(DATASET, "1.0", connections=2)
    dest = LocalStore(tmp_path / "bucket")
    plan = plan_standard(dest, DATASET.id, "1.0", raw_store=raw)
    assert convert(plan, dest, workers=1, part_bytes=1, work_dir=tmp_path / "work").complete
    ds = StandardDataset.open(dest, DATASET.id, "1.0")
    np.testing.assert_array_equal(ds.read("r1", "100Hz"), ECG[None, :])
    entries = {f.path: f for f in plan.raw.files}
    expected = hashlib.sha256(FILES["r1.dat"]).hexdigest()
    assert entries["r1.dat"].source_checksum == f"sha256:{expected}"
    assert entries["r1.dat"].md5 is None  # checked against SHA256SUMS.txt, not kept
    assert entries["notes.txt"].sha256 == hashlib.sha256(b"x").hexdigest()  # not listed: read
    assert ds.file("notes.txt") == b"x"
    assert not (tmp_path / "bucket" / "raw").exists()  # no raw version written


def test_a_file_sha256sums_lists_but_the_copy_lacks_is_refused(copy: object) -> None:
    copy.delete_object(Bucket="physionet-open", Key=PREFIX + "r1.dat")  # type: ignore[attr-defined]
    with pytest.raises(physionet_mirror.MirrorError, match=r"1 files of SHA256SUMS.txt.*r1\.dat"):
        physionet_mirror.mirror(DATASET, "1.0", connections=2)


def test_a_file_that_differs_from_sha256sums_fails_its_recording(
    copy: object, tmp_path: Path
) -> None:
    copy.put_object(Bucket="physionet-open", Key=PREFIX + "r1.dat", Body=ECG[::-1].tobytes())  # type: ignore[attr-defined]
    raw = physionet_mirror.mirror(DATASET, "1.0", connections=2)
    dest = LocalStore(tmp_path / "bucket")
    plan = plan_standard(dest, DATASET.id, "1.0", raw_store=raw)
    report = convert(plan, dest, workers=1, part_bytes=1, work_dir=tmp_path / "work")
    assert list(report.failed) == ["r1"] and "SHA-256" in report.failed["r1"]


def test_the_store_goes_to_a_conversion_process_without_its_client(copy: object) -> None:
    raw = physionet_mirror.mirror(DATASET, "1.0", connections=2)
    assert raw.process_safe
    raw.get_bytes(raw._data + "r1.hea")  # opens the client
    again = pickle.loads(pickle.dumps(raw))  # noqa: S301 - our own object, as spawn sends it
    assert again._client is None and again._manifest == raw._manifest
