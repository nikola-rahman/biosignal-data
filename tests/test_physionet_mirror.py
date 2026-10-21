from __future__ import annotations

import hashlib
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from biosignal_data.fetchers import physionet_mirror
from biosignal_data.read.standard import StandardDataset
from biosignal_data.standard.convert import HEADER_READS, convert, header_reads, plan_standard
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
    assert header_reads(raw) == 64 and header_reads(LocalStore(Path("x"))) == HEADER_READS
    raw.get_bytes(raw._data + "r1.hea")  # opens the client
    again = pickle.loads(pickle.dumps(raw))  # noqa: S301 - our own object, as spawn sends it
    assert again._client is None and again._manifest is None  # data files only
    assert again.get_bytes(again._data + "r1.hea") == FILES["r1.hea"]


class _AccessPoints:
    """PhysioNet's second bucket behind access points, for an AWS account linked to
    PhysioNet: in the mock, access point n is the bucket `ap-0n`; ap-01 refuses this
    account (an access point that PhysioNet replaced), ap-02 has the version."""

    def __init__(self, s3: Any) -> None:
        self._s3 = s3
        self.exceptions = s3.exceptions

    def get_object(self, **kw: Any) -> Any:
        if kw["Bucket"] == "ap-01":
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        return self._s3.get_object(**kw)

    def __getattr__(self, name: str) -> object:
        return getattr(self._s3, name)


def test_a_version_behind_an_access_point_is_read_with_the_linked_accounts_keys(
    copy: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s3: Any = copy
    s3.create_bucket(Bucket="ap-02")
    for key in [*FILES, "SHA256SUMS.txt"]:  # not in the public bucket: behind ap-02
        body = s3.get_object(Bucket="physionet-open", Key=PREFIX + key)["Body"].read()
        s3.put_object(Bucket="ap-02", Key=PREFIX + key, Body=body)
        s3.delete_object(Bucket="physionet-open", Key=PREFIX + key)
    clients: list[object] = []

    def client(**kw: object) -> object:
        clients.append(kw.get("keys"))
        return s3 if kw.get("keys") is None else _AccessPoints(s3)

    monkeypatch.setattr(physionet_mirror, "s3_client", client)
    monkeypatch.setattr(physionet_mirror, "access_point", lambda slug, version, n: f"ap-{n:02d}")
    with pytest.raises(physionet_mirror.MirrorError, match="no AWS keys given"):
        physionet_mirror.mirror(DATASET, "1.0", connections=2)
    raw = physionet_mirror.mirror(DATASET, "1.0", connections=2, keys=("id", "secret"))
    assert raw.uri == "s3://ap-02/icentia11k-continuous-ecg/1.0"
    dest = LocalStore(tmp_path / "bucket")
    plan = plan_standard(dest, DATASET.id, "1.0", raw_store=raw)
    assert convert(plan, dest, workers=1, part_bytes=1, work_dir=tmp_path / "work").complete
    np.testing.assert_array_equal(
        StandardDataset.open(dest, DATASET.id, "1.0").read("r1", "100Hz"), ECG[None, :]
    )
    assert ("id", "secret") in clients  # the store reads with the keys too
    entries = {f.path: f for f in plan.raw.files}
    assert entries["r1.dat"].source_url == f"s3://ap-02/{PREFIX}r1.dat"


def test_the_access_point_arn_follows_physionet_build() -> None:
    assert physionet_mirror.access_point("ltstdb", "1.0.0", 1) == (
        "arn:aws:s3:us-east-1:724665945834:accesspoint/ltstdb-v1-0-0-01"
    )
