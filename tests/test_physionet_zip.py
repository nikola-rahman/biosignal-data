from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import boto3
import numpy as np
import pytest
from moto import mock_aws

from biosignal_data.fetchers import physionet_zip
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
    id="physionet/mitdb", source={"type": "physionet", "slug": "mitdb", "version": "1.0.0"}
)


def _zip(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in files.items():  # below one folder, as PhysioNet makes them
            z.writestr(f"mit-bih-arrhythmia-database-1.0.0/{name}", body)
    return out.getvalue()


@pytest.fixture
def mirror(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        # The mock refuses unsigned requests; the real mirror allows them.
        monkeypatch.setattr(physionet_zip, "s3_client", lambda **_: s3)
        s3.create_bucket(Bucket="physionet-open")
        sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in FILES.items())
        s3.put_object(Bucket="physionet-open", Key="mitdb/1.0.0/SHA256SUMS.txt", Body=sums)
        yield s3


def test_a_staged_zip_converts_exactly_and_keeps_the_raw_manifest(
    mirror: object, tmp_path: Path
) -> None:
    mirror.put_object(  # type: ignore[attr-defined]
        Bucket="physionet-open", Key="mitdb/mitdb-1.0.0.zip", Body=_zip(FILES)
    )
    staged = physionet_zip.stage(DATASET, "1.0.0", tmp_path / "work", connections=2)
    assert sorted(f.path for f in staged.manifest.files) == ["RECORDS", "r1.dat", "r1.hea"]
    assert all(f.source_checksum for f in staged.manifest.files)
    assert not (staged.folder / "archive.zip").exists()  # only the extracted, checked tree

    dest = LocalStore(tmp_path / "bucket")
    plan = plan_standard(dest, "physionet/mitdb", "1.0.0", raw_store=staged.store)
    assert convert(plan, dest, workers=1, part_bytes=1, work_dir=tmp_path / "convert").complete
    ds = StandardDataset.open(dest, "physionet/mitdb", "1.0.0")
    np.testing.assert_array_equal(ds.read("r1", "100Hz"), ECG[None, :])
    raw_manifest = dest.get_bytes(ds.location.raw_manifest_key) or b""
    assert hashlib.sha256(raw_manifest).hexdigest() == ds.manifest.raw_manifest_sha256
    assert {f["path"] for f in json.loads(raw_manifest)["files"]} == set(FILES)


def test_files_added_after_the_zip_come_from_the_version_folder(
    mirror: object, tmp_path: Path
) -> None:
    mirror.put_object(  # type: ignore[attr-defined]
        Bucket="physionet-open", Key="mitdb/mitdb-1.0.0.zip", Body=_zip(FILES)
    )
    mirror.put_object(  # type: ignore[attr-defined]
        Bucket="physionet-open", Key="mitdb/1.0.0/r1.apn", Body=b"answers"
    )  # added later: on the copy, not in the ZIP, not in SHA256SUMS.txt
    staged = physionet_zip.stage(DATASET, "1.0.0", tmp_path / "work", connections=2)
    added = next(f for f in staged.manifest.files if f.path == "r1.apn")
    assert added.size == 7 and added.source_checksum is None


def test_an_incomplete_zip_is_refused(mirror: object, tmp_path: Path) -> None:
    mirror.put_object(  # type: ignore[attr-defined]
        Bucket="physionet-open", Key="mitdb/mitdb-1.0.0.zip",
        Body=_zip({k: v for k, v in FILES.items() if k != "r1.dat"}),
    )  # fmt: skip
    with pytest.raises(physionet_zip.ZipCheckError, match=r"incomplete.*r1\.dat"):
        physionet_zip.stage(DATASET, "1.0.0", tmp_path / "work", connections=2)


def test_a_member_that_differs_from_sha256sums_is_refused(mirror: object, tmp_path: Path) -> None:
    mirror.put_object(  # type: ignore[attr-defined]
        Bucket="physionet-open", Key="mitdb/mitdb-1.0.0.zip",
        Body=_zip(FILES | {"r1.dat": ECG[::-1].tobytes()}),
    )  # fmt: skip
    with pytest.raises(physionet_zip.ZipCheckError, match=r"r1\.dat differs"):
        physionet_zip.stage(DATASET, "1.0.0", tmp_path / "work", connections=2)


def test_the_etag_of_a_multipart_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(physionet_zip, "PART", 4)
    path = tmp_path / "f"
    path.write_bytes(b"0123456789")  # parts 0123 | 4567 | 89
    parts = [hashlib.md5(p).digest() for p in (b"0123", b"4567", b"89")]  # noqa: S324
    assert physionet_zip._etag(path) == hashlib.md5(b"".join(parts)).hexdigest() + "-3"  # noqa: S324
    path.write_bytes(b"012")
    assert physionet_zip._etag(path) == hashlib.md5(b"012").hexdigest()  # noqa: S324
