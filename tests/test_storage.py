from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from biosignal_data.storage.layout import UnsafePathError, VersionLocation, safe_relpath
from biosignal_data.storage.manifest import HashingReader
from biosignal_data.storage.stores import LocalStore, S3Store, open_store


@pytest.mark.parametrize("path", ["../x", "a/../../x", "", "sub/..", "/"])
def test_unsafe_paths_are_refused(path: str) -> None:
    with pytest.raises(UnsafePathError):
        safe_relpath(path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/sub-01\\eeg/x.edf", "sub-01/eeg/x.edf"),
        ("a//b/./c", "a/b/c"),
        ("./a", "a"),
    ],
)
def test_paths_are_normalised(path: str, expected: str) -> None:
    assert safe_relpath(path) == expected


def test_layout_keys() -> None:
    loc = VersionLocation("physionet/mitdb", "1.0.0")
    assert loc.data_key("100.hea") == "raw/physionet/mitdb/1.0.0/data/100.hea"
    assert loc.manifest_key == "raw/physionet/mitdb/1.0.0/_manifest.json"
    assert loc.success_key == "raw/physionet/mitdb/1.0.0/_SUCCESS"
    with pytest.raises(ValueError, match="version"):
        VersionLocation("physionet/mitdb", "1.0/../x")


def test_hashing_reader_counts_and_hashes() -> None:
    reader = HashingReader(io.BytesIO(b"abc" * 1000))
    while reader.read(7):
        pass
    assert reader.bytes_read == 3000
    assert reader.sha256 == hashlib.sha256(b"abc" * 1000).hexdigest()
    assert reader.md5 == hashlib.md5(b"abc" * 1000, usedforsecurity=False).hexdigest()


def test_local_store_refuses_escaping_keys(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        store.put_bytes("../outside", b"x")


def test_open_store_parses_uris(tmp_path: Path) -> None:
    assert isinstance(open_store(str(tmp_path)), LocalStore)
    assert isinstance(open_store(tmp_path.as_uri()), LocalStore)
    with pytest.raises(ValueError, match="unsupported"):
        open_store("gs://bucket")


@pytest.fixture
def s3_store() -> Iterator[S3Store]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="lake")
        yield S3Store("lake", "prefix", client=client, multipart_chunksize=5 * 1024 * 1024)


def test_s3_store_roundtrip_including_multipart(s3_store: S3Store) -> None:
    body = bytes(range(256)) * 50_000  # ~12.8 MB -> multipart with 5 MB parts
    reader = HashingReader(io.BytesIO(body))
    s3_store.put_stream("raw/x/1/data/big.bin", reader)  # type: ignore[arg-type]
    assert reader.bytes_read == len(body)
    assert s3_store.size("raw/x/1/data/big.bin") == len(body)
    assert s3_store.get_bytes("raw/x/1/data/big.bin") == body
    assert s3_store.size("raw/x/1/data/missing") is None
    assert s3_store.get_bytes("raw/x/1/data/missing") is None
    assert list(s3_store.list("raw/x/")) == ["raw/x/1/data/big.bin"]
    s3_store.delete("raw/x/1/data/big.bin")
    assert not s3_store.exists("raw/x/1/data/big.bin")
