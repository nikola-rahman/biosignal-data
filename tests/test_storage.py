from __future__ import annotations

import base64
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
        yield S3Store("lake", "prefix", client=client, part_size=5 * 1024 * 1024)


def test_s3_store_roundtrip_including_multipart(s3_store: S3Store) -> None:
    body = bytes(range(256)) * 50_000  # ~12.8 MB -> multipart with 5 MB parts
    reader = HashingReader(io.BytesIO(body))
    s3_store.put_stream("raw/x/1/data/big.bin", reader, len(body))  # type: ignore[arg-type]
    assert reader.bytes_read == len(body)
    assert s3_store.size("raw/x/1/data/big.bin") == len(body)
    assert s3_store.get_bytes("raw/x/1/data/big.bin") == body
    assert s3_store.size("raw/x/1/data/missing") is None
    assert s3_store.get_bytes("raw/x/1/data/missing") is None
    assert list(s3_store.list("raw/x/")) == ["raw/x/1/data/big.bin"]
    s3_store.delete("raw/x/1/data/big.bin")
    assert not s3_store.exists("raw/x/1/data/big.bin")
    keys = [f"raw/x/1/data/{i}" for i in range(1005)]  # more than one request
    for key in keys:
        s3_store.put_bytes(key, b"x")
    s3_store.delete_many(keys)
    assert list(s3_store.list("raw/x/")) == []


@pytest.mark.parametrize("size", [None, 1, 3 * 1024 * 1024 * 1024 * 1024])
def test_s3_store_reads_the_stream_to_its_end_whatever_size_is_given(
    s3_store: S3Store, size: int | None
) -> None:
    # No size, a wrong small one, one that needs bigger parts: the whole stream is stored.
    body = bytes(range(256)) * 30_000  # ~7.7 MB: one part or two, by the part size
    reader = HashingReader(io.BytesIO(body))
    s3_store.put_stream("raw/x/1/data/f.bin", reader, size)  # type: ignore[arg-type]
    assert reader.bytes_read == len(body)
    assert s3_store.get_bytes("raw/x/1/data/f.bin") == body


def test_s3_settings_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from biosignal_data.fetchers.s3 import s3_client  # noqa: PLC0415
    from biosignal_data.storage.stores import S3Settings, open_store  # noqa: PLC0415

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://elsewhere.example")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-key-not-to-be-used")
    monkeypatch.delenv("BIOSIGNAL_ACCESS_KEY_ID", raising=False)
    with pytest.raises(ValueError, match="BIOSIGNAL_ACCESS_KEY_ID"):
        S3Settings.from_env("https://s3.hippius.com", "decentralized", "path")
    with pytest.raises(ValueError, match="needs"):
        open_store("s3://bucket")

    monkeypatch.setenv("BIOSIGNAL_ACCESS_KEY_ID", "hip_key")
    monkeypatch.setenv("BIOSIGNAL_SECRET_ACCESS_KEY", "secret")
    settings = S3Settings.from_env("https://s3.hippius.com", "decentralized", "path")
    client = settings.client(connections=4)
    assert client.meta.endpoint_url == "https://s3.hippius.com"
    assert client.meta.region_name == "decentralized"
    assert client.meta.config.s3 == {"addressing_style": "path"}  # type: ignore[attr-defined]
    credentials = client._request_signer._credentials  # type: ignore[attr-defined]
    assert credentials.access_key == "hip_key"
    assert "secret" not in repr(settings)

    # Sources on AWS ignore endpoint settings meant for the destination.
    assert s3_client(region="us-east-1", connections=1).meta.endpoint_url == (
        "https://s3.amazonaws.com"
    )


def test_unfinished_uploads_are_found_and_discarded(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    (tmp_path / "raw" / "a").mkdir(parents=True)
    (tmp_path / "raw" / "a" / ".x.edf.123.part").write_bytes(b"12345")
    (tmp_path / "raw" / "a" / "y.edf").write_bytes(b"done")
    left = store.unfinished("raw")
    assert [(u.key, u.size) for u in left] == [("raw/a/.x.edf.123.part", 5)]
    store.discard(left[0])
    assert store.unfinished("raw") == () and store.exists("raw/a/y.edf")


def test_s3_unfinished_multipart_uploads() -> None:
    from biosignal_data.storage.stores import S3Store  # noqa: PLC0415

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="bucket-x")
        upload = client.create_multipart_upload(Bucket="bucket-x", Key="pre/raw/x/big.edf")
        client.upload_part(
            Bucket="bucket-x",
            Key="pre/raw/x/big.edf",
            UploadId=upload["UploadId"],
            PartNumber=1,
            Body=b"z" * 1000,
        )
        store = S3Store("bucket-x", "pre", client=client)
        left = store.unfinished("raw")
        assert [(u.key, u.size) for u in left] == [("raw/x/big.edf", 1000)]
        store.discard(left[0])
        assert store.unfinished("raw") == ()


def test_s3_store_put_bytes_sends_the_md5(s3_store: S3Store) -> None:
    """The storage checks what it received against Content-MD5 (BadDigest if it differs)."""
    sent: list[str] = []
    s3_store.client.meta.events.register(
        "before-send.s3.PutObject",
        lambda request, **_: sent.append(request.headers["Content-MD5"].decode()),
    )
    s3_store.put_bytes("a/b", b"payload")
    assert sent == [base64.b64encode(hashlib.md5(b"payload").digest()).decode()]  # noqa: S324
    assert s3_store.get_bytes("a/b") == b"payload"
