"""Fetchers against recorded API responses (no network)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
import responses
from moto import mock_aws

from biosignal_data.fetchers import (
    AccessRestrictedError,
    FetchError,
    FetchOptions,
    fetcher_for,
    physionet,
)
from biosignal_data.fetchers.base import LocalDirFetcher, RemoteFile
from biosignal_data.fetchers.physionet import PhysioNetFetcher, parse_sha256sums
from biosignal_data.ingest import ingest
from biosignal_data.storage.stores import LocalStore
from tests.conftest import make_dataset

SUMS = (
    "8d8a5349fb16638ebbf649f1779d12e96d91b736b2aafe59db43719ae583d471 100.atr\n"
    "b2ea3c250e56e48f4b7b90697832b8ecd1afa1e0bb31f2dcfea4ed6e1075a639  sub/100.dat\n"
)


def test_parse_sha256sums() -> None:
    assert list(parse_sha256sums(SUMS)) == [
        ("100.atr", "8d8a5349fb16638ebbf649f1779d12e96d91b736b2aafe59db43719ae583d471"),
        ("sub/100.dat", "b2ea3c250e56e48f4b7b90697832b8ecd1afa1e0bb31f2dcfea4ed6e1075a639"),
    ]
    with pytest.raises(FetchError):
        list(parse_sha256sums("nothex file\n"))


def _no_https(self: PhysioNetFetcher, version: str) -> Iterator[RemoteFile]:
    raise AssertionError("must not fall back to physionet.org")


@responses.activate
def test_physionet_lists_from_sha256sums_and_verifies(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib  # noqa: PLC0415

    body = b"header"
    digest = hashlib.sha256(body).hexdigest()
    base = "https://physionet.org/files/mitdb/1.0.0"
    responses.get(f"{base}/SHA256SUMS.txt", body=f"{digest} 100.hea\n")
    responses.get(f"{base}/100.hea", body=body)
    monkeypatch.setenv("BIOSIGNAL_PHYSIONET_USERNAME", "u")
    monkeypatch.setenv("BIOSIGNAL_PHYSIONET_PASSWORD", "p")
    dataset = make_dataset(  # credentialed: only on physionet.org
        id="physionet/mitdb",
        source={"type": "physionet", "slug": "mitdb", "version": "1.0.0", "credentialed": True},
    )
    report = ingest(dataset, store)
    assert report.complete
    assert store.get_bytes("raw/physionet/mitdb/1.0.0/data/100.hea") == body
    assert store.exists("raw/physionet/mitdb/1.0.0/data/SHA256SUMS.txt")


def test_physionet_downloads_from_aws_mirror(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib  # noqa: PLC0415

    files = {"100.hea": b"header", "100.dat": b"\x00" * 100}
    sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in files.items())
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        # The mock refuses unsigned requests; the real mirror allows them.
        monkeypatch.setattr(physionet, "s3_client", lambda **_: s3)
        monkeypatch.setattr(PhysioNetFetcher, "_https_files", _no_https)
        s3.create_bucket(Bucket="physionet-open")
        s3.put_object(Bucket="physionet-open", Key="mitdb/1.0.0/SHA256SUMS.txt", Body=sums)
        for name, body in files.items():
            s3.put_object(Bucket="physionet-open", Key=f"mitdb/1.0.0/{name}", Body=body)
        s3.put_object(Bucket="physionet-open", Key="mitdb/1.0.0/extra.html", Body=b"x")
        dataset = make_dataset(
            id="physionet/mitdb", source={"type": "physionet", "slug": "mitdb", "version": "1.0.0"}
        )
        listed = list(fetcher_for(dataset).list_files())
        assert {f.path: f.size for f in listed} == {
            "100.hea": 6,
            "100.dat": 100,
            "SHA256SUMS.txt": len(sums),
        }
        assert all(f.url and f.url.startswith("s3://physionet-open/mitdb/1.0.0/") for f in listed)
        report = ingest(dataset, store)
    assert report.complete
    assert store.get_bytes("raw/physionet/mitdb/1.0.0/data/100.dat") == files["100.dat"]


@responses.activate
def test_physionet_version_without_sha256sums_walks_the_folder_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIOSIGNAL_PHYSIONET_USERNAME", "u")
    monkeypatch.setenv("BIOSIGNAL_PHYSIONET_PASSWORD", "p")
    base = "https://physionet.org/files/old/1.0.0"
    responses.get(f"{base}/SHA256SUMS.txt", status=404)
    responses.get(
        f"{base}/",
        body='<a href="../">up</a> <a href="a.csv">a</a> <a href="training/">t</a>',
    )
    responses.get(f"{base}/training/", body='<a href="../">up</a> <a href="A%2001.mat">x</a>')
    dataset = make_dataset(  # credentialed: read from physionet.org's folder index
        id="physionet/old",
        source={"type": "physionet", "slug": "old", "version": "1.0.0", "credentialed": True},
    )
    files = list(fetcher_for(dataset).list_files())
    assert [(f.path, f.checksum) for f in files] == [("a.csv", None), ("training/A 01.mat", None)]
    assert files[1].url == f"{base}/training/A%2001.mat"


def test_physionet_open_version_missing_on_aws_copy_is_an_error(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        monkeypatch.setattr(physionet, "s3_client", lambda **_: s3)
        monkeypatch.setattr(PhysioNetFetcher, "_https_files", _no_https)
        s3.create_bucket(Bucket="physionet-open")
        dataset = make_dataset(
            id="physionet/x", source={"type": "physionet", "slug": "x", "version": "9.9"}
        )
        with pytest.raises(FetchError, match="not on PhysioNet's AWS copy"):
            ingest(dataset, store)


@responses.activate
def test_physionet_resolves_latest_version_from_redirect() -> None:
    responses.get(
        "https://physionet.org/content/ptb-xl/",
        status=302,
        headers={"Location": "https://physionet.org/content/ptb-xl/1.0.3/"},
    )
    responses.get("https://physionet.org/content/ptb-xl/1.0.3/", body="ok")
    dataset = make_dataset(id="physionet/ptb-xl", source={"type": "physionet", "slug": "ptb-xl"})
    assert fetcher_for(dataset).resolve_version().value == "1.0.3"


def test_physionet_credentialed_needs_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHYSIONET_USERNAME", raising=False)
    dataset = make_dataset(
        id="physionet/x",
        source={"type": "physionet", "slug": "x", "version": "1", "credentialed": True},
    )
    with pytest.raises(FetchError, match="PHYSIONET_USERNAME"):
        list(fetcher_for(dataset).list_files())


@responses.activate
def test_zenodo_lists_files_with_md5() -> None:
    responses.get(
        "https://zenodo.org/api/records/123",
        json={
            "files": [
                {
                    "key": "a.zip",
                    "size": 3,
                    "checksum": "md5:abc",
                    "links": {"self": "https://zenodo.org/api/records/123/files/a.zip/content"},
                }
            ]
        },
    )
    dataset = make_dataset(id="zenodo/123", source={"type": "zenodo", "record": 123})
    fetcher = fetcher_for(dataset)
    assert fetcher.resolve_version().value == "record-123"
    (f,) = fetcher.list_files()
    assert (f.path, f.size, f.checksum) == ("a.zip", 3, "md5:abc")


@responses.activate
def test_figshare_collection_prefixes_article_ids() -> None:
    api = "https://api.figshare.com/v2"
    collection = f"{api}/collections/9/articles"
    responses.get(f"{collection}?page=1&page_size=1000", json=[{"id": 1}, {"id": 2}])
    responses.get(f"{collection}?page=2&page_size=1000", json=[])  # every page is read
    for article in (1, 2):
        responses.get(
            f"{api}/articles/{article}/files?page=1&page_size=1000",
            json=[
                {
                    "name": "x.csv",
                    "size": 1,
                    "computed_md5": "ff",
                    "download_url": f"https://d/{article}",
                }
            ],
        )
        responses.get(f"{api}/articles/{article}/files?page=2&page_size=1000", json=[])
    dataset = make_dataset(id="figshare/9", source={"type": "figshare", "collection": 9})
    assert [f.path for f in fetcher_for(dataset).list_files()] == ["1/x.csv", "2/x.csv"]


@responses.activate
def test_figshare_embargo_is_restricted_access(store: LocalStore) -> None:
    api = "https://api.figshare.com/v2"
    responses.get(f"{api}/articles/5", json={"version": 3, "is_embargoed": True})
    responses.get(f"{api}/articles/5/files?page=1&page_size=1000", json=[])
    dataset = make_dataset(id="figshare/5", source={"type": "figshare", "article": 5})
    with pytest.raises(AccessRestrictedError, match="embargo"):
        ingest(dataset, store)


def test_figshare_requires_exactly_one_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        make_dataset(source={"type": "figshare"})


def test_s3_source_without_prefix_asks_for_it() -> None:
    dataset = make_dataset(
        source={"type": "s3", "bucket": "b", "region": "us-east-1", "anonymous": True}
    )
    with pytest.raises(FetchError, match=r"source\.prefix"):
        list(fetcher_for(dataset).list_files())


def test_s3_source_lists_relative_paths(store: LocalStore, monkeypatch: pytest.MonkeyPatch) -> None:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="src")
        client.put_object(Bucket="src", Key="PSG/a/1.edf", Body=b"one")
        client.put_object(Bucket="src", Key="PSG/", Body=b"")
        dataset = make_dataset(
            source={
                "type": "s3",
                "bucket": "src",
                "prefix": "PSG",
                "version": "3.0",
                "region": "us-east-1",
                "anonymous": False,
                "key_id_env": "TEST_KEY_ID",
                "secret_env": "TEST_SECRET",
            }
        )
        # The mock needs signed requests; the keys come from the catalog-named variables.
        monkeypatch.setenv("TEST_KEY_ID", "k")
        monkeypatch.setenv("TEST_SECRET", "s")
        report = ingest(dataset, store)
        assert report.complete
        assert store.get_bytes("raw/test/example/3.0/data/a/1.edf") == b"one"


def test_manual_source_points_to_import() -> None:
    dataset = make_dataset()
    with pytest.raises(FetchError, match="biosignal-data import"):
        fetcher_for(dataset).prepare()


def test_from_dir_ingests_a_local_folder(tmp_path: Path, store: LocalStore) -> None:
    (tmp_path / "in" / "sub").mkdir(parents=True)
    (tmp_path / "in" / "sub" / "x.edf").write_bytes(b"edf")
    dataset = make_dataset()
    options = FetchOptions(from_dir=tmp_path / "in", version="2026-01-01")
    assert isinstance(fetcher_for(dataset, options), LocalDirFetcher)
    report = ingest(dataset, store, options=options)
    assert report.complete
    assert store.get_bytes("raw/test/example/2026-01-01/data/sub/x.edf") == b"edf"


def test_http_source_paths_default_to_file_name() -> None:
    dataset = make_dataset(
        source={
            "type": "http",
            "files": [
                {"url": "https://example.org/a/b.tar"},
                {"url": "https://example.org/c", "path": "d/c"},
            ],
        }
    )
    assert [f.path for f in fetcher_for(dataset).list_files()] == ["b.tar", "d/c"]


def test_openneuro_old_snapshot_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from biosignal_data.fetchers import openneuro_nemar  # noqa: PLC0415

    monkeypatch.setattr(openneuro_nemar, "openneuro_latest_snapshot", lambda _: "2.0.0")
    dataset = make_dataset(
        id="openneuro/ds000001",
        source={"type": "openneuro", "accession": "ds000001", "snapshot": "1.0.0"},
    )
    with pytest.raises(FetchError, match="DataLad"):
        fetcher_for(dataset).resolve_version()


def test_rsync_needs_staging_and_the_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
    monkeypatch.setenv("TEST_SSH_KEY", str(tmp_path / "key"))
    dataset = make_dataset(
        source={
            "type": "rsync",
            "remote": "u@h:p",
            "version": "v1",
            "ssh_identity_env": "TEST_SSH_KEY",
        }
    )
    with pytest.raises(FetchError, match=r"staging\.dir"):
        fetcher_for(dataset).prepare()  # no staging folder configured
    with pytest.raises(FetchError, match="rsync"):
        fetcher_for(dataset, FetchOptions(staging_dir=tmp_path, keep_staging=True)).prepare()
