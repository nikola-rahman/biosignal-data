"""OpenNeuro and NEMAR, each through its own official route.

OpenNeuro (https://openneuro.org): the public S3 bucket `s3://openneuro.org/<accession>/`
holds the latest snapshot of every dataset; the snapshot tag comes from OpenNeuro's GraphQL
API. The bucket publishes no checksums, so integrity rests on the SHA-256 we compute while
copying. Older snapshots are only available through DataLad; asking for one fails clearly.

NEMAR (https://nemar.org): the official `nemar-py` client resolves a version-pinned
manifest with per-file SHA-256/MD5; our pipeline streams the bytes and verifies them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any, BinaryIO

import requests

from biosignal_data.catalog.models import NemarSource, OpenNeuroSource
from biosignal_data.fetchers.base import (
    Fetcher,
    FetchError,
    RemoteFile,
    ResolvedVersion,
    http_opener,
    http_session,
)
from biosignal_data.fetchers.s3 import iter_s3_files, s3_client

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
OPENNEURO_BUCKET = "openneuro.org"


def openneuro_latest_snapshot(accession: str) -> str:
    query = {"query": f'query {{ dataset(id: "{accession}") {{ latestSnapshot {{ tag }} }} }}'}
    response = http_session().post(OPENNEURO_GRAPHQL, json=query, timeout=30)
    response.raise_for_status()
    try:
        return str(response.json()["data"]["dataset"]["latestSnapshot"]["tag"])
    except (KeyError, TypeError) as exc:
        raise FetchError(f"OpenNeuro returned no snapshot for {accession}") from exc


class OpenNeuroFetcher(Fetcher):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._latest: str | None = None

    @property
    def source(self) -> OpenNeuroSource:
        return self._source(OpenNeuroSource)

    def resolve_version(self) -> ResolvedVersion:
        if self._latest is None:
            self._latest = openneuro_latest_snapshot(self.source.accession)
        wanted = self.options.version or self.source.snapshot
        if wanted and wanted != self._latest:
            raise FetchError(
                f"{self.dataset.id}: snapshot {wanted} requested, but OpenNeuro's S3 bucket "
                f"serves only the latest ({self._latest}). Get {wanted} with DataLad, then "
                "ingest it with --from-dir and --version."
            )
        return ResolvedVersion(self._latest, "source")

    def list_files(self) -> Iterator[RemoteFile]:
        version = self.resolve_version().value
        client = s3_client(region="us-east-1", connections=self.options.connections)
        files = list(iter_s3_files(client, OPENNEURO_BUCKET, f"{self.source.accession}/"))
        # The bucket only tracks "latest": refuse to label files with a tag that moved on
        # while we were listing.
        if openneuro_latest_snapshot(self.source.accession) != version:
            raise FetchError(f"{self.dataset.id} got a new snapshot during listing; re-run")
        yield from files


def _nemar() -> Any:
    import nemar  # noqa: PLC0415 - heavy import, only when a NEMAR dataset is fetched

    return nemar


class _FallbackStream(AbstractContextManager[BinaryIO]):
    """Open `url`; if it is refused (expired signature), retry once with `bytes_url`."""

    def __init__(self, session: requests.Session, url: str, fallback: str | None) -> None:
        self._primary = http_opener(session, url)
        self._fallback = http_opener(session, fallback) if fallback else None
        self._active: AbstractContextManager[BinaryIO] | None = None

    def __enter__(self) -> BinaryIO:
        try:
            self._active = self._primary()
            return self._active.__enter__()
        except FetchError:
            if self._fallback is None:
                raise
            self._active = self._fallback()
            return self._active.__enter__()

    def __exit__(self, *exc: Any) -> None:
        if self._active is not None:
            self._active.__exit__(*exc)


def _fallback_opener(
    session: requests.Session, url: str, fallback: str | None
) -> Callable[[], AbstractContextManager[BinaryIO]]:
    return lambda: _FallbackStream(session, url, fallback)


class NemarFetcher(Fetcher):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._manifest: Any = None
        self._version: ResolvedVersion | None = None

    @property
    def source(self) -> NemarSource:
        return self._source(NemarSource)

    def _load(self) -> ResolvedVersion:
        if self._version is not None:
            return self._version
        nemar = _nemar()
        try:
            with nemar.NEMARClient() as client:
                index = client.fetch_index(self.source.accession)
                version = index.resolve_version(self.options.version or self.source.version)
                self._manifest = client.fetch_manifest(index, version)
        except nemar.NemarError as exc:
            raise FetchError(f"{self.dataset.id}: {exc}") from exc
        self._version = ResolvedVersion(str(version.version), "source")
        return self._version

    def resolve_version(self) -> ResolvedVersion:
        return self._load()

    def list_files(self) -> Iterator[RemoteFile]:
        self._load()
        session = http_session()
        for item in self._manifest:
            if item.sha256:
                checksum: str | None = f"sha256:{item.sha256}"
            elif item.md5:
                checksum = f"md5:{item.md5}"
            else:
                checksum = None
            yield RemoteFile(
                path=item.path,
                size=item.size,
                checksum=checksum,
                url=item.url,
                opener=_fallback_opener(session, item.url, item.bytes_url),
            )
