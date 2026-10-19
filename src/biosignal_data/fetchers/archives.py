"""General research-data repositories with public JSON APIs: Zenodo, figshare, DANDI,
Hugging Face. Each lists files with sizes and checksums; bytes are streamed over HTTPS."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from biosignal_data.catalog.models import (
    DandiSource,
    FigshareSource,
    HttpSource,
    HuggingFaceSource,
    ZenodoSource,
)
from biosignal_data.fetchers.base import (
    Fetcher,
    FetchError,
    RemoteFile,
    ResolvedVersion,
    http_opener,
    http_session,
)

# --- Zenodo -------------------------------------------------------------------------------


class ZenodoFetcher(Fetcher):
    API = "https://zenodo.org/api/records"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._record: dict[str, Any] | None = None

    @property
    def source(self) -> ZenodoSource:
        return self._source(ZenodoSource)

    def _get(self) -> dict[str, Any]:
        if self._record is None:
            headers = {}
            if token := os.environ.get("ZENODO_TOKEN"):  # only needed for restricted records
                headers["Authorization"] = f"Bearer {token}"
            response = http_session().get(
                f"{self.API}/{self.source.record}", headers=headers, timeout=60
            )
            response.raise_for_status()
            self._record = response.json()
        return self._record

    def resolve_version(self) -> ResolvedVersion:
        if self.options.version:
            return super().resolve_version()
        # A Zenodo record id is one immutable version; the id itself is the safest label.
        return ResolvedVersion(f"record-{self.source.record}", "source")

    def list_files(self) -> Iterator[RemoteFile]:
        record = self._get()
        files = record.get("files") or []
        if not files:
            raise FetchError(
                f"Zenodo record {self.source.record} lists no files (restricted? set ZENODO_TOKEN)"
            )
        session = http_session()
        for f in files:
            links = f.get("links", {})
            url = links.get("content") or links.get("self")
            checksum = f.get("checksum")  # "md5:<hex>"
            yield RemoteFile(
                path=f["key"],
                size=f.get("size"),
                checksum=checksum,
                url=url,
                opener=http_opener(session, url),
            )


# --- figshare -----------------------------------------------------------------------------


class FigshareFetcher(Fetcher):
    @property
    def source(self) -> FigshareSource:
        return self._source(FigshareSource)

    def _api(self, path: str) -> Any:
        response = http_session().get(f"{str(self.source.api).rstrip('/')}/{path}", timeout=60)
        response.raise_for_status()
        return response.json()

    def _article_files(self, article_id: int) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        page = 1
        while batch := self._api(f"articles/{article_id}/files?page={page}&page_size=1000"):
            files.extend(batch)
            page += 1
        return files

    def resolve_version(self) -> ResolvedVersion:
        if self.options.version:
            return super().resolve_version()
        if self.source.article is not None:
            meta = self._api(f"articles/{self.source.article}")
            return ResolvedVersion(f"v{meta.get('version', 1)}", "source")
        meta = self._api(f"collections/{self.source.collection}")
        return ResolvedVersion(f"v{meta.get('version', 1)}", "source")

    def list_files(self) -> Iterator[RemoteFile]:
        session = http_session()
        if self.source.article is not None:
            articles = [(self.source.article, "")]
        else:
            listing = self._api(f"collections/{self.source.collection}/articles?page_size=1000")
            articles = [(a["id"], f"{a['id']}/") for a in listing]
        for article_id, folder in articles:
            for f in self._article_files(article_id):
                md5 = f.get("computed_md5") or f.get("supplied_md5")
                yield RemoteFile(
                    path=f"{folder}{f['name']}",
                    size=f.get("size"),
                    checksum=f"md5:{md5}" if md5 else None,
                    url=f["download_url"],
                    opener=http_opener(session, f["download_url"]),
                )


# --- DANDI --------------------------------------------------------------------------------


def _dandi_client() -> Any:
    from dandi.dandiapi import DandiAPIClient  # noqa: PLC0415 - heavy import, only when used

    return DandiAPIClient(token=os.environ.get("DANDI_API_KEY"))


class DandiFetcher(Fetcher):
    @property
    def source(self) -> DandiSource:
        return self._source(DandiSource)

    def _dandiset(self, client: Any) -> Any:
        version = self.options.version or self.source.version
        dandiset = client.get_dandiset(self.source.dandiset, version)
        if version is None and dandiset.most_recent_published_version is not None:
            dandiset = client.get_dandiset(
                self.source.dandiset, dandiset.most_recent_published_version.identifier
            )
        return dandiset

    def resolve_version(self) -> ResolvedVersion:
        with _dandi_client() as client:
            return ResolvedVersion(str(self._dandiset(client).version_id), "source")

    def list_files(self) -> Iterator[RemoteFile]:
        session = http_session()
        with _dandi_client() as client:
            for asset in self._dandiset(client).get_assets():
                try:
                    digest = asset.get_raw_digest("dandi:sha2-256")
                    checksum = f"sha256:{digest}"
                except Exception:  # noqa: BLE001 - zarr assets carry other digest kinds
                    checksum = None
                url = asset.get_content_url(follow_redirects=1, strip_query=False)
                yield RemoteFile(
                    path=asset.path,
                    size=asset.size,
                    checksum=checksum,
                    url=url,
                    opener=http_opener(session, url),
                )


# --- Hugging Face -------------------------------------------------------------------------


def _hf() -> Any:
    import huggingface_hub  # noqa: PLC0415 - heavy import, only when used

    return huggingface_hub


class HuggingFaceFetcher(Fetcher):
    @property
    def source(self) -> HuggingFaceSource:
        return self._source(HuggingFaceSource)

    def resolve_version(self) -> ResolvedVersion:
        if self.options.version:
            return super().resolve_version()
        hf = _hf()
        info = hf.HfApi().repo_info(
            self.source.repo_id, repo_type=self.source.repo_type, revision=self.source.revision
        )
        # The commit hash is the only immutable identifier of a Hugging Face repo state.
        return ResolvedVersion(str(info.sha), "source")

    def list_files(self) -> Iterator[RemoteFile]:
        hf = _hf()
        revision = self.resolve_version().value
        token = hf.get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        session = http_session()
        tree = hf.HfApi().list_repo_tree(
            self.source.repo_id, repo_type=self.source.repo_type, revision=revision, recursive=True
        )
        for entry in tree:
            if not isinstance(entry, hf.hf_api.RepoFile):
                continue
            lfs = getattr(entry, "lfs", None)
            checksum = f"sha256:{lfs.sha256}" if lfs is not None else None
            url = hf.hf_hub_url(
                self.source.repo_id, entry.path, repo_type=self.source.repo_type, revision=revision
            )
            yield RemoteFile(
                path=entry.path,
                size=entry.size,
                checksum=checksum,
                url=url,
                opener=http_opener(session, url, headers),
            )


# --- plain HTTP ---------------------------------------------------------------------------


class HttpFetcher(Fetcher):
    @property
    def source(self) -> HttpSource:
        return self._source(HttpSource)

    def list_files(self) -> Iterator[RemoteFile]:
        session = http_session()
        for f in self.source.files:
            url = str(f.url)
            path = f.path or url.rstrip("/").rsplit("/", 1)[-1]
            yield RemoteFile(path=path, url=url, opener=http_opener(session, url))
