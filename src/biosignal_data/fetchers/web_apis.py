"""NSRR (sleepdata.org) and Kaggle, read through their web APIs and streamed: no local
disk, no external tools.

NSRR: the same requests the official `nsrr` gem makes (github.com/nsrr/nsrr-gem):
  list      GET https://sleepdata.org/api/v1/datasets/<slug>/files.json?auth_token=&path=
  download  GET https://sleepdata.org/datasets/<slug>/files/a/<token>/m/<client>/<path>
The listing gives each file's size and MD5, which are checked after the copy.

Kaggle: the official Python SDK (`kagglesdk`, part of the `kaggle` package) returns the
download as a stream. Kaggle serves a dataset or a competition as one zip file; the raw
layer keeps it as Kaggle serves it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, BinaryIO, cast
from urllib.parse import quote

import requests

from biosignal_data.catalog.models import KaggleSource, NsrrSource
from biosignal_data.fetchers.base import (
    Fetcher,
    FetchError,
    MissingCredentialsError,
    RemoteFile,
    http_opener,
    http_session,
    require_env,
)

NSRR_TOKEN_VAR = "BIOSIGNAL_NSRR_TOKEN"  # noqa: S105 - a variable name
KAGGLE_TOKEN_VAR = "BIOSIGNAL_KAGGLE_API_TOKEN"  # noqa: S105 - a variable name

NSRR_WEBSITE = "https://sleepdata.org"
NSRR_CLIENT = "biosignal-data"  # the gem sends "nsrr-gem-v<version>" in this path segment


class NsrrFetcher(Fetcher):
    """The token is the one shown on https://sleepdata.org/token after the dataset's data
    request has been approved."""

    @property
    def source(self) -> NsrrSource:
        return self._source(NsrrSource)

    def check_access(self) -> None:
        require_env(NSRR_TOKEN_VAR)

    def list_files(self) -> Iterator[RemoteFile]:
        (token,) = require_env(NSRR_TOKEN_VAR)
        yield from self._walk(http_session(), token, self.source.subpath or "")

    def _walk(self, session: requests.Session, token: str, path: str) -> Iterator[RemoteFile]:
        slug = self.source.slug
        response = session.get(
            f"{NSRR_WEBSITE}/api/v1/datasets/{slug}/files.json",
            params={"auth_token": token, "path": path},
            timeout=60,
        )
        if response.status_code in (401, 403):
            raise MissingCredentialsError(
                f"{self.dataset.id}: sleepdata.org refused {NSRR_TOKEN_VAR}; is the data "
                "request for this dataset approved?"
            )
        response.raise_for_status()
        for entry in response.json() or []:
            full_path = str(entry["full_path"])
            if not entry.get("is_file"):
                yield from self._walk(session, token, full_path)
                continue
            md5 = entry.get("file_checksum_md5")
            download = (
                f"{NSRR_WEBSITE}/datasets/{slug}/files/a/{token}/m/{NSRR_CLIENT}/{quote(full_path)}"
            )
            yield RemoteFile(
                path=full_path,
                size=int(entry["file_size"]) if entry.get("file_size") is not None else None,
                checksum=f"md5:{md5}" if md5 else None,
                # The manifest keeps a URL without the token.
                url=f"{NSRR_WEBSITE}/datasets/{slug}/files/{quote(full_path)}",
                opener=http_opener(session, download),
            )


class KaggleFetcher(Fetcher):
    """Competitions also require accepting their rules on the website with the same account
    first; otherwise Kaggle refuses the download."""

    _response: Any = None
    _client: Any = None

    @property
    def source(self) -> KaggleSource:
        return self._source(KaggleSource)

    def check_access(self) -> None:
        require_env(KAGGLE_TOKEN_VAR)

    def list_files(self) -> Iterator[RemoteFile]:
        from kagglesdk import KaggleClient  # noqa: PLC0415 - heavy import, only when used

        (token,) = require_env(KAGGLE_TOKEN_VAR)
        # An explicit token: the SDK reads ~/.kaggle or KAGGLE_* only when none is given.
        self._client = KaggleClient(api_token=token)
        try:
            response, name = self._request(self._client)
        except requests.HTTPError as exc:
            raise FetchError(f"{self.dataset.id}: Kaggle refused the download: {exc}") from exc
        self._response = response
        length = response.headers.get("Content-Length")

        @contextmanager
        def opener() -> Iterator[BinaryIO]:
            # The response is already open and streams its body once.
            try:
                yield cast(BinaryIO, response.raw)
            finally:
                response.close()

        yield RemoteFile(
            path=name,
            size=int(length) if length else None,
            url=f"https://www.kaggle.com/{self._page()}",
            opener=opener,
        )

    def _request(self, client: Any) -> tuple[Any, str]:
        ref = self.source.ref
        if self.source.kind == "competition":
            from kagglesdk.competitions.types.competition_api_service import (  # noqa: PLC0415
                ApiDownloadDataFilesRequest,
            )

            request = ApiDownloadDataFilesRequest()
            request.competition_name = ref
            api = client.competitions.competition_api_client
            return api.download_data_files(request), f"{ref}.zip"
        from kagglesdk.datasets.types.dataset_api_service import (  # noqa: PLC0415
            ApiDownloadDatasetRequest,
        )

        owner, _, slug = ref.partition("/")
        request = ApiDownloadDatasetRequest()
        request.owner_slug, request.dataset_slug = owner, slug
        return client.datasets.dataset_api_client.download_dataset(request), f"{slug}.zip"

    def _page(self) -> str:
        if self.source.kind == "competition":
            return f"competitions/{self.source.ref}/data"
        return f"datasets/{self.source.ref}"

    def cleanup(self, complete: bool) -> None:
        if self._response is not None:
            self._response.close()
        if self._client is not None:
            self._client.__exit__(None, None, None)
