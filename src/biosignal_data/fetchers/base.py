"""Fetcher interface.

A fetcher turns a catalog `source` into (1) a version label and (2) a list of files, each of
which can be opened as a byte stream. The pipeline streams every file into the store, so a
fetcher never needs local disk unless its upstream tool insists on writing files
(rsync, the NSRR gem, the Kaggle CLI, manual downloads). Those fetchers stage to a local
folder first and then behave like `LocalDirFetcher`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, TypeVar, cast

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from biosignal_data.catalog.models import Dataset
from biosignal_data.storage.layout import retrieval_version, safe_relpath

USER_AGENT = "biosignal-data (+https://github.com/nikola-rahman/biosignal-data)"
CHUNK = 8 * 1024 * 1024


class FetchError(RuntimeError):
    """The source could not be read. The message says what to do next where possible."""


class MissingCredentialsError(FetchError):
    """Credentials for a gated source are not set."""


class AccessRestrictedError(FetchError):
    """The source holds the files back (embargo, request needed) from this account."""


class UnavailableError(FetchError):
    """The dataset is known, but there is no way to get it (catalog source `unavailable`)."""


class SourceFileMissingError(FetchError):
    """The source lists this file, but answers that it does not exist (HTTP 404)."""


class ManualDownloadError(FetchError):
    """The source has no programmatic download; a person downloads it, then `--from-dir`."""


@dataclass(frozen=True)
class ResolvedVersion:
    value: str
    origin: Literal["source", "retrieval-date", "user"]


@dataclass(frozen=True)
class RemoteFile:
    """One file to ingest. `path` is relative to the dataset root as the source publishes it."""

    path: str
    opener: Callable[[], AbstractContextManager[BinaryIO]] = field(repr=False, compare=False)
    size: int | None = None
    checksum: str | None = None  # "<algorithm>:<hex>", e.g. "sha256:..." or "md5:..."
    url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", safe_relpath(self.path))

    def open(self) -> AbstractContextManager[BinaryIO]:
        return self.opener()


@dataclass
class FetchOptions:
    """Run-time settings that are not part of the catalog: from the config file (staging,
    connections) or from `import` (from_dir, version). A fetcher that needs a setting that
    is None stops with an error."""

    staging_dir: Path | None = None  # disk for sources whose tools write files first
    keep_staging: bool | None = None
    connections: int = 1  # parallel requests to one source: download.workers in the CLI
    from_dir: Path | None = None  # `import`: files downloaded by hand
    version: str | None = None  # `import`: the version label the user gives


S = TypeVar("S")


class Fetcher(ABC):
    # True for sources that cannot be asked for a version (plain HTTP, rsync, an S3 prefix):
    # a version in the catalog is then only a label for what the source serves now.
    version_is_label = False

    def __init__(self, dataset: Dataset, options: FetchOptions | None = None) -> None:
        self.dataset = dataset
        self.options = options or FetchOptions()

    def _source(self, kind: type[S]) -> S:
        """The catalog source, checked to be the type this fetcher reads."""
        source = self.dataset.source
        if not isinstance(source, kind):
            raise TypeError(f"{type(self).__name__} cannot read a {source.type!r} source")
        return source

    def resolve_version(self) -> ResolvedVersion:
        """Override when the source can report its own release."""
        if self.options.version:
            return ResolvedVersion(self.options.version, "user")
        pinned = self.dataset.pinned_version
        if pinned:
            return ResolvedVersion(pinned, "source")
        return ResolvedVersion(retrieval_version(), "retrieval-date")

    def check_access(self) -> None:  # noqa: B027 - optional hook
        """Fail fast, without network, if a person must act first: raise
        `MissingCredentialsError` or `ManualDownloadError`."""

    def prepare(self) -> None:  # noqa: B027 - optional hook
        """Run anything that must happen before listing (e.g. staging a download)."""

    def cleanup(self, complete: bool) -> None:  # noqa: B027 - optional hook
        """Remove what `prepare` created; `complete`: the version is now complete."""

    @abstractmethod
    def list_files(self) -> Iterator[RemoteFile]: ...


# --- helpers shared by HTTP-based fetchers --------------------------------------------------


def http_session(
    auth: tuple[str, str] | None = None, *, ipv4_only: bool = False
) -> requests.Session:
    """A session with retries. `ipv4_only` binds every connection to an IPv4 address, for
    sources that refuse requests over IPv6 (figshare's API answers them with 403)."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    if auth:
        session.auth = auth
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "POST"),
    )
    # One session serves all parallel downloads of a dataset: keep enough connections.
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=64)
    if ipv4_only:
        adapter = _IPv4Adapter(max_retries=retry, pool_connections=4, pool_maxsize=64)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class _IPv4Adapter(HTTPAdapter):
    """Connects over IPv4 only: a socket bound to 0.0.0.0 cannot reach an IPv6 address, so
    urllib3 skips a host's IPv6 addresses and uses its IPv4 ones."""

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        kwargs["source_address"] = ("0.0.0.0", 0)  # noqa: S104 - a bind for outgoing sockets
        super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy: str, **kwargs: Any) -> Any:
        kwargs["source_address"] = ("0.0.0.0", 0)  # noqa: S104 - as above
        return super().proxy_manager_for(proxy, **kwargs)


class _ResponseStream(AbstractContextManager[BinaryIO]):
    def __init__(self, session: requests.Session, url: str, headers: dict[str, str]) -> None:
        self._session = session
        self._url = url
        self._headers = headers
        self._response: requests.Response | None = None

    def __enter__(self) -> BinaryIO:
        response = self._session.get(
            self._url, stream=True, timeout=(10, 300), headers=self._headers
        )
        if response.status_code in (401, 403):
            response.close()
            raise MissingCredentialsError(
                f"{self._url} answered {response.status_code}: check access and credentials"
            )
        if response.status_code == 404:
            response.close()
            raise SourceFileMissingError(f"{self._url} answered 404 Not Found")
        response.raise_for_status()
        response.raw.decode_content = True
        self._response = response
        return cast(BinaryIO, response.raw)

    def __exit__(self, *exc: object) -> None:
        if self._response is not None:
            self._response.close()


def http_opener(
    session: requests.Session, url: str, headers: dict[str, str] | None = None
) -> Callable[[], AbstractContextManager[BinaryIO]]:
    return lambda: _ResponseStream(session, url, headers or {})


def require_env(*names: str) -> tuple[str, ...]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise MissingCredentialsError(
            f"set {', '.join(missing)} in the environment (see docs/access.md)"
        )
    return tuple(os.environ[n] for n in names)


class LocalDirFetcher(Fetcher):
    """Files already on local disk: a manual download or a staged upstream tool."""

    root: Path

    def __init__(
        self, dataset: Dataset, options: FetchOptions | None = None, root: Path | None = None
    ) -> None:
        super().__init__(dataset, options)
        chosen = root or self.options.from_dir
        if chosen is None:
            raise FetchError(
                f"{dataset.id} has no programmatic download; follow the steps in its docs "
                "page, then run `ingest --from-dir <folder>`"
            )
        self.root = chosen.expanduser().resolve()

    def list_files(self) -> Iterator[RemoteFile]:
        if not self.root.is_dir():
            raise FetchError(f"{self.root} is not a directory")
        for path in sorted(self.root.rglob("*")):
            target = path.resolve()  # follows symlinks (rsync -L and some tools leave them)
            if target.is_file():
                yield RemoteFile(
                    path=path.relative_to(self.root).as_posix(),
                    size=target.stat().st_size,
                    opener=_file_opener(target),
                )


def _file_opener(path: Path) -> Callable[[], AbstractContextManager[BinaryIO]]:
    return lambda: path.open("rb")
