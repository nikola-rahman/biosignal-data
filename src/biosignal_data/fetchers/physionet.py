"""PhysioNet (https://physionet.org).

Every published PhysioNet version ships a `SHA256SUMS.txt` listing all of its files with
their SHA-256, so it is used as the file list and every file is verified against it.

One source per kind of dataset, never a switch between them:

- Open datasets come only from PhysioNet's own copy on AWS
  (`s3://physionet-open/<slug>/<version>/`, AWS Open Data program, no credentials): faster
  than physionet.org, and it gives file sizes. A version or a file that is not on it is an
  error.
- Credentialed datasets are not on that copy; they come only from physionet.org, with the
  user's PhysioNet account (BIOSIGNAL_PHYSIONET_USERNAME / _PASSWORD) after the DUA is signed.

Old versions publish no SHA256SUMS.txt; their file list then comes from the AWS listing or
the physionet.org folder index, without checksums, with a warning.

The latest version (for entries that pin none) is read from physionet.org's redirect.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote

import requests
from botocore.exceptions import BotoCoreError, ClientError

from biosignal_data.catalog.models import PhysioNetSource
from biosignal_data.fetchers.base import (
    Fetcher,
    FetchError,
    MissingCredentialsError,
    RemoteFile,
    ResolvedVersion,
    http_opener,
    http_session,
    require_env,
)
from biosignal_data.fetchers.s3 import s3_client, s3_opener

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

log = logging.getLogger(__name__)

BASE = "https://physionet.org"
SUMS = "SHA256SUMS.txt"
MIRROR_BUCKET = "physionet-open"
MIRROR_REGION = "us-east-1"
CREDENTIAL_VARS = ("BIOSIGNAL_PHYSIONET_USERNAME", "BIOSIGNAL_PHYSIONET_PASSWORD")
_LATEST_RE = re.compile(r"/content/(?P<slug>[^/]+)/(?P<version>[^/]+)/?$")


def parse_sha256sums(text: str) -> Iterator[tuple[str, str]]:
    """Yield (relative_path, sha256) from a SHA256SUMS.txt body."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition(" ")
        name = name.lstrip(" *")  # `sha256sum` writes "<hash>  <name>" or "<hash> *<name>"
        if len(digest) != 64 or not name:
            raise FetchError(f"unexpected line in {SUMS}: {line!r}")
        yield name, digest.lower()


class PhysioNetFetcher(Fetcher):
    @property
    def source(self) -> PhysioNetSource:
        return self._source(PhysioNetSource)

    def _session(self) -> requests.Session:
        auth: tuple[str, str] | None = None
        if self.source.credentialed:
            user, password = require_env(*CREDENTIAL_VARS)
            auth = (user, password)
        return http_session(auth)

    def check_access(self) -> None:
        if self.source.credentialed:
            require_env(*CREDENTIAL_VARS)

    def resolve_version(self) -> ResolvedVersion:
        if self.options.version or self.source.version:
            return super().resolve_version()
        # /content/<slug>/ redirects to the latest published version.
        response = self._session().get(f"{BASE}/content/{self.source.slug}/", timeout=30)
        response.raise_for_status()
        match = _LATEST_RE.search(response.url)
        if not match:
            raise FetchError(
                f"could not work out the latest version of {self.source.slug} from "
                f"{response.url}; pin `version` in the catalog entry"
            )
        return ResolvedVersion(match["version"], "source")

    def list_files(self) -> Iterator[RemoteFile]:
        version = self.resolve_version().value
        if self.source.credentialed:
            yield from self._https_files(version)
        else:
            yield from self._mirror_files(version)

    def _mirror_files(self, version: str) -> list[RemoteFile]:
        """The files of this version on PhysioNet's AWS copy, the only source of open data."""
        prefix = f"{self.source.slug}/{version}/"
        where = f"s3://{MIRROR_BUCKET}/{prefix}"
        try:
            client = s3_client(region=MIRROR_REGION, connections=self.options.connections)
            sizes = _list_sizes(client, prefix)
            if not sizes:
                raise FetchError(
                    f"{self.dataset.id} {version} is not on PhysioNet's AWS copy ({where}), "
                    "the only source for open PhysioNet data (docs/access.md)"
                )
            if SUMS not in sizes:
                return self._mirror_files_without_sums(client, prefix, sizes)
            sums = client.get_object(Bucket=MIRROR_BUCKET, Key=prefix + SUMS)["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise FetchError(f"{self.dataset.id}: reading {where} failed: {exc}") from exc
        listed = list(parse_sha256sums(sums.decode("utf-8")))
        missing = [name for name, _ in listed if name not in sizes]
        if missing:
            raise FetchError(
                f"{self.dataset.id} {version}: {len(missing)} files of {SUMS} are not on "
                f"PhysioNet's AWS copy ({where}), e.g. {missing[0]}"
            )
        files = [
            RemoteFile(
                path=name,
                size=sizes[name],
                checksum=f"sha256:{digest}",
                url=f"{where}{name}",
                opener=s3_opener(client, MIRROR_BUCKET, prefix + name),
            )
            for name, digest in listed
        ]
        # The checksum list is itself part of the published version; keep it.
        files.append(
            RemoteFile(
                path=SUMS,
                size=len(sums),
                url=f"{where}{SUMS}",
                opener=s3_opener(client, MIRROR_BUCKET, prefix + SUMS),
            )
        )
        return files

    def _mirror_files_without_sums(
        self, client: S3Client, prefix: str, sizes: dict[str, int]
    ) -> list[RemoteFile]:
        """An old version without SHA256SUMS.txt: the mirror's object listing is the file
        list, with sizes but no checksums."""
        self._warn_no_sums("the AWS mirror's object listing")
        return [
            RemoteFile(
                path=name,
                size=size,
                url=f"s3://{MIRROR_BUCKET}/{prefix}{name}",
                opener=s3_opener(client, MIRROR_BUCKET, prefix + name),
            )
            for name, size in sorted(sizes.items())
        ]

    def _warn_no_sums(self, source: str) -> None:
        log.warning(
            "%s: this version publishes no %s; the file list comes from %s, and the files "
            "cannot be checked against published checksums",
            self.dataset.id,
            SUMS,
            source,
        )

    def _crawl(self, session: requests.Session, base: str, folder: str) -> Iterator[RemoteFile]:
        """Walk the physionet.org folder index, as PhysioNet's documented `wget -r` does."""
        response = session.get(f"{base}/{folder}", timeout=60)
        response.raise_for_status()
        for link in _index_links(response.text):
            path = folder + link
            if link.endswith("/"):
                yield from self._crawl(session, base, path)
            else:
                url = f"{base}/{quote(path)}"
                yield RemoteFile(path=path, url=url, opener=http_opener(session, url))

    def _https_files(self, version: str) -> Iterator[RemoteFile]:
        session = self._session()
        base = f"{BASE}/files/{self.source.slug}/{version}"
        response = session.get(f"{base}/{SUMS}", timeout=60)
        if response.status_code in (401, 403):
            raise MissingCredentialsError(
                f"{self.dataset.id}: access denied. Sign the data use agreement on "
                f"{BASE}/content/{self.source.slug}/ and set {' and '.join(CREDENTIAL_VARS)}"
            )
        if response.status_code == 404:
            self._warn_no_sums("the folder index on physionet.org")
            yield from self._crawl(session, base, "")
            return
        response.raise_for_status()
        sums_body = response.content
        for name, digest in parse_sha256sums(sums_body.decode("utf-8")):
            url = f"{base}/{name}"
            yield RemoteFile(
                path=name, checksum=f"sha256:{digest}", url=url, opener=http_opener(session, url)
            )
        # The checksum list is itself part of the published version; keep it.
        sums_url = f"{base}/{SUMS}"
        yield RemoteFile(
            path=SUMS, size=len(sums_body), url=sums_url, opener=http_opener(session, sums_url)
        )


_HREF = re.compile(r'href="([^"?#]+)"')


def _index_links(html: str) -> list[str]:
    """Links of a physionet.org folder index: files, and folders ending with "/"."""
    return [
        unquote(h)
        for h in _HREF.findall(html)
        if not h.startswith(("/", "..", "http:", "https:", "mailto:"))
    ]


def _list_sizes(client: S3Client, prefix: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=MIRROR_BUCKET, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            sizes[item["Key"][len(prefix) :]] = int(item["Size"])
    return sizes
