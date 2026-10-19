"""PhysioNet (https://physionet.org).

Every published PhysioNet version ships a `SHA256SUMS.txt` listing all of its files with
their SHA-256, so it is used as the file list and every file is verified against it.

Open datasets are downloaded from PhysioNet's official AWS mirror
(`s3://physionet-open/<slug>/<version>/`, AWS Open Data program, no credentials), which is
much faster than physionet.org and also gives file sizes. A dataset or file that is not on
the mirror is downloaded from physionet.org over HTTPS. The file list and checksums come from
`SHA256SUMS.txt` either way, so both routes copy the same files.

Credentialed projects use HTTPS with basic auth with the user's PhysioNet account
(PHYSIONET_USERNAME / PHYSIONET_PASSWORD) after the DUA has been signed on the website.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

import requests
from botocore.exceptions import BotoCoreError, ClientError

from biosignal_data.catalog.models import PhysioNetSource
from biosignal_data.fetchers.base import (
    Fetcher,
    FetchError,
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

    def _route(self) -> str:
        """Where to read from: credentialed data is only on physionet.org; open data as
        `download.physionet_source` in the config file says."""
        if self.source.credentialed:
            return "physionet.org"
        if self.options.physionet_source is None:
            raise FetchError("set download.physionet_source in the config file")
        return self.options.physionet_source

    def check_access(self) -> None:
        if self.source.credentialed:
            require_env(*CREDENTIAL_VARS)
        self._route()

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
        route = self._route()
        if route == "physionet.org":
            yield from self._https_files(version)
            return
        mirrored = self._mirror_files(version, website_for_missing=route != "aws-mirror")
        if mirrored is not None:
            yield from mirrored
        elif route == "aws-mirror":
            raise FetchError(
                f"{self.dataset.id} {version} is not on PhysioNet's AWS mirror, and "
                "download.physionet_source is aws-mirror"
            )
        else:
            log.warning(
                "%s: not on PhysioNet's AWS mirror; reading physionet.org "
                "(download.physionet_source: aws-mirror-else-physionet.org)",
                self.dataset.id,
            )
            yield from self._https_files(version)

    def _mirror_files(self, version: str, website_for_missing: bool) -> list[RemoteFile] | None:
        """The files from the AWS mirror, or None if this version is not on it."""
        prefix = f"{self.source.slug}/{version}/"
        try:
            client = s3_client(region=MIRROR_REGION, connections=self.options.connections)
            sums = client.get_object(Bucket=MIRROR_BUCKET, Key=prefix + SUMS)["Body"].read()
            sizes = _list_sizes(client, prefix)
        except (BotoCoreError, ClientError) as exc:
            log.debug("%s: mirror lookup failed: %s", self.dataset.id, exc)
            return None
        session = self._session()
        base = f"{BASE}/files/{self.source.slug}/{version}"
        files: list[RemoteFile] = []
        missing = 0
        for name, digest in parse_sha256sums(sums.decode("utf-8")):
            if name in sizes:
                files.append(
                    RemoteFile(
                        path=name,
                        size=sizes[name],
                        checksum=f"sha256:{digest}",
                        url=f"s3://{MIRROR_BUCKET}/{prefix}{name}",
                        opener=s3_opener(client, MIRROR_BUCKET, prefix + name),
                    )
                )
            else:  # not mirrored: physionet.org
                missing += 1
                url = f"{base}/{name}"
                files.append(
                    RemoteFile(
                        path=name,
                        checksum=f"sha256:{digest}",
                        url=url,
                        opener=http_opener(session, url),
                    )
                )
        if missing and not website_for_missing:
            raise FetchError(
                f"{self.dataset.id}: {missing} files are not on PhysioNet's AWS mirror, and "
                "download.physionet_source is aws-mirror"
            )
        if missing:
            log.warning(
                "%s: %d files are not on the AWS mirror; reading them from physionet.org "
                "(download.physionet_source: aws-mirror-else-physionet.org)",
                self.dataset.id,
                missing,
            )
        files.append(
            RemoteFile(
                path=SUMS,
                size=len(sums),
                url=f"s3://{MIRROR_BUCKET}/{prefix}{SUMS}",
                opener=s3_opener(client, MIRROR_BUCKET, prefix + SUMS),
            )
        )
        log.info(
            "%s: reading PhysioNet's AWS mirror s3://%s/%s (download.physionet_source)",
            self.dataset.id,
            MIRROR_BUCKET,
            prefix,
        )
        return files

    def _https_files(self, version: str) -> Iterator[RemoteFile]:
        session = self._session()
        base = f"{BASE}/files/{self.source.slug}/{version}"
        response = session.get(f"{base}/{SUMS}", timeout=60)
        if response.status_code in (401, 403):
            raise FetchError(
                f"{self.dataset.id}: access denied. Sign the data use agreement on "
                f"{BASE}/content/{self.source.slug}/ and set PHYSIONET_USERNAME/PASSWORD"
            )
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


def _list_sizes(client: S3Client, prefix: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=MIRROR_BUCKET, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            sizes[item["Key"][len(prefix) :]] = int(item["Size"])
    return sizes
