"""PhysioNet versions read straight from PhysioNet's AWS copy (pipeline `physionet-mirror`),
for the standard layer.

For a version on the AWS copy with no ZIP there (icentia11k 1.0: 1.6 million files, 1.1 TB),
the converter reads the version's files from the copy as it needs them: each recording's
files are downloaded to a temporary folder, checked against SHA256SUMS.txt, converted and
removed. Nothing is kept as a raw version and no disk holds the whole version.

Open versions that are not in the public bucket (ltstdb, challenge-2020, mimic-iv-ecg, i-care,
mimic3wdb, ...: "open access, login required" on PhysioNet) are in PhysioNet's second bucket,
read through one S3 access point per version (`<slug>-v<version with dashes>-<nn>`, nn from
01; PhysioNet's physionet-build, project/cloud/s3.py). It needs an AWS account linked to a
PhysioNet account (Settings, Cloud) that requested AWS access on the version's page, and
that account's keys (docs/access.md). PhysioNet pays for the transfer: the bucket is not
requester-pays (2026-10-21: requests without x-amz-request-payer succeed).

MirrorStore is a read-only store that looks like a raw version (raw/<id>/<version>/: its
manifest, _SUCCESS, data/<path>): the manifest is made from SHA256SUMS.txt and one listing
of the version folder (sizes); data/<path> is <slug>/<version>/<path> on the copy. Files on
the copy that SHA256SUMS.txt does not list are read once when the store is made, and their
SHA-256 is computed, as the file-by-file copy does; a file SHA256SUMS.txt lists that the
copy does not have is an error.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, BinaryIO

from biosignal_data import __version__
from biosignal_data.catalog.models import Dataset, PhysioNetSource
from biosignal_data.fetchers.physionet import MIRROR_BUCKET, MIRROR_REGION, SUMS, parse_sha256sums
from biosignal_data.fetchers.s3 import s3_client
from biosignal_data.progress import Ticker
from biosignal_data.storage.layout import MANIFEST, SUCCESS, VersionLocation
from biosignal_data.storage.manifest import FileEntry, Manifest, utcnow

if TYPE_CHECKING:
    from biosignal_data.storage.stores import Unfinished

log = logging.getLogger(__name__)
ACCESS_POINT_ACCOUNT = "724665945834"  # PhysioNet's AWS account
ACCESS_POINTS = 9  # access point numbers tried: 01 ... 09


class MirrorError(RuntimeError):
    """The version on the AWS copy is not complete or not there."""


class MirrorStore:
    """A raw version on PhysioNet's AWS copy, read only (see the module). It can go to
    conversion processes: each process opens its own S3 client."""

    process_safe = True
    header_reads = 64  # small files read at the same time (standard/convert.py)

    def __init__(
        self, slug: str, version: str, location: VersionLocation, manifest: bytes,
        connections: int, bucket: str = MIRROR_BUCKET, keys: tuple[str, str] | None = None,
    ) -> None:  # fmt: skip
        self.uri = f"s3://{bucket}/{slug}/{version}"
        self._bucket = bucket  # the public bucket, or an access point's ARN
        self._keys = keys
        self._prefix = f"{slug}/{version}/"
        self._location = location
        self._data = location.data_prefix
        self._manifest = manifest
        self._connections = connections
        self._client: Any = None

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__ | {"_client": None}

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = s3_client(
                region=MIRROR_REGION, connections=max(self._connections, self.header_reads),
                keys=self._keys,
            )  # fmt: skip
        return self._client

    def _source_key(self, key: str) -> str:
        if not key.startswith(self._data):
            raise KeyError(f"{key}: not a file of {self._location.prefix}")
        return self._prefix + key[len(self._data) :]

    def _local(self, key: str) -> bytes | None:
        if key == f"{self._location.prefix}/{MANIFEST}":
            return self._manifest
        if key == f"{self._location.prefix}/{SUCCESS}":
            return b""
        return None

    def exists(self, key: str) -> bool:
        if self._local(key) is not None:
            return True
        return self.size(key) is not None

    def size(self, key: str) -> int | None:
        if (local := self._local(key)) is not None:
            return len(local)
        try:
            head = self.client.head_object(Bucket=self._bucket, Key=self._source_key(key))
        except self.client.exceptions.ClientError:
            return None
        return int(head["ContentLength"])

    def get_bytes(self, key: str) -> bytes | None:
        if (local := self._local(key)) is not None:
            return local
        try:
            body = self.client.get_object(Bucket=self._bucket, Key=self._source_key(key))
        except self.client.exceptions.NoSuchKey:
            return None
        data: bytes = body["Body"].read()
        return data

    def get_range(self, key: str, offset: int, length: int) -> bytes:
        response = self.client.get_object(
            Bucket=self._bucket, Key=self._source_key(key),
            Range=f"bytes={offset}-{offset + length - 1}",
        )  # fmt: skip
        data: bytes = response["Body"].read()
        return data

    def open(self, key: str) -> BinaryIO:
        if (local := self._local(key)) is not None:
            return io.BytesIO(local)
        body = self.client.get_object(Bucket=self._bucket, Key=self._source_key(key))["Body"]
        return body  # type: ignore[no-any-return]

    def list(self, prefix: str) -> Iterator[str]:
        raise NotImplementedError("the files of a mirrored version are in its manifest")

    def sizes(self, prefix: str) -> dict[str, int]:
        raise NotImplementedError("the files of a mirrored version are in its manifest")

    def _read_only(self, *args: object, **kwargs: object) -> None:
        raise PermissionError(f"{self.uri} is read only")

    put_stream = put_bytes = delete = delete_many = _read_only

    def unfinished(self, prefix: str) -> tuple[Unfinished, ...]:
        return ()

    def discard(self, item: Unfinished) -> None:
        self._read_only()


def access_point(slug: str, version: str, number: int) -> str:
    """The ARN of a version's access point to PhysioNet's second bucket (see the module)."""
    name = f"{slug}-v{version.replace('.', '-')}-{number:02d}"
    return f"arn:aws:s3:{MIRROR_REGION}:{ACCESS_POINT_ACCOUNT}:accesspoint/{name}"


def _sums(
    slug: str, version: str, connections: int, keys: tuple[str, str] | None
) -> tuple[Any, str, bytes] | None:
    """(client, bucket, SHA256SUMS.txt) from the public bucket, else from the version's
    access point with `keys`; None if neither has the version."""
    key = f"{slug}/{version}/{SUMS}"
    client = s3_client(region=MIRROR_REGION, connections=max(connections, MirrorStore.header_reads))
    try:
        sums = client.get_object(Bucket=MIRROR_BUCKET, Key=key)["Body"].read()
        return client, MIRROR_BUCKET, sums
    except client.exceptions.NoSuchKey:
        pass
    if keys is None:
        return None
    client = s3_client(
        region=MIRROR_REGION, connections=max(connections, MirrorStore.header_reads), keys=keys
    )
    for number in range(1, ACCESS_POINTS + 1):
        bucket = access_point(slug, version, number)
        try:
            return client, bucket, client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except client.exceptions.NoSuchKey:
            return None
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] != "AccessDenied":
                raise
    return None


def mirror(
    dataset: Dataset, version: str, *, connections: int, keys: tuple[str, str] | None = None
) -> MirrorStore:
    """The version on the AWS copy as a raw version (MirrorStore), with its manifest made
    from SHA256SUMS.txt and one listing of the version folder. From the public bucket, else
    with `keys` (an AWS account linked to PhysioNet) through the version's access point."""
    source = dataset.source
    if not isinstance(source, PhysioNetSource):
        raise TypeError(f"{dataset.id}: not a PhysioNet dataset")
    if source.credentialed:
        raise MirrorError(f"{dataset.id}: credentialed; not on the open AWS copy")
    prefix = f"{source.slug}/{version}/"
    found = _sums(source.slug, version, connections, keys)
    if found is None:
        where = "PhysioNet's AWS copy" + ("" if keys else " (public bucket; no AWS keys given)")
        raise MirrorError(f"{dataset.id} {version}: not on {where}")
    client, bucket, sums_text = found
    sums = dict(parse_sha256sums(sums_text.decode("utf-8")))
    sizes: dict[str, int] = {}
    listing = Ticker(f"{dataset.id} {version}: listing", len(sums))
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            sizes[item["Key"][len(prefix) :]] = int(item["Size"])
        listing.add(len(page.get("Contents", [])), sum(i["Size"] for i in page.get("Contents", [])))
    listing.finish()
    missing = sorted(set(sums) - set(sizes))
    if missing:
        raise MirrorError(
            f"{dataset.id} {version}: {len(missing)} files of {SUMS} are not in {bucket}, "
            f"e.g. {missing[0]}"
        )
    entries = []
    unlisted = sorted(p for p in sizes if p not in sums)
    for path in sorted(sums):
        entries.append(
            FileEntry(
                path=path, size=sizes[path], sha256=sums[path], md5=None,
                source_checksum=f"sha256:{sums[path]}",
                source_url=f"s3://{bucket}/{prefix}{path}",
            )
        )  # fmt: skip
    # Not in SHA256SUMS.txt (itself, files added later): read once, HEADER_READS at a time
    # (challenge-2020 1.0.2: 72 files, 2.8 GB, the teams' code as ZIPs).
    reading = Ticker(f"{dataset.id} {version}: SHA-256 of files not in {SUMS}", len(unlisted))

    def unlisted_entry(path: str) -> FileEntry:
        data = client.get_object(Bucket=bucket, Key=prefix + path)["Body"].read()
        reading.add(1, len(data))
        return FileEntry(
            path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
            md5=hashlib.md5(data).hexdigest(),  # noqa: S324 - recorded, not for security
            source_url=f"s3://{bucket}/{prefix}{path}",
        )  # fmt: skip

    with ThreadPoolExecutor(max_workers=MirrorStore.header_reads) as pool:
        entries += pool.map(unlisted_entry, unlisted)
    reading.finish()
    if [p for p in unlisted if p != SUMS]:
        log.warning(
            "%s: %d files on the AWS copy are not in %s (their SHA-256 computed now), e.g. %s",
            dataset.id, len(unlisted) - (SUMS in unlisted), SUMS,
            next(p for p in unlisted if p != SUMS),
        )  # fmt: skip
    location = VersionLocation(dataset.id, version, "raw")
    manifest = Manifest(
        dataset_id=dataset.id, version=version, version_origin="source",
        source=source.model_dump(mode="json") | {"pipeline": "physionet-mirror"},
        license=dataset.license, retrieved_at=utcnow(), completed_at=utcnow(),
        tool=f"biosignal-data {__version__}", files=sorted(entries, key=lambda e: e.path),
    )  # fmt: skip
    log.info(
        "%s %s: %d files on the AWS copy (%.1f GB, %s), %d checked against %s",
        dataset.id, version, len(entries), sum(sizes.values()) / 1e9, bucket, len(sums), SUMS,
    )  # fmt: skip
    return MirrorStore(
        source.slug, version, location, manifest.to_json(), connections, bucket,
        keys if bucket != MIRROR_BUCKET else None,
    )  # fmt: skip
