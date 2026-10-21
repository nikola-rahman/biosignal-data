"""PhysioNet versions read straight from PhysioNet's AWS copy (pipeline `physionet-mirror`),
for the standard layer.

For a version on the AWS copy with no ZIP there (icentia11k 1.0: 1.6 million files, 1.1 TB),
the converter reads the version's files from the copy as it needs them: each recording's
files are downloaded to a temporary folder, checked against SHA256SUMS.txt, converted and
removed. Nothing is kept as a raw version and no disk holds the whole version.

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
from typing import TYPE_CHECKING, Any, BinaryIO

from biosignal_data import __version__
from biosignal_data.catalog.models import Dataset, PhysioNetSource
from biosignal_data.fetchers.physionet import MIRROR_BUCKET, MIRROR_REGION, SUMS, parse_sha256sums
from biosignal_data.fetchers.s3 import s3_client
from biosignal_data.storage.layout import MANIFEST, SUCCESS, VersionLocation
from biosignal_data.storage.manifest import FileEntry, Manifest, utcnow

if TYPE_CHECKING:
    from biosignal_data.storage.stores import Unfinished

log = logging.getLogger(__name__)


class MirrorError(RuntimeError):
    """The version on the AWS copy is not complete or not there."""


class MirrorStore:
    """A raw version on PhysioNet's AWS copy, read only (see the module). It can go to
    conversion processes: each process opens its own S3 client."""

    process_safe = True

    def __init__(
        self, slug: str, version: str, location: VersionLocation, manifest: bytes,
        connections: int,
    ) -> None:  # fmt: skip
        self.uri = f"s3://{MIRROR_BUCKET}/{slug}/{version}"
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
            self._client = s3_client(region=MIRROR_REGION, connections=self._connections)
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
            head = self.client.head_object(Bucket=MIRROR_BUCKET, Key=self._source_key(key))
        except self.client.exceptions.ClientError:
            return None
        return int(head["ContentLength"])

    def get_bytes(self, key: str) -> bytes | None:
        if (local := self._local(key)) is not None:
            return local
        try:
            body = self.client.get_object(Bucket=MIRROR_BUCKET, Key=self._source_key(key))
        except self.client.exceptions.NoSuchKey:
            return None
        data: bytes = body["Body"].read()
        return data

    def get_range(self, key: str, offset: int, length: int) -> bytes:
        response = self.client.get_object(
            Bucket=MIRROR_BUCKET, Key=self._source_key(key),
            Range=f"bytes={offset}-{offset + length - 1}",
        )  # fmt: skip
        data: bytes = response["Body"].read()
        return data

    def open(self, key: str) -> BinaryIO:
        if (local := self._local(key)) is not None:
            return io.BytesIO(local)
        body = self.client.get_object(Bucket=MIRROR_BUCKET, Key=self._source_key(key))["Body"]
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


def mirror(dataset: Dataset, version: str, *, connections: int) -> MirrorStore:
    """The version on the AWS copy as a raw version (MirrorStore), with its manifest made
    from SHA256SUMS.txt and one listing of the version folder."""
    source = dataset.source
    if not isinstance(source, PhysioNetSource):
        raise TypeError(f"{dataset.id}: not a PhysioNet dataset")
    if source.credentialed:
        raise MirrorError(f"{dataset.id}: credentialed; not on the open AWS copy")
    client = s3_client(region=MIRROR_REGION, connections=connections)
    prefix = f"{source.slug}/{version}/"
    try:
        sums_text = client.get_object(Bucket=MIRROR_BUCKET, Key=prefix + SUMS)["Body"].read()
    except client.exceptions.NoSuchKey as exc:
        raise MirrorError(f"{dataset.id} {version}: not on PhysioNet's AWS copy") from exc
    sums = dict(parse_sha256sums(sums_text.decode("utf-8")))
    sizes: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=MIRROR_BUCKET, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            sizes[item["Key"][len(prefix) :]] = int(item["Size"])
    missing = sorted(set(sums) - set(sizes))
    if missing:
        raise MirrorError(
            f"{dataset.id} {version}: {len(missing)} files of {SUMS} are not on the AWS copy, "
            f"e.g. {missing[0]}"
        )
    entries = []
    unlisted = sorted(p for p in sizes if p not in sums)
    for path in sorted(sums):
        entries.append(
            FileEntry(
                path=path, size=sizes[path], sha256=sums[path], md5=None,
                source_checksum=f"sha256:{sums[path]}",
                source_url=f"s3://{MIRROR_BUCKET}/{prefix}{path}",
            )
        )  # fmt: skip
    for path in unlisted:  # not in SHA256SUMS.txt (itself, files added later): read once
        data = client.get_object(Bucket=MIRROR_BUCKET, Key=prefix + path)["Body"].read()
        entries.append(
            FileEntry(
                path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                md5=hashlib.md5(data).hexdigest(),  # noqa: S324 - recorded, not for security
                source_url=f"s3://{MIRROR_BUCKET}/{prefix}{path}",
            )
        )  # fmt: skip
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
        "%s %s: %d files on the AWS copy (%.1f GB), %d checked against %s",
        dataset.id, version, len(entries), sum(sizes.values()) / 1e9, len(sums), SUMS,
    )  # fmt: skip
    return MirrorStore(source.slug, version, location, manifest.to_json(), connections)
