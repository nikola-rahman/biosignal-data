"""PhysioNet versions staged from their ZIP, for the standard layer (pipeline `physionet-zip`).

The raw layer is a build input of the standard layer. For a PhysioNet version with a
complete ZIP, the fastest checked path is the ZIP itself (2026-10-20): one object on
PhysioNet's AWS copy (`s3://physionet-open/<slug>/<slug>-<version>.zip`, no credentials),
read in parallel 8 MiB ranges, instead of one request per published file (PTB-XL: about 220
requests for 1.8 GB instead of 87,000). Checks:

- the ZIP against its S3 ETag (the MD5 of its 8 MiB parts, as PhysioNet uploads it);
- every member against the version's SHA256SUMS.txt (PhysioNet writes it before the ZIP;
  members are below one folder `<title>-<version>/`, removed here), and the ZIP's own
  CRC-32 when it is read;
- every file that SHA256SUMS.txt lists must be in the ZIP: an incomplete ZIP (challenge-2018
  1.0.0: 48 MB of 267 GB) is refused.

Files in the ZIP that SHA256SUMS.txt does not list are kept with their size and SHA-256,
and named in a warning, as the file-by-file copy does. Files that PhysioNet added to the
version on its AWS copy after the ZIP was made (apnea-ecg 1.0.0: the test-set answers
x01.apn ... x35.apn, 17,268 labelled minutes) are not in the ZIP: one listing of the
version folder finds them; they are copied and checked by their size, as the file-by-file
copy does, and named in a warning. The staged version is a local raw
version (`raw/<id>/<version>/data/...`, `_manifest.json`, `_SUCCESS`) for the converter; it
is deleted when the standard version is made.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from boto3.s3.transfer import TransferConfig

from biosignal_data import __version__
from biosignal_data.catalog.models import Dataset, PhysioNetSource
from biosignal_data.fetchers.physionet import MIRROR_BUCKET, MIRROR_REGION, SUMS, parse_sha256sums
from biosignal_data.fetchers.s3 import s3_client
from biosignal_data.storage.layout import MANIFEST, SUCCESS, VersionLocation
from biosignal_data.storage.manifest import FileEntry, Manifest, utcnow
from biosignal_data.storage.stores import LocalStore

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

log = logging.getLogger(__name__)
PART = 8 * 2**20  # PhysioNet uploads with boto3's default parts; the ETag counts them
CHUNK = 2**20


class ZipCheckError(RuntimeError):
    """The ZIP or a member differs from what PhysioNet published, or the ZIP is incomplete."""


@dataclass(frozen=True)
class Staged:
    store: LocalStore  # holds raw/<id>/<version>/ for the converter
    manifest: Manifest
    folder: Path  # remove when done


def stage(dataset: Dataset, version: str, work: Path, *, connections: int) -> Staged:
    """Download, check and extract one version into `work`; return a local raw version."""
    source = dataset.source
    if not isinstance(source, PhysioNetSource):
        raise TypeError(f"{dataset.id}: not a PhysioNet dataset")
    if source.credentialed:
        raise ZipCheckError(f"{dataset.id}: credentialed; not on the open AWS copy")
    folder = work / f"zip-{source.slug}-{version}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    client = s3_client(region=MIRROR_REGION, connections=connections)
    prefix = f"{source.slug}/{version}/"
    sums = dict(
        parse_sha256sums(
            client.get_object(Bucket=MIRROR_BUCKET, Key=prefix + SUMS)["Body"]
            .read()
            .decode("utf-8")
        )
    )
    key = f"{source.slug}/{source.slug}-{version}.zip"
    head = client.head_object(Bucket=MIRROR_BUCKET, Key=key)
    archive = folder / "archive.zip"
    log.info(
        "%s: downloading s3://%s/%s (%d bytes)",
        dataset.id,
        MIRROR_BUCKET,
        key,
        head["ContentLength"],
    )
    client.download_file(
        MIRROR_BUCKET, key, str(archive),
        Config=TransferConfig(multipart_chunksize=PART, max_concurrency=connections),
    )  # fmt: skip
    if archive.stat().st_size != head["ContentLength"]:
        raise ZipCheckError(
            f"{key}: {archive.stat().st_size} bytes, expected {head['ContentLength']}"
        )
    if _etag(archive) != head["ETag"].strip('"'):
        raise ZipCheckError(f"{key}: the MD5 of its parts differs from the S3 ETag")

    location = VersionLocation(dataset.id, version, "raw")
    store = LocalStore(folder / "store")
    entries, extra = _extract(archive, key, sums, folder / "store", location)
    archive.unlink()
    entries += _added_later(client, prefix, {e.path for e in entries}, folder / "store", location)
    missing = sorted(set(sums) - {e.path for e in entries})
    if missing:
        raise ZipCheckError(
            f"{key} is incomplete: {len(missing)} files of SHA256SUMS.txt are not in it, "
            f"e.g. {missing[0]}"
        )
    if extra:
        log.warning(
            "%s: %d files in the ZIP are not in SHA256SUMS.txt (kept, with their SHA-256), "
            "e.g. %s", dataset.id, len(extra), extra[0],
        )  # fmt: skip
    manifest = Manifest(
        dataset_id=dataset.id, version=version, version_origin="source",
        source=source.model_dump(mode="json") | {"pipeline": "physionet-zip", "archive": key},
        license=dataset.license, retrieved_at=utcnow(), completed_at=utcnow(),
        tool=f"biosignal-data {__version__}", files=sorted(entries, key=lambda e: e.path),
    )  # fmt: skip
    store.put_bytes(f"{location.prefix}/{MANIFEST}", manifest.to_json())
    store.put_bytes(f"{location.prefix}/{SUCCESS}", b"", content_type="text/plain")
    return Staged(store, manifest, folder)


def _added_later(
    client: S3Client, prefix: str, have: set[str], root: Path, location: VersionLocation
) -> list[FileEntry]:
    """Files of the version on the AWS copy that the ZIP does not have: copied, checked by
    their size (SHA256SUMS.txt does not list them either)."""
    paginator = client.get_paginator("list_objects_v2")
    found = []
    for page in paginator.paginate(Bucket=MIRROR_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            path = item["Key"][len(prefix) :]
            if path and path not in have and path != SUMS:
                target = root / location.data_key(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(MIRROR_BUCKET, item["Key"], str(target))
                if target.stat().st_size != item["Size"]:
                    raise ZipCheckError(
                        f"{item['Key']}: {target.stat().st_size} bytes, expected {item['Size']}"
                    )
                data = target.read_bytes()
                found.append(
                    FileEntry(
                        path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                        md5=hashlib.md5(data).hexdigest(),  # noqa: S324 - recorded, not for security
                        source_checksum=None, source_url=f"s3://{MIRROR_BUCKET}/{item['Key']}",
                    )
                )  # fmt: skip
    if found:
        log.warning(
            "%d files on the AWS copy are not in the ZIP (added later; checked by size), e.g. %s",
            len(found), found[0].path,
        )  # fmt: skip
    return found


def _extract(
    archive: Path, key: str, sums: dict[str, str], root: Path, location: VersionLocation
) -> tuple[list[FileEntry], list[str]]:
    """Every member into `root` as a raw version's data, each checked against SHA256SUMS.txt
    (and the ZIP's CRC-32); returns the entries and the members SHA256SUMS.txt does not list."""
    entries, extra = [], []
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        top = _version_folder(key, [i.filename for i in members], sums)
        outside = [i.filename for i in members if not i.filename.startswith(top)]
        if outside:
            log.warning(
                "%s: %d files are outside the version folder %r (not part of the published "
                "version; left out), e.g. %s", key, len(outside), top, outside[0],
            )  # fmt: skip
        written: set[str] = set()
        for info in (i for i in members if i.filename.startswith(top)):
            path = info.filename[len(top) :]
            if path in written:
                raise ZipCheckError(f"{key}: {path} is in the ZIP twice")
            written.add(path)
            target = root / location.data_key(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            sha, md5 = hashlib.sha256(), hashlib.md5()  # noqa: S324 - recorded, not for security
            with z.open(info) as src, target.open("wb") as dst:  # the CRC-32 is checked here
                while chunk := src.read(CHUNK):
                    sha.update(chunk)
                    md5.update(chunk)
                    dst.write(chunk)
            if path in sums and sha.hexdigest() != sums[path]:
                raise ZipCheckError(f"{key}: {path} differs from SHA256SUMS.txt")
            if path not in sums and path != SUMS:
                extra.append(path)
            entries.append(
                FileEntry(
                    path=path, size=info.file_size, sha256=sha.hexdigest(), md5=md5.hexdigest(),
                    source_checksum=f"sha256:{sums[path]}" if path in sums else None,
                    source_url=f"s3://{MIRROR_BUCKET}/{key}#{info.filename}",
                )
            )  # fmt: skip
    return entries, extra


def _version_folder(key: str, names: list[str], sums: dict[str, str]) -> str:
    """The folder the version's files are in (`<folder>/`): the top folder that holds the
    files SHA256SUMS.txt lists. Other folders next to it (incartdb 1.0.0: `.old-headers/`,
    the 2008 headers that the version replaced, with the same file names) are not part of
    the published version."""
    listed: dict[str, int] = {}
    for name in names:
        top, sep, rest = name.partition("/")
        if sep:
            listed[top + "/"] = listed.get(top + "/", 0) + (rest in sums)
    best = max(listed.values(), default=0)
    tops = [t for t, n in listed.items() if n == best]
    if best == 0 or len(tops) != 1:
        raise ZipCheckError(f"{key}: no single folder holds the files of {SUMS} ({listed})")
    return tops[0]


def _etag(path: Path) -> str:
    """The S3 ETag of a file uploaded in PART-byte parts: MD5 of the part MD5s, `-<n>`;
    a file of one part has the plain MD5."""
    digests = []
    with path.open("rb") as f:
        while part := f.read(PART):
            digests.append(hashlib.md5(part).digest())  # noqa: S324 - the S3 ETag scheme
    if len(digests) == 1:
        return digests[0].hex()
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"  # noqa: S324
