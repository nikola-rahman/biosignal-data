"""Any S3 bucket or access point: public (anonymous) or credentialed (e.g. BDSP)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, BinaryIO

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from biosignal_data.catalog.models import S3Source
from biosignal_data.fetchers.base import Fetcher, FetchError, RemoteFile

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def s3_client(*, anonymous: bool, region: str, profile: str | None = None) -> S3Client:
    config = Config(signature_version=UNSIGNED) if anonymous else Config()
    config = config.merge(Config(retries={"max_attempts": 10, "mode": "adaptive"}))
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3", region_name=region, config=config)


def s3_opener(
    client: S3Client, bucket: str, key: str
) -> Callable[[], AbstractContextManager[BinaryIO]]:
    @contextmanager
    def opener() -> Iterator[BinaryIO]:
        body: Any = client.get_object(Bucket=bucket, Key=key)["Body"]
        try:
            yield body
        finally:
            body.close()

    return opener


def iter_s3_files(client: S3Client, bucket: str, prefix: str) -> Iterator[RemoteFile]:
    """List every object under `prefix`; paths are relative to `prefix`."""
    prefix = prefix if not prefix or prefix.endswith("/") else prefix + "/"
    paginator = client.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue  # folder placeholder objects
            found = True
            yield RemoteFile(
                path=key[len(prefix) :],
                size=int(item["Size"]),
                url=f"s3://{bucket}/{key}",
                opener=s3_opener(client, bucket, key),
            )
    if not found:
        raise FetchError(f"no objects under s3://{bucket}/{prefix}")


class S3Fetcher(Fetcher):
    @property
    def source(self) -> S3Source:
        return self._source(S3Source)

    def list_files(self) -> Iterator[RemoteFile]:
        prefix = self.options.source_prefix or self.source.prefix
        if prefix is None:
            raise FetchError(
                f"{self.dataset.id}: the source key prefix is not in the catalog. Look it up "
                "in the provider's instructions after approval and pass --source-prefix."
            )
        client = s3_client(
            anonymous=self.source.anonymous,
            region=self.source.region,
            profile=self.options.aws_profile,
        )
        yield from iter_s3_files(client, self.source.bucket, prefix)
