"""Any S3 bucket or access point: public (anonymous) or credentialed (e.g. BDSP)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Any, BinaryIO

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from biosignal_data.catalog.models import S3Source
from biosignal_data.fetchers.base import Fetcher, FetchError, RemoteFile, require_env

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def s3_client(*, region: str, connections: int, keys: tuple[str, str] | None = None) -> S3Client:
    """A client for a source bucket on AWS: anonymous, or with the given (key id, secret).
    `connections`: the most requests it runs at the same time. Nothing is read from AWS
    config files or AWS_* variables."""
    config = Config(signature_version=UNSIGNED) if keys is None else Config()
    config = config.merge(
        Config(  # type: ignore[call-arg]  # stubs lag botocore 1.31+
            retries={"max_attempts": 10, "mode": "adaptive"},
            ignore_configured_endpoint_urls=True,
            max_pool_connections=connections,
        )
    )
    return boto3.session.Session().client(
        "s3",
        region_name=region,
        aws_access_key_id=keys[0] if keys else None,
        aws_secret_access_key=keys[1] if keys else None,
        config=config,
    )


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
    version_is_label = True

    @property
    def source(self) -> S3Source:
        return self._source(S3Source)

    def _keys(self) -> tuple[str, str] | None:
        if self.source.anonymous:
            return None
        if not (self.source.key_id_env and self.source.secret_env):  # the schema ensures it
            raise FetchError(f"{self.dataset.id}: key_id_env/secret_env missing in the catalog")
        key_id, secret = require_env(self.source.key_id_env, self.source.secret_env)
        return key_id, secret

    def check_access(self) -> None:
        self._keys()

    def list_files(self) -> Iterator[RemoteFile]:
        if self.source.prefix is None:
            raise FetchError(
                f"{self.dataset.id}: the key prefix is not known yet. The provider gives it "
                f"after approval; write it into catalog/{self.dataset.id}.yaml as source.prefix."
            )
        client = s3_client(
            region=self.source.region, connections=self.options.connections, keys=self._keys()
        )
        yield from iter_s3_files(client, self.source.bucket, self.source.prefix)
