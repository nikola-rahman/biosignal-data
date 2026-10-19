"""TUH (rsync over SSH, which can only write into a folder: the folder is then uploaded)
and manual downloads (`biosignal-data import`).

Staging keeps what the tool produced: the raw layer is a copy of what the source handed
over, not an unpacked version of it.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from biosignal_data.catalog.models import ManualSource, RsyncSource
from biosignal_data.fetchers.base import (
    Fetcher,
    FetchError,
    LocalDirFetcher,
    ManualDownloadError,
    RemoteFile,
    require_env,
)

log = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    tool = cmd[0]
    if shutil.which(tool) is None:
        raise FetchError(f"`{tool}` is not installed or not on PATH (see docs/access.md)")
    result = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=False)
    if result.returncode != 0:
        raise FetchError(f"`{' '.join(cmd[:2])} ...` exited with code {result.returncode}")


class StagedFetcher(Fetcher):
    """Base: subclasses implement `stage(folder)`; files are then read from that folder.

    The folder is <staging.dir>/<dataset id> from the config file, reused across runs so
    that the tools resume. After a complete download it is deleted unless staging.keep."""

    _staging: Path | None = None
    _owned = False  # True: a staging folder we created (never a user's --from folder)

    def stage(self, folder: Path) -> None:
        raise NotImplementedError

    def prepare(self) -> None:
        if self.options.from_dir is not None:
            self._staging = self.options.from_dir  # `import`: downloaded by hand
            return
        if self.options.staging_dir is None or self.options.keep_staging is None:
            raise FetchError(
                f"{self.dataset.id} is downloaded to disk first: set staging.dir and "
                "staging.keep in the config file"
            )
        folder = self.options.staging_dir.expanduser() / self.dataset.id.replace("/", "__")
        folder.mkdir(parents=True, exist_ok=True)
        self._staging, self._owned = folder, True
        self.stage(folder)

    def list_files(self) -> Iterator[RemoteFile]:
        if self._staging is None:
            raise FetchError("prepare() must run before list_files()")
        yield from LocalDirFetcher(self.dataset, self.options, root=self._staging).list_files()

    def cleanup(self, complete: bool) -> None:
        if self._owned and self._staging and complete and not self.options.keep_staging:
            log.warning(
                "%s: deleting the staging folder %s (staging.keep: false)",
                self.dataset.id,
                self._staging,
            )
            shutil.rmtree(self._staging, ignore_errors=True)


class RsyncFetcher(StagedFetcher):
    @property
    def source(self) -> RsyncSource:
        return self._source(RsyncSource)

    def check_access(self) -> None:
        require_env(self.source.ssh_identity_env)

    def stage(self, folder: Path) -> None:
        (key,) = require_env(self.source.ssh_identity_env)
        # Only this key, and no ~/.ssh/config: nothing from the machine is used implicitly.
        ssh = f"ssh -F /dev/null -o IdentitiesOnly=yes -i {shlex.quote(key)}"
        # -L follows links (TUH subcorpora link into TUEG); trailing slash copies contents.
        remote = self.source.remote.rstrip("/") + "/"
        _run(["rsync", "-auxL", "--partial", "-e", ssh, remote, str(folder) + "/"], cwd=folder)


class ManualFetcher(StagedFetcher):
    """Nothing to download programmatically: `biosignal-data import` a folder instead."""

    @property
    def source(self) -> ManualSource:
        return self._source(ManualSource)

    def check_access(self) -> None:
        raise ManualDownloadError(
            f"download by hand (see `biosignal-data info {self.dataset.id}`), then run "
            f"`biosignal-data import {self.dataset.id} <folder> --version <label> --dest ...`"
        )

    def prepare(self) -> None:
        self.check_access()  # always raises: there is nothing to download

    def stage(self, folder: Path) -> None:
        self.check_access()
