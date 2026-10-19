"""Sources whose official tool writes to local disk: rsync (TUH), the NSRR gem, the Kaggle
CLI, and manual downloads. They stage into a folder, then the folder is ingested as-is.

Staging keeps what the tool produced. Archives stay archives: the raw layer is a copy of
what the source handed over, not an unpacked version of it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from biosignal_data.catalog.models import KaggleSource, ManualSource, NsrrSource, RsyncSource
from biosignal_data.fetchers.base import (
    Fetcher,
    FetchError,
    LocalDirFetcher,
    MissingCredentialsError,
    RemoteFile,
    require_env,
)


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    tool = cmd[0]
    if shutil.which(tool) is None:
        raise FetchError(f"`{tool}` is not installed or not on PATH (see docs/access.md)")
    result = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=False)
    if result.returncode != 0:
        raise FetchError(f"`{' '.join(cmd[:2])} ...` exited with code {result.returncode}")


class StagedFetcher(Fetcher):
    """Base: subclasses implement `stage(folder)`; files are then read from that folder."""

    _staging: Path | None = None
    _owns_staging = False

    def stage(self, folder: Path) -> None:
        raise NotImplementedError

    def prepare(self) -> None:
        if self.options.from_dir is not None:
            self._staging = self.options.from_dir  # already downloaded by hand
            return
        base = self.options.staging_dir
        if base is not None:
            folder = base.expanduser() / self.dataset.id.replace("/", "__")
            folder.mkdir(parents=True, exist_ok=True)  # reused across runs: tools resume
        else:
            folder = Path(tempfile.mkdtemp(prefix=self.dataset.id.replace("/", "__") + "-"))
            self._owns_staging = True
        self.stage(folder)
        self._staging = folder

    def list_files(self) -> Iterator[RemoteFile]:
        if self._staging is None:
            raise FetchError("prepare() must run before list_files()")
        yield from LocalDirFetcher(self.dataset, self.options, root=self._staging).list_files()

    def cleanup(self) -> None:
        if self._owns_staging and self._staging and not self.options.keep_staging:
            shutil.rmtree(self._staging, ignore_errors=True)


class RsyncFetcher(StagedFetcher):
    @property
    def source(self) -> RsyncSource:
        return self._source(RsyncSource)

    def stage(self, folder: Path) -> None:
        ssh = "ssh"
        if self.source.ssh_identity_env:
            (key,) = require_env(self.source.ssh_identity_env)
            ssh = f"ssh -i {key}"
        # -L follows links (TUH subcorpora link into TUEG); trailing slash copies contents.
        remote = self.source.remote.rstrip("/") + "/"
        _run(["rsync", "-auxL", "--partial", "-e", ssh, remote, str(folder) + "/"], cwd=folder)


class NsrrFetcher(StagedFetcher):
    """Uses the official NSRR Ruby gem (`gem install nsrr`). The token is the one shown on
    https://sleepdata.org/token after the dataset's data request has been approved."""

    @property
    def source(self) -> NsrrSource:
        return self._source(NsrrSource)

    def stage(self, folder: Path) -> None:
        (token,) = require_env("NSRR_TOKEN")
        target = self.source.slug + (f"/{self.source.subpath}" if self.source.subpath else "")
        # The gem takes the token as an argument, so it is visible in the process list of
        # this machine while the download runs. Run on a single-user machine.
        _run(["nsrr", "download", target, f"--token={token}"], cwd=folder)

    def list_files(self) -> Iterator[RemoteFile]:
        # The gem writes into <folder>/<slug>/...; keep paths relative to the dataset root.
        if self._staging is None:
            raise FetchError("prepare() must run before list_files()")
        root = self._staging / self.source.slug
        if not root.is_dir():
            root = self._staging
        yield from LocalDirFetcher(self.dataset, self.options, root=root).list_files()


class KaggleFetcher(StagedFetcher):
    """Uses the official Kaggle CLI. Competitions also require accepting their rules on the
    website with the same account first."""

    @property
    def source(self) -> KaggleSource:
        return self._source(KaggleSource)

    def stage(self, folder: Path) -> None:
        kaggle_dir = Path.home() / ".kaggle"
        has_file = any((kaggle_dir / n).is_file() for n in ("access_token", "kaggle.json"))
        has_env = bool(os.environ.get("KAGGLE_API_TOKEN")) or bool(
            os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
        )
        if not (has_file or has_env):
            raise MissingCredentialsError(
                "Kaggle needs a login: `kaggle auth login`, or KAGGLE_API_TOKEN / "
                "~/.kaggle/access_token (see docs/access.md)"
            )
        if self.source.kind == "competition":
            cmd = ["kaggle", "competitions", "download", "-c", self.source.ref, "-p", "."]
        else:
            cmd = ["kaggle", "datasets", "download", "-d", self.source.ref, "-p", "."]
        _run(cmd, cwd=folder)


class ManualFetcher(StagedFetcher):
    """Nothing to download programmatically: requires `--from-dir`."""

    @property
    def source(self) -> ManualSource:
        return self._source(ManualSource)

    def stage(self, folder: Path) -> None:
        raise FetchError(
            f"{self.dataset.id} must be downloaded by hand (see its docs page), then ingested "
            "with `--from-dir <folder>`"
        )
