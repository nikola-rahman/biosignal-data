"""Terminal output for the CLI: logging and the download progress display.

Built from rich's own parts: two `Progress` displays in one `Live` (rich's documented
pattern for several progress bars), with rich's built-in columns.

    all datasets ━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━  12/31  0:03:10
    ⠴ physionet/mitdb 1.0.0 ━━━━━━━╺━━━━━━━  2.1/5.9 MB  410 kB/s  0:00:04  3/5 files  103.dat

Outside a terminal (output to a file) the bars are off and logs are plain lines from the
standard library handler.
"""

from __future__ import annotations

import logging
import threading
from types import TracebackType

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Column

from biosignal_data.text import plural

console = Console(stderr=True)


def setup_logging(verbose: bool) -> None:
    """Our logs at INFO (DEBUG with -v); third-party libraries only at WARNING."""
    handler: logging.Handler
    if console.is_terminal:
        handler = RichHandler(console=console, show_path=False, markup=False)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.WARNING, format=fmt, handlers=[handler], force=True)
    logging.getLogger("biosignal_data").setLevel(logging.DEBUG if verbose else logging.INFO)


class DownloadDisplay:
    """The run bar (datasets done) above one bar per dataset in progress."""

    def __init__(self, datasets: int) -> None:
        enabled = console.is_terminal
        self.run = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=not enabled,
        )
        self.datasets = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", table_column=Column(no_wrap=True)),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[files]}", table_column=Column(no_wrap=True)),
            TextColumn(
                "[dim]{task.fields[current]}",
                table_column=Column(no_wrap=True, overflow="ellipsis", ratio=1),
            ),
            console=console,
            expand=True,
            disable=not enabled,
        )
        self._live = Live(Group(self.run, self.datasets), console=console, transient=False)
        self._enabled = enabled
        self._run_task = self.run.add_task("all datasets", total=datasets)
        self._failed = 0
        self._lock = threading.Lock()

    def __enter__(self) -> DownloadDisplay:
        if self._enabled:
            self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._enabled:
            if exc_type is not None:
                self._live.transient = True  # a stopped run: no stale last frame
            self._live.__exit__(exc_type, exc, tb)

    def dataset_done(self, ok: bool) -> None:
        with self._lock:
            if not ok:
                self._failed += 1
            failed = f" [red]· {self._failed} not copied[/]" if self._failed else ""
            self.run.update(self._run_task, advance=1, description=f"all datasets{failed}")

    def bar(self, dataset_id: str) -> ProgressBar:
        return ProgressBar(self.datasets, dataset_id)


class ProgressBar:
    """Implements `ingest.IngestProgress` with one task of the dataset display: the bar
    measures bytes; the files done and the newest file in progress are shown beside it."""

    def __init__(self, progress: Progress, dataset_id: str) -> None:
        self._progress = progress
        self._dataset_id = dataset_id
        self._lock = threading.Lock()
        self._active: list[str] = []  # files downloading now, oldest first
        self._files = 0
        self._files_done = 0
        self._task: TaskID = progress.add_task(
            f"{dataset_id} [dim]listing files", total=None, files="", current=""
        )

    def start(self, version: str, files: int, done: int, total_bytes: int | None) -> None:
        self._files, self._files_done = files, done
        self._progress.update(
            self._task,
            description=f"{self._dataset_id} [dim]{version}",
            total=total_bytes,
            files=self._files_text(),
        )

    def _files_text(self) -> str:
        return f"{self._files_done:,}/{plural(self._files, 'file')}"

    def file_started(self, path: str) -> None:
        with self._lock:
            self._active.append(path)
            self._show_current()

    def advance(self, n: int) -> None:
        self._progress.advance(self._task, n)

    def file_done(self, path: str, ok: bool) -> None:
        with self._lock:
            if path in self._active:
                self._active.remove(path)
            if ok:
                self._files_done += 1
                self._progress.update(self._task, files=self._files_text())
            self._show_current()

    def _show_current(self) -> None:
        more = f" (+{len(self._active) - 1})" if len(self._active) > 1 else ""
        current = f"{self._active[-1]}{more}" if self._active else ""
        self._progress.update(self._task, current=current)

    def finish(self) -> None:
        """Remove the bar; the CLI prints one result line instead."""
        self._progress.remove_task(self._task)
