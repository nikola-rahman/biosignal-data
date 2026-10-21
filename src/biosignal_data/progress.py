"""Progress lines for the phases without a progress bar (listing, header reads, files read
for their SHA-256): a line every few seconds, so that a long phase shows it is working."""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)
EVERY = 5.0  # seconds between lines


class Ticker:
    """Counts the items of one phase, from any thread, and logs `what: done of total (rate,
    ETA)` at most every EVERY seconds; `total` is None when not known ahead."""

    def __init__(self, what: str, total: int | None = None, unit: str = "files") -> None:
        self.what, self.total, self.unit = what, total, unit
        self.done = 0
        self.nbytes = 0
        self._start = self._last = time.monotonic()
        self._lock = threading.Lock()

    def add(self, n: int = 1, nbytes: int = 0) -> None:
        with self._lock:
            self.done += n
            self.nbytes += nbytes
            now = time.monotonic()
            if now - self._last < EVERY:
                return
            self._last = now
            line = self._line(now)
        log.info("%s", line)

    def finish(self) -> None:
        """The last line, when the phase took long enough to have logged."""
        now = time.monotonic()
        if now - self._start >= EVERY:
            log.info("%s: done", self._line(now))

    def _line(self, now: float) -> str:
        elapsed = max(now - self._start, 1e-9)
        rate = self.done / elapsed
        size = f", {self.nbytes / 1e9:.1f} GB" if self.nbytes else ""
        if self.total is None:
            return f"{self.what}: {self.done:,} {self.unit}{size} ({rate:,.0f}/s)"
        left = (self.total - self.done) / rate if rate > 0 else float("inf")
        eta = f"{left / 60:.0f} min left" if left != float("inf") else "?"
        return (
            f"{self.what}: {self.done:,} of {self.total:,} {self.unit}{size} ({rate:,.0f}/s, {eta})"
        )
