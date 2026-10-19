"""Small text helpers for messages."""

from __future__ import annotations


def plural(n: int, word: str) -> str:
    """`1 file`, `2 files`."""
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"
