"""Bounded sampling of a raw source file for detection.

The Detector (Boundary 3) must operate on bounded samples, never the
whole file. Sampling is capped by BOTH bytes and lines so neither one
enormous line nor a huge number of tiny lines can blow up memory.

Encoding detection (fills the ``encoding`` field of DetectionResult):
  - utf-8 with BOM -> "utf-8" (BOM stripped by the sampler)
  - strict utf-8  -> "utf-8"
  - cp1252        -> "cp1252" (the dominant encoding in the reference
                     tools, which hardcode "cp1252, errors=ignore")
  - latin-1       -> "latin-1" as a last resort (decodes every byte)

The detection method is recorded so the user can see how trustworthy
the encoding guess is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_BYTES = 1_000_000  # ~1 MB of text
DEFAULT_MAX_LINES = 20_000

# Bytes read to decide encoding; enough to be representative and cheap.
_ENCODING_PREFIX_BYTES = 256 * 1024

# BOM signatures we understand.
_UTF8_BOM = b"\xef\xbb\xbf"


@dataclass(slots=True)
class Sample:
    """A bounded window of physical lines from a raw file."""

    lines: list[str]
    encoding: str
    encoding_method: str
    bytes_read: int

    @property
    def line_count(self) -> int:
        return len(self.lines)


def detect_encoding(path: str | Path, prefix_bytes: int = _ENCODING_PREFIX_BYTES) -> tuple[str, str]:
    """Return ``(encoding, method)`` for a file, or ``(None, ...)`` on failure.

    Method is used for display/confidence only; the encoding is what the
    parser will actually open the file with.
    """
    with open(path, "rb") as handle:
        prefix = handle.read(prefix_bytes)

    if prefix.startswith(_UTF8_BOM):
        return "utf-8", "utf-8-bom"

    for encoding in ("utf-8", "cp1252"):
        try:
            prefix.decode(encoding)
            return encoding, encoding
        except UnicodeDecodeError:
            continue

    return "latin-1", "latin-1-fallback"


def read_bounded_sample(
    path: str | Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_lines: int = DEFAULT_MAX_LINES,
) -> Sample:
    """Read a bounded sample of physical lines from ``path``.

    Lines are newline-stripped. A leading BOM is stripped from the first
    line so downstream fixed-width slicing never sees ``\\ufeff``.
    """
    encoding, method = detect_encoding(path)

    lines: list[str] = []
    total_bytes = 0
    with open(path, encoding=encoding, errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if total_bytes == 0 and line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")
            lines.append(line)
            total_bytes += len(raw.encode(encoding, errors="replace"))
            if total_bytes >= max_bytes or len(lines) >= max_lines:
                break

    return Sample(
        lines=lines,
        encoding=encoding,
        encoding_method=method,
        bytes_read=total_bytes,
    )
