"""Checksum helpers for download integrity verification.

MFT metadata may expose a checksum per file; when it does, the File Mover
verifies it after the size check (03, section 2). Files are hashed in
fixed-size chunks so arbitrarily large files are never loaded into memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# 8 MB read chunks strike a good balance between syscall overhead and RAM.
_CHUNK_SIZE = 8 * 1024 * 1024

# Algorithms we are willing to use. md5 is not for security here; it is a
# cheap integrity check against a known MFT-provided value.
SUPPORTED_ALGORITHMS = frozenset({"md5", "sha1", "sha256"})


def checksum_file(path: str | Path, algorithm: str = "md5") -> str:
    """Return the hex digest of a file without loading it into memory.

    Args:
        path: file to hash.
        algorithm: any algorithm in ``SUPPORTED_ALGORITHMS``.

    Returns:
        Lower-case hex digest string.

    Raises:
        ValueError: for an unsupported algorithm.
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported checksum algorithm '{algorithm}'; "
            f"choose from {sorted(SUPPORTED_ALGORITHMS)}."
        )

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def checksums_match(actual: str, expected: str) -> bool:
    """Compare two checksum hex strings, case-insensitively."""
    return actual.lower() == expected.lower()
