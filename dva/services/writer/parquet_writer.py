"""Structured Parquet writer with atomic writes and verification.

The parser hands over an in-memory frame; this module persists it as a
local Parquet file under a ``.part`` name and atomically renames it, so a
crash never leaves a half-written file. ``verify_parquet`` then confirms
the written file's row count and columns by re-reading its metadata, and
returns a checksum for the integrity record.

Schema overrides from an ``ApprovedConfig`` are applied here as a
"trust but verify" cast: values that cannot be cast become nulls
(``strict=False``) and downstream validation reports them instead of the
whole file failing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from utils.checksums import checksum_file

# Polars type names accepted in schema_overrides.
_ALLOWED_OVERRIDES = ("Utf8", "Int64", "Int32", "Float64", "Float32", "Boolean")

# Truthy/falsy string tokens accepted when casting text to Boolean; Polars
# cannot cast Utf8 -> Boolean directly, so unknown tokens become null.
_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "on"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "off", ""}


def _override_expr(name: str, dtype_name: str) -> pl.Expr:
    """Cast expression honouring the strict=False null-on-fail promise."""
    if dtype_name == "Boolean":
        lowered = pl.col(name).cast(pl.String()).str.to_lowercase()
        return (
            pl.when(lowered.is_in(_TRUE_TOKENS))
            .then(pl.lit(True))
            .when(lowered.is_in(_FALSE_TOKENS))
            .then(pl.lit(False))
            .otherwise(None)
            .alias(name)
        )
    return pl.col(name).cast(pl.String()).cast(
        getattr(pl, dtype_name), strict=False
    )


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome of one structured-file write, for the file's metadata."""
    source_path: str
    target_path: str
    rows: int
    columns: list[str]
    bytes_written: int
    sha256: str


class ParquetWriteError(Exception):
    """Raised when the structured file cannot be persisted."""


def apply_schema_overrides(
    frame: pl.DataFrame, overrides: dict[str, str]
) -> pl.DataFrame:
    """Cast columns per ``schema_overrides``; uncastable values become null."""
    unknown = set(overrides) - set(frame.columns)
    if unknown:
        raise ParquetWriteError(
            f"schema_overrides reference unknown columns: {sorted(unknown)}"
        )
    for name, dtype_name in overrides.items():
        if dtype_name not in _ALLOWED_OVERRIDES:
            raise ParquetWriteError(
                f"Unsupported override dtype '{dtype_name}' for {name}."
            )
        frame = frame.with_columns(_override_expr(name, dtype_name))
    return frame


def write_parquet(
    frame: pl.DataFrame,
    target: str | Path,
    *,
    source_path: str = "",
    schema_overrides: dict[str, str] | None = None,
    compression: str = "snappy",
) -> WriteResult:
    """Persist ``frame`` to ``target`` atomically and report the result."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if schema_overrides:
        frame = apply_schema_overrides(frame, schema_overrides)

    part_path = target.with_name(target.name + ".part")
    try:
        frame.write_parquet(
            part_path,
            compression=compression,
            statistics=True,
            row_group_size=250_000,
        )
        os.replace(part_path, target)
    except Exception as exc:
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        raise ParquetWriteError(f"Could not write {target}: {exc}") from exc

    return WriteResult(
        source_path=source_path,
        target_path=str(target),
        rows=frame.height,
        columns=list(frame.columns),
        bytes_written=target.stat().st_size,
        sha256=checksum_file(target, algorithm="sha256"),
    )


def verify_parquet(
    target: str | Path,
    *,
    expected_rows: int | None = None,
    expected_columns: list[str] | None = None,
) -> dict:
    """Re-read ``target`` metadata; returns row/column verification dict.

    Columns are compared as a set (order is not guaranteed to round-trip
    identically through different writers).
    """
    target = Path(target)
    if not target.exists():
        raise ParquetWriteError(f"Parquet file missing: {target}")

    try:
        schema = pl.read_parquet_schema(target)
        rows = pl.scan_parquet(target).select(pl.len()).collect().item()
    except Exception as exc:
        raise ParquetWriteError(f"Could not read back {target}: {exc}") from exc

    actual_columns = list(schema.keys())
    checks = {
        "rows_matched": expected_rows is None or rows == expected_rows,
        "columns_matched": (
            expected_columns is None or set(actual_columns) == set(expected_columns)
        ),
    }
    return {
        "rows": rows,
        "columns": actual_columns,
        "sha256": checksum_file(target, algorithm="sha256"),
        **checks,
    }