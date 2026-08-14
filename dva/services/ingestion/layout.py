"""Fixed-width layout loading and sample validation.

Layout files are CSV with three columns, matching the reference tools
(file_merge.py ``load_layout``):

    Field, From, Length

where ``From`` is 1-based. The detector converts this into a
``FixedWidthLayout`` with 0-based ``[start, end)`` ranges:

    start = From - 1
    end   = start + Length

``validate_layout_against_sample`` catches obvious mismatches (e.g. a
layout wider than the actual lines) so the user can MODIFY/REPROCESS at
detection time instead of failing at parse time.
"""

from __future__ import annotations

import csv
from pathlib import Path

from models.detection_models import ColumnSpec, FixedWidthLayout

_REQUIRED_COLUMNS = {"Field", "From", "Length"}


class LayoutLoadError(Exception):
    """Raised when a layout CSV cannot be read or is malformed."""


class LayoutValidationError(Exception):
    """Raised when a layout does not fit the sampled file's lines."""


def load_layout_csv(path: str | Path) -> FixedWidthLayout:
    """Load a From/Length/Field layout CSV into a validated layout.

    Raises:
        LayoutLoadError: on missing columns or malformed numeric values.
    """
    specs: list[ColumnSpec] = []
    with open(path, newline="", encoding="cp1252", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if not _REQUIRED_COLUMNS.issubset(fieldnames):
            raise LayoutLoadError(
                f"Layout CSV must contain columns: {sorted(_REQUIRED_COLUMNS)}."
            )

        for row in reader:
            try:
                start = int(row["From"]) - 1  # 1-based -> 0-based
                length = int(row["Length"])
            except (TypeError, ValueError) as exc:
                raise LayoutLoadError(f"Invalid From/Length row: {row}") from exc
            specs.append(
                ColumnSpec(
                    field=str(row["Field"]).strip(),
                    start=start,
                    end=start + length,
                )
            )

    try:
        return FixedWidthLayout.from_specs(specs)
    except ValueError as exc:
        raise LayoutLoadError(f"Layout is invalid: {exc}") from exc


def validate_layout_against_sample(
    layout: FixedWidthLayout,
    sample_lines: list[str],
    tolerance: float = 0.05,
) -> list[str]:
    """Return a list of warnings when sample lines do not fit the layout.

    A warning is raised when more than ``tolerance`` (default 5%) of the
    sampled lines are shorter than the layout's total width.
    """
    total = len(sample_lines)
    if total == 0:
        return ["sample is empty; cannot validate layout."]

    covered = sum(1 for line in sample_lines if len(line) >= layout.width)
    coverage = covered / total
    if coverage < (1.0 - tolerance):
        return [
            f"only {covered}/{total} sampled lines are >= layout width "
            f"{layout.width} ({coverage:.0%} coverage)."
        ]
    return []
