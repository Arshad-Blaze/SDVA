"""Field segmentation & record-structure heuristics (Detector internals).

These are pure, stateless helpers operating on physical text lines; the
Detector combines them into a ``DetectionResult``. They are split out to
keep ``detector.py`` under the project's 500-line budget.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter

# Quote-aware segmentation drops quoted segments before counting
# delimiters so a delimiter inside a quoted field is not a separator.
_QUOTED = re.compile(r'"[^"]*"')

# Short-prefix used to group lines when looking for record types.
RECORD_KEY_LEN = 2

# A record is multiline if its last physical line ends with this marker.
CONTINUATION_MARKER = "\\"


def strip_quoted(line: str) -> str:
    """Drop quoted segments so delimiters inside quotes are ignored."""
    return _QUOTED.sub("", line)


def parse_fields(line: str, delimiter: str | None) -> list[str]:
    """Quote-aware field split; falls back to plain split for no
    delimiter and tolerates malformed rows."""
    if not delimiter:
        return [line]
    # Protect escaped delimiters (e.g. ``\\,``) so csv does not see
    # them as separators; the parser reassembles them later.
    safe = line.replace("\\" + delimiter, "\ue000")
    try:
        fields = next(
            csv.reader([safe], delimiter=delimiter, skipinitialspace=True)
        )
        return [field.replace("\ue000", delimiter) for field in fields]
    except (csv.Error, StopIteration):
        return line.split(delimiter)


def first_key(line: str, delimiter: str | None) -> str:
    """First column of the line (or a length-limited prefix): the record
    key used for grouping. Blank lines produce an empty key, never an
    ``IndexError``."""
    fields = parse_fields(line, delimiter)
    if not fields:
        return ""
    if delimiter:
        return fields[0].strip()
    return line[:RECORD_KEY_LEN].strip()


def is_record_typed(lines: list[str], delimiter: str | None) -> bool:
    """Interleaved header/detail rows usually differ in shape: group
    lines by their first key and compare shapes across groups."""
    groups: dict[str, Counter] = {}
    for line in lines:
        if not line.strip():
            continue  # blank lines are padding, not a record type
        key = first_key(line, delimiter)
        shape = len(parse_fields(line, delimiter))
        groups.setdefault(key, Counter())[shape] += 1

    distinct_keys = [k for k in groups if groups[k]]
    if len(distinct_keys) < 2:
        return False  # a single dominant shape/record type

    # Record-typed only when groups actually differ in shape.
    shapes = {groups[k].most_common(1)[0][0] for k in distinct_keys}
    return len(shapes) >= 2


def is_multiline_delimited(lines: list[str]) -> bool:
    # Explicit continuation marker, or an unterminated quoted field.
    has_marker = any(line.rstrip().endswith(CONTINUATION_MARKER) for line in lines)
    unclosed_quotes = sum(line.count('"') % 2 for line in lines)
    return has_marker or unclosed_quotes > 0


def is_multiline_fixed_width(lines: list[str]) -> bool:
    # Fixed-width expects stable lengths; high variance means records
    # may span physical lines. To avoid flagging arbitrary junk text:
    #   - a dominant chunk length must exist (mode >= half the lines),
    #   - the longest line must be an integer multiple of that chunk,
    #     i.e. lines are chunks of larger fixed-width records,
    #   - lines share a stable record-start prefix.
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) < 2:
        return False
    lengths = [len(line) for line in non_empty]
    mode_len, mode_count = Counter(lengths).most_common(1)[0]
    if mode_count / len(lengths) < 0.5:
        return False
    longest = max(lengths)
    if longest <= mode_len:
        return False
    ratio = longest / mode_len
    if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        return False
    prefixes = [line[:RECORD_KEY_LEN] for line in non_empty]
    common, occurrences = Counter(prefixes).most_common(1)[0]
    return occurrences / len(non_empty) >= 0.6


def numeric_like_fraction(values: list[str]) -> float:
    """Fraction of values that look numeric; used to tell headers apart."""
    if not values:
        return 0.0
    numeric = sum(
        value.replace(".", "", 1).replace("-", "", 1).isdigit()
        for value in values
        if value != ""
    )
    return numeric / len(values)