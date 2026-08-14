"""Unit tests for detection, layout, and approval models."""

import pytest

from models.detection_models import (
    ApprovalDecision,
    ApprovedConfig,
    ColumnSpec,
    DetectionResult,
    FileFormat,
    FixedWidthLayout,
    StructureType,
)


# ---------------------------------------------------------------------
# ColumnSpec / FixedWidthLayout
# ---------------------------------------------------------------------
def test_column_spec_rejects_negative_or_inverted_range():
    with pytest.raises(ValueError, match="start"):
        ColumnSpec(field="a", start=-1, end=5)
    with pytest.raises(ValueError, match="greater than start"):
        ColumnSpec(field="a", start=5, end=5)


def test_layout_sorts_columns_and_computes_width():
    layout = FixedWidthLayout.from_specs(
        [ColumnSpec("STORE", 4, 8), ColumnSpec("UPC", 0, 4)]
    )
    assert layout.column_names() == ["UPC", "STORE"]
    assert layout.width == 8


def test_layout_rejects_overlaps():
    with pytest.raises(ValueError, match="Overlapping"):
        FixedWidthLayout.from_specs(
            [ColumnSpec("a", 0, 5), ColumnSpec("b", 4, 9)]
        )


def test_layout_rejects_empty():
    with pytest.raises(ValueError, match="at least one column"):
        FixedWidthLayout(tuple())


# ---------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------
def test_confidence_must_be_bounded():
    with pytest.raises(ValueError, match="confidence"):
        DetectionResult(
            format=FileFormat.DELIMITED,
            structure_type=StructureType.FLAT,
            confidence=1.5,
        )


@pytest.mark.parametrize(
    "fmt, structure, delimiter, layout, expected",
    [
        (FileFormat.DELIMITED, StructureType.FLAT, ",", None, True),
        (FileFormat.DELIMITED, StructureType.MULTILINE, ",", None, True),
        (FileFormat.FIXED_WIDTH, StructureType.FLAT, None, "layout", True),
        (FileFormat.FIXED_WIDTH, StructureType.MULTILINE, None, "layout", False),
        (FileFormat.DELIMITED, StructureType.RECORD_TYPED, ",", None, False),
        (FileFormat.FIXED_WIDTH, StructureType.RECORD_TYPED, None, "layout", False),
        (FileFormat.UNSUPPORTED, StructureType.FLAT, None, None, False),
        # Missing delimiter / layout means the format was not confirmed.
        (FileFormat.DELIMITED, StructureType.FLAT, None, None, False),
        (FileFormat.FIXED_WIDTH, StructureType.FLAT, None, None, False),
    ],
)
def test_is_parseable(fmt, structure, delimiter, layout, expected):
    layout_obj = None
    if layout == "layout":
        layout_obj = FixedWidthLayout.from_specs(
            [ColumnSpec(field="A", start=0, end=4)]
        )
    result = DetectionResult(
        format=fmt,
        structure_type=structure,
        delimiter=delimiter,
        layout=layout_obj,
        confidence=0.9,
    )
    assert result.is_parseable() is expected


def test_needs_approval_uses_threshold():
    confident = DetectionResult(
        format=FileFormat.DELIMITED,
        structure_type=StructureType.FLAT,
        confidence=0.95,
    )
    uncertain = DetectionResult(
        format=FileFormat.DELIMITED,
        structure_type=StructureType.FLAT,
        confidence=0.6,
    )
    assert not confident.needs_approval()
    assert uncertain.needs_approval()


# ---------------------------------------------------------------------
# ApprovedConfig
# ---------------------------------------------------------------------
def test_delimited_config_requires_delimiter():
    config = ApprovedConfig(
        source_format=FileFormat.DELIMITED,
        delimiter=None,
        columns=["store", "units"],
    )
    with pytest.raises(ValueError, match="delimiter"):
        config.validate()


def test_delimited_config_rejects_multi_char_delimiter():
    config = ApprovedConfig(
        source_format=FileFormat.DELIMITED,
        delimiter="||",
    )
    with pytest.raises(ValueError, match="single character"):
        config.validate()


def test_delimited_config_rejects_layout():
    config = ApprovedConfig(
        source_format=FileFormat.DELIMITED,
        delimiter="|",
        fixed_width_layout=FixedWidthLayout.from_specs(
            [ColumnSpec("a", 0, 4)]
        ),
    )
    with pytest.raises(ValueError, match="must not carry"):
        config.validate()


def test_fixed_width_config_requires_layout():
    config = ApprovedConfig(
        source_format=FileFormat.FIXED_WIDTH,
        fixed_width_layout=None,
    )
    with pytest.raises(ValueError, match="approved layout"):
        config.validate()


def test_fixed_width_config_rejects_delimiter():
    config = ApprovedConfig(
        source_format=FileFormat.FIXED_WIDTH,
        delimiter="|",
        fixed_width_layout=FixedWidthLayout.from_specs(
            [ColumnSpec("a", 0, 4)]
        ),
    )
    with pytest.raises(ValueError, match="must not carry a delimiter"):
        config.validate()


def test_valid_delimited_config_passes():
    config = ApprovedConfig(
        source_format=FileFormat.DELIMITED,
        delimiter="|",
        columns=["store", "units"],
    )
    config.validate()  # should not raise


def test_unsupported_format_cannot_be_approved():
    config = ApprovedConfig(source_format=FileFormat.UNSUPPORTED)
    with pytest.raises(ValueError, match="unsupported"):
        config.validate()


def test_approval_decision_enum_values():
    assert ApprovalDecision.ACCEPT.value == "accept"
    assert ApprovalDecision.MODIFY.value == "modify"
    assert ApprovalDecision.REPROCESS.value == "reprocess"


# ---------------------------------------------------------------------
# JSON persistence round-trips.
# ---------------------------------------------------------------------
def test_fixed_width_layout_json_round_trip():
    layout = FixedWidthLayout.from_specs(
        [ColumnSpec("STORE", 0, 4), ColumnSpec("PRICE", 4, 9)]
    )
    restored = FixedWidthLayout.from_json_dict(layout.to_json_dict())
    assert restored.column_names() == ["STORE", "PRICE"]
    assert restored.width == layout.width


def test_approved_config_delimited_json_round_trip():
    config = ApprovedConfig(
        source_format=FileFormat.DELIMITED,
        delimiter=",",
        encoding="cp1252",
        header_present=True,
        columns=["STORE", "UNITS", "PRICE"],
        schema_overrides={"PRICE": "Float64"},
    )
    restored = ApprovedConfig.from_json_dict(config.to_json_dict())
    assert restored.delimiter == ","
    assert restored.columns == ["STORE", "UNITS", "PRICE"]
    assert restored.schema_overrides == {"PRICE": "Float64"}
    assert restored.source_format is FileFormat.DELIMITED
    assert restored.fixed_width_layout is None


def test_approved_config_skip_rows_round_trip_and_validation():
    config = ApprovedConfig(
        source_format=FileFormat.DELIMITED,
        delimiter=",",
        header_present=True,
        columns=["STORE"],
        skip_rows=3,
    )
    restored = ApprovedConfig.from_json_dict(config.to_json_dict())
    assert restored.skip_rows == 3

    config.skip_rows = -1
    with pytest.raises(ValueError, match="skip_rows"):
        config.validate()


def test_approved_config_fixed_width_json_round_trip():
    config = ApprovedConfig(
        source_format=FileFormat.FIXED_WIDTH,
        header_present=False,
        columns=["STORE", "PRICE"],
        fixed_width_layout=FixedWidthLayout.from_specs(
            [ColumnSpec("STORE", 0, 4), ColumnSpec("PRICE", 4, 9)]
        ),
    )
    restored = ApprovedConfig.from_json_dict(config.to_json_dict())
    assert restored.source_format is FileFormat.FIXED_WIDTH
    assert restored.fixed_width_layout is not None
    assert restored.fixed_width_layout.column_names() == ["STORE", "PRICE"]
