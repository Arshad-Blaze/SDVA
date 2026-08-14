"""Unit tests for validation config and report models."""

from pathlib import Path

import pytest

from models.validation_models import (
    ColumnMapping,
    PriceType,
    UnitType,
    ValidationConfig,
    ValidationReport,
)


def test_required_for_reports_missing_columns():
    mapping = ColumnMapping(store_col="STORE", units_col=None, price_col=None)
    assert mapping.required_for("store") == ["units_col", "price_col"]
    assert mapping.required_for("sales") == []


def test_enabled_analyses_lists_selection():
    config = ValidationConfig(store_analysis=True, upc_analysis=True)
    assert config.enabled_analyses() == ["store", "upc"]


def test_validate_passes_when_complete():
    config = ValidationConfig(
        store_analysis=True,
        bau=ColumnMapping(store_col="S", units_col="U", price_col="P"),
        test=ColumnMapping(store_col="S", units_col="U", price_col="P"),
    )
    config.validate()  # should not raise


def test_validate_raises_on_missing_columns():
    config = ValidationConfig(
        store_analysis=True,
        bau=ColumnMapping(store_col="S", units_col=None, price_col="P"),
        test=ColumnMapping(store_col="S", units_col="U", price_col="P"),
    )
    with pytest.raises(ValueError, match="missing"):
        config.validate()


def test_validate_ignores_disabled_analyses():
    # upc_analysis is off, so missing upc/desc columns are irrelevant.
    config = ValidationConfig(
        bau=ColumnMapping(),
        test=ColumnMapping(),
    )
    config.validate()  # should not raise


def test_json_round_trip():
    config = ValidationConfig(
        store_analysis=True,
        upc_analysis=True,
        bau=ColumnMapping(
            store_col="S", units_col="U", price_col="P",
            upc_col="UPC", desc_col="DESC",
        ),
        test=ColumnMapping(
            store_col="S", units_col="U", price_col="P",
            upc_col="UPC", desc_col="DESC",
        ),
        price_type_bau=PriceType.UNIT_PRICE,
        implied_dollars_bau=True,
    )
    payload = config.to_json_dict()
    assert payload["price_type_bau"] == "unit_price"
    assert payload["implied_dollars_bau"] is True
    assert payload["bau"]["store_col"] == "S"
    assert payload["bau"]["units_type"] == "qty"
    config.bau.units_type = UnitType.WEIGHT
    assert config.to_json_dict()["bau"]["units_type"] == "weight"


def test_report_accumulates_artifacts():
    report = ValidationReport(bau_dataset="d_bau", test_dataset="d_test")
    report.add_artifact("store", Path("/tmp/store.xlsx"))
    report.add_artifact("store", Path("/tmp/store_2.xlsx"))
    report.add_artifact("upc", Path("/tmp/upc.xlsx"))
    assert len(report.artifacts["store"]) == 2
    assert len(report.artifacts["upc"]) == 1
