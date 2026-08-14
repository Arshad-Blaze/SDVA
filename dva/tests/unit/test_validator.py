import polars as pl
import pytest

from models.validation_models import (
    ColumnMapping,
    PriceType,
    UnitType,
    ValidationConfig,
)
from services.validator.validator import (
    ValidationError,
    validate,
)


def write_parquet(tmp_path, name, data: pl.DataFrame):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(path)
    return path


def build_config(**kwargs):
    kwargs.setdefault(
        "bau",
        ColumnMapping(store_col="STORE", units_col="UNITS", price_col="PRICE"),
    )
    kwargs.setdefault(
        "test",
        ColumnMapping(store_col="STORE", units_col="UNITS", price_col="PRICE"),
    )
    return ValidationConfig(**kwargs)


def make_datasets(tmp_path):
    bau = write_parquet(
        tmp_path,
        "bau.parquet",
        pl.DataFrame(
            {
                "STORE": ["1001", "1001", "1002"],
                "UNITS": [2, 3, 5],
                "PRICE": [10, 10, 20],
            }
        ),
    )
    test = write_parquet(
        tmp_path,
        "test.parquet",
        pl.DataFrame(
            {
                "STORE": ["1001", "1003"],
                "UNITS": [1, 7],
                "PRICE": [10, 5],
            }
        ),
    )
    return bau, test


def test_missing_store_metrics_computed(tmp_path):
    bau, test = make_datasets(tmp_path)
    result = validate(
        bau, test, build_config(store_analysis=True, sales_analysis=True)
    )
    metrics = dict(result.metrics["store"])
    assert metrics["Stores present in TEST (missing in BAU)"] == "1"
    assert metrics["Stores present in BAU (missing in TEST)"] == "1"
    assert metrics["Total number of records (BAU)"] == "3"
    assert metrics["Total number of records (TEST)"] == "2"


def test_sales_totals_and_dollars(tmp_path):
    bau, test = make_datasets(tmp_path)
    result = validate(bau, test, build_config(sales_analysis=True))
    bau_df = result.frames["sales"]["bau"]
    assert bau_df["store"].to_list() == ["1001", "1002"]
    assert bau_df["records"].to_list() == [2, 1]
    assert bau_df["units"].to_list() == [5, 5]
    # dollars = units * price (unit_price -> price*units)... default total price.
    assert bau_df["dollars"].to_list() == [20, 20]


def test_unit_price_and_implied_decimals(tmp_path):
    bau, _ = make_datasets(tmp_path)
    cfg = build_config(
        sales_analysis=True,
        price_type_bau=PriceType.UNIT_PRICE,
        implied_dollars_bau=True,
    )
    result = validate(bau, write_parquet(
        bau.parent, "empty_test.parquet",
        pl.DataFrame({"STORE": [], "UNITS": [], "PRICE": []}),
    ), cfg)
    bau_df = result.frames["sales"]["bau"]
    # dollars = UNITS * PRICE / 100 for store 1001: 5*10/100 = 0.5
    assert bau_df["dollars"].to_list() == [0.5, 1.0]


def test_implied_units_scales_totals_and_dollars(tmp_path):
    bau, test = make_datasets(tmp_path)
    cfg = build_config(
        sales_analysis=True,
        price_type_bau=PriceType.UNIT_PRICE,
        implied_units_bau=True,
        implied_units_test=True,
    )
    result = validate(bau, test, cfg)
    bau_df = result.frames["sales"]["bau"]
    # units stored as cents: 2+3 -> 0.05, 5 -> 0.05
    assert bau_df["units"].to_list() == pytest.approx([0.05, 0.05])
    # dollars (unit price) = price * implied units: 10 * 0.05 = 0.5
    assert bau_df["dollars"].to_list() == pytest.approx([0.5, 1.0])


def test_sales_comparison_difference_and_percent(tmp_path):
    bau, test = make_datasets(tmp_path)
    result = validate(bau, test, build_config(sales_analysis=True))
    comparison = result.frames["sales"]["comparison"]
    rows = {r["STORE_NUMBER"]: r for r in comparison.iter_rows(named=True)}
    assert set(rows) == {"1001", "1002", "1003"}
    # Store 1001: BAU units 5, TEST 1 -> diff 4, BAU$ 20 TEST$ 10 -> 100%
    assert rows["1001"]["UNITS DIFFERENCE"] == 4
    assert rows["1001"]["DOLLAR DIFFERENCE"] == 10
    assert rows["1001"]["Unit % Difference"] == 80.0
    assert rows["1001"]["Dollar % Difference"] == 50.0
    metrics = dict(result.metrics["sales"])
    assert metrics["Unique Store Count"] == "3"
    assert metrics["Total BAU Sales"] == "40.0"
    assert metrics["Total Test Sales"] == "15.0"
    assert metrics["Total Unit Difference"] == "2.0"
    assert "top_5_sales" in result.frames["sales"]
    assert "bottom_5_units" in result.frames["sales"]


def test_store_level_weighted_units(tmp_path):
    def with_weighted(data: pl.DataFrame):
        return data.with_columns(
            pl.col("UNITS").alias("WTD").cast(pl.Float64).fill_null(0) + 10
        )

    bau = write_parquet(
        tmp_path, "bau.parquet",
        pl.DataFrame(
            {
                "STORE": ["1001", "1001", "1002"],
                "UNITS": [2, 3, 5],
                "PRICE": [10, 10, 20],
            }
        ),
    )
    bau_w = with_weighted(pl.read_parquet(bau))
    bau_w.write_parquet(bau)
    test = write_parquet(
        tmp_path, "test.parquet",
        pl.DataFrame(
            {
                "STORE": ["1001", "1003"],
                "UNITS": [1, 7],
                "PRICE": [10, 5],
                "WTD": [None, 90],
            }
        ),
    )
    cfg = build_config(
        sales_analysis=True,
        price_type_bau=PriceType.UNIT_PRICE,
        bau=ColumnMapping(store_col="STORE", units_col="UNITS", price_col="PRICE", weighted_units_col="WTD"),
        test=ColumnMapping(store_col="STORE", units_col="UNITS", price_col="PRICE", weighted_units_col="WTD"),
    )
    result = validate(bau, test, cfg)
    bau_df = result.frames["sales"]["bau"]
    # BAU store 1001: WTD 12 & 13 -> weighted sum 25; dollars = 10*12 + 10*13 = 250
    assert bau_df["units"].to_list() == [25.0, 15.0]
    assert bau_df["dollars"].to_list() == [250.0, 300.0]
    test_df = result.frames["sales"]["test"]
    # TEST store 1001: WTD None -> raw units 1; store 1003: weighted 90
    assert test_df["units"].to_list() == [1.0, 90.0]


def test_unit_type_column_in_sales(tmp_path):
    bau, test = make_datasets(tmp_path)
    cfg = build_config(
        sales_analysis=True,
        bau=ColumnMapping(store_col="STORE", units_col="UNITS", price_col="PRICE", units_type=UnitType.WEIGHT),
        test=ColumnMapping(store_col="STORE", units_col="UNITS", price_col="PRICE", units_type=UnitType.WEIGHT),
    )
    result = validate(bau, test, cfg)
    bau_df = result.frames["sales"]["bau"]
    assert bau_df["unit_type"].to_list() == ["weight", "weight"]
    comparison = result.frames["sales"]["comparison"]
    assert set(comparison["Units Type"].to_list()) == {"weight"}


def test_upc_analysis_missing_upcs(tmp_path):
    bau = write_parquet(
        tmp_path, "bau.parquet",
        pl.DataFrame(
            {"UPC": ["11111", "22222"], "DESC": ["a", "b"], "UNITS": [1, 2], "PRICE": [3, 4]}
        ),
    )
    test = write_parquet(
        tmp_path, "test.parquet",
        pl.DataFrame(
            {"UPC": ["22222", "33333"], "DESC": ["b", "c"], "UNITS": [1, 2], "PRICE": [3, 4]}
        ),
    )
    cfg = build_config(
        upc_analysis=True,
        bau=ColumnMapping(upc_col="UPC", desc_col="DESC", units_col="UNITS", price_col="PRICE"),
        test=ColumnMapping(upc_col="UPC", desc_col="DESC", units_col="UNITS", price_col="PRICE"),
    )
    result = validate(bau, test, cfg)
    metrics = dict(result.metrics["upc"])
    assert metrics["UPCs missing in TEST"] == "1"
    assert metrics["UPCs missing in BAU"] == "1"


def test_store_keys_are_normalised(tmp_path):
    # Leading/trailing spaces and case differences collapse to one key,
    # but a genuinely different store number stays distinct.
    bau = write_parquet(
        tmp_path, "bau.parquet",
        pl.DataFrame({"STORE": [" 1001 ", "1001", "1002"]}),
    )
    test = write_parquet(
        tmp_path, "test.parquet",
        pl.DataFrame({"STORE": ["1001", "1003"]}),
    )
    result = validate(
        bau, test, build_config(store_analysis=True)
    )
    metrics = dict(result.metrics["store"])
    assert metrics["Total number of unique stores in BAU"] == "2"
    assert metrics["Stores present in BAU (missing in TEST)"] == "1"
    assert metrics["Stores present in TEST (missing in BAU)"] == "1"


def test_disabled_analyses_are_skipped(tmp_path):
    bau, test = make_datasets(tmp_path)
    result = validate(bau, test, build_config())  # nothing enabled
    assert result.metrics == {}
    assert result.frames == {}


def test_incomplete_config_raises(tmp_path):
    bau, test = make_datasets(tmp_path)
    cfg = ValidationConfig(store_analysis=True)  # no column mappings
    with pytest.raises(ValueError, match="Validation config incomplete"):
        validate(bau, test, cfg)


def test_missing_dataset_raises(tmp_path):
    cfg = build_config(store_analysis=True)
    with pytest.raises(ValidationError, match="not found"):
        validate(tmp_path / "nope.parquet", tmp_path / "also.parquet", cfg)