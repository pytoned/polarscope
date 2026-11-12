"""xray() on a LazyFrame by default: streaming statistics, no column in memory."""

from __future__ import annotations

import importlib
import math
import re
import warnings
from datetime import date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

import polarscope as ps

xray_mod = importlib.import_module("polarscope.xray")
NAN = float("nan")


# The frames test_lazyframe.py and test_xray_audit_fixes.py use.
def _mixed_frame(n: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "i": list(range(n)),
            "f_nan": [float("nan") if k % 7 == 0 else k * 1.5 for k in range(n)],
            "f_nulls": [None if k % 3 == 0 else k / 3 for k in range(n)],
            "all_null": pl.Series([None] * n, dtype=pl.Float64),
            "const": [5] * n,
            "flag": [k % 3 == 0 for k in range(n)],
            "dec": pl.Series(
                [Decimal(f"{k}.25") for k in range(n)], dtype=pl.Decimal(10, 2)
            ),
            "s": [f"v{k % 6}" for k in range(n)],
            "s_nulls": [None if k % 4 == 0 else f"name_{k}" for k in range(n)],
            "cat": pl.Series([f"c{k % 3}" for k in range(n)], dtype=pl.Categorical),
            "d": [date(2020, 1, 1) + timedelta(days=k) for k in range(n)],
            "ts": [datetime(2021, 5, 1) + timedelta(hours=k) for k in range(n)],
            "dur": [timedelta(minutes=k) for k in range(n)],
            "lst": [[k, k + 1] for k in range(n)],
        }
    )


def _mixed_with_nan(n: int = 400) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [float((i * 37) % 101) for i in range(n)],
            "nan_col": [NAN if i % 13 == 0 else float(i % 29) for i in range(n)],
            "nulls": [None if i % 7 == 0 else i % 11 for i in range(n)],
            "target": [NAN if i % 50 == 0 else float(i % 17) * 1.5 for i in range(n)],
            "s": [f"v{i % 9}" for i in range(n)],
            "d": [date(2020, 1, 1) + timedelta(days=i % 40) for i in range(n)],
        }
    )

# Statistics the streaming summary leaves out: they need all of a column's
# values at once.
EXACT_ONLY = {"25%", "50%", "75%", "iqr", "mad", "n_duplicates", "pct_duplicates"}


def xray_quietly(frame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ps.xray(frame, **kwargs)


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        # Plain output rounds to 2 decimals; streaming sums in another order.
        return a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 0.0101
    return a == b


FRAMES = {
    "mixed": _mixed_frame,
    "mixed_with_nan": _mixed_with_nan,
    "titanic": ps.titanic,
}
CONFIGS = [
    {},
    {"include": "all"},
    {"include": "all", "expanded": True},
    {"include": "all", "expanded": True, "model_usability": True, "target": True},
    {"include": "all", "normality_test": "anderson", "uniformity_test": "chi2", "expanded": True},
]


@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("config", CONFIGS)
def test_streaming_values_equal_the_dataframe(frame, config):
    df = FRAMES[frame]()
    config = dict(config)
    if config.pop("target", False):
        config["corr_target"] = "Survived" if frame == "titanic" else df.columns[0]
    exact = xray_quietly(df, great_tables=False, outlier_method="zscore", **config)
    streamed = xray_quietly(df.lazy(), great_tables=False, **config)

    assert set(exact.columns) - set(streamed.columns) <= EXACT_ONLY
    assert not set(streamed.columns) & EXACT_ONLY
    assert streamed["column"].to_list() == exact["column"].to_list()
    # Distinct counts are exact up to a million values, so every column
    # matches - quality flags included.
    for column in streamed.columns:
        for a, b in zip(exact[column].to_list(), streamed[column].to_list()):
            assert _same(a, b), (column, a, b)


def test_nanoplot_payloads_equal_the_dataframe():
    df = _mixed_with_nan()
    exact = xray_quietly(df, include="all", outlier_method="zscore")._tbl_data
    streamed = xray_quietly(df.lazy(), include="all")._tbl_data
    assert (
        exact["distribution_plot"].to_list() == streamed["distribution_plot"].to_list()
    )


def test_outliers_use_the_z_score_rule():
    values = [0.0] * 50 + [1.0] * 50 + [100.0]
    lf = pl.LazyFrame({"x": values})
    out = xray_quietly(lf, great_tables=False)
    assert out["n_outliers"][0] == 1
    assert xray_quietly(lf, great_tables=False, outlier_method="zscore").equals(out)


@pytest.mark.parametrize(
    "kwargs, named",
    [
        ({"outlier_method": "iqr"}, "outlier_method='iqr'"),
        (
            {"outlier_method": "percentile", "outlier_bounds": [0.05, 0.95]},
            "outlier_method='percentile'",
        ),
        ({"percentiles": [0.5]}, "percentiles"),
        ({"corr_target": "x", "corr_method": "spearman"}, "corr_method='spearman'"),
    ],
)
def test_statistics_that_need_every_value_ask_for_exact(kwargs, named):
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match=re.escape(named) + ".*exact=True"):
        ps.xray(lf, **kwargs)
    # The same options work with exact=True, and on a DataFrame.
    xray_quietly(lf, exact=True, great_tables=False, **kwargs)
    xray_quietly(lf.collect(), great_tables=False, **kwargs)


def test_dataframes_ignore_exact():
    df = _mixed_frame()
    assert xray_quietly(df, great_tables=False).equals(
        xray_quietly(df, great_tables=False, exact=False)
    )


def test_high_cardinality_strings_skip_top_values(monkeypatch):
    monkeypatch.setattr(xray_mod, "_TOP_VALUES_MAX_DISTINCT", 3)
    lf = pl.LazyFrame(
        {
            "few": ["a", "b", "a", "c"] * 50,
            "many": [f"id-{i:03d}" for i in range(200)],
            "x": [float(i) for i in range(200)],
        }
    )
    out = xray_quietly(lf, include="all", expanded=True, great_tables=False)
    few = out.filter(pl.col("column") == "few").row(0, named=True)
    many = out.filter(pl.col("column") == "many").row(0, named=True)
    assert (few["top"], few["top_freq"], few["n_unique"]) == ("a", 100, 3)
    assert many["top"] is None and many["top_3"] is None
    assert (many["min_length"], many["median_length"], many["max_length"]) == (6, 6.0, 6)
    assert many["count"] == 200

    html = xray_quietly(lf, include="all").as_raw_html()
    assert "Top values are left out for strings with more than 3 distinct" in html


def test_source_note_says_how_the_summary_was_made():
    df = pl.DataFrame({"x": [float(i % 97) for i in range(6000)]})
    note = re.compile(r'class="gt_sourcenote"[^>]*>(?:<[^>]+>)*([^<]*)')

    minimal = note.findall(xray_quietly(df.lazy()).as_raw_html())
    assert len(minimal) == 1
    assert "z-score" in minimal[0] and "exact=True" in minimal[0]
    assert "sample" not in minimal[0]  # minimal mode shows no tests

    expanded = note.findall(xray_quietly(df.lazy(), expanded=True).as_raw_html())[0]
    assert "Distinct counts above 1,000,000 are estimated" in expanded
    assert "systematic sample of about 3,000 rows" in expanded

    for frame in (df, df.lazy()):
        html = xray_quietly(frame, exact=True).as_raw_html()
        assert not note.findall(html)


def test_tests_use_a_systematic_sample_above_5000_rows():
    if not xray_mod._check_scipy_availability():
        pytest.skip("scipy not installed")
    skewed = [-math.log(1 - (k + 0.5) / 20_000) for k in range(20_000)]
    lf = pl.LazyFrame({"x": skewed, "u": [k / 20_000 for k in range(20_000)]})
    out = xray_quietly(lf, expanded=True, great_tables=False)
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    assert rows["x"]["normality_test"] == "NON-NORMAL (p<0.001)"
    assert rows["u"]["uniformity_test"].startswith("UNIFORM")


@pytest.mark.parametrize("distinct", [1, 999, 40_000, 50_000])
def test_distinct_counts(monkeypatch, distinct):
    # Exact up to _DISTINCT_SAMPLE; above it estimated from a hash sample,
    # never more than the number of values.
    monkeypatch.setattr(xray_mod, "_DISTINCT_SAMPLE", 10_000)
    lf = pl.LazyFrame({"x": [float(i % distinct) for i in range(50_000)]})
    out = xray_quietly(lf, expanded=True, great_tables=False).row(0, named=True)
    if distinct <= 10_000:
        assert out["n_unique"] == distinct
    else:
        assert abs(out["n_unique"] - distinct) <= 0.05 * distinct
        assert out["n_unique"] <= out["count"] == 50_000
        assert out["uniqueness_ratio"] <= 1.0


def test_empty_and_all_null_columns():
    lf = pl.LazyFrame(
        {
            "x": pl.Series([None] * 5, dtype=pl.Float64),
            "s": pl.Series([None] * 5, dtype=pl.String),
            "d": pl.Series([None] * 5, dtype=pl.Date),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = xray_quietly(lf, include="all", expanded=True, great_tables=False)
    assert out.filter(pl.col("column") == "x")["count"][0] == 0
    assert out.filter(pl.col("column") == "y")["mean"][0] == 3.0
    xray_quietly(lf, include="all")  # renders
    empty = xray_quietly(lf.head(0), include="all", great_tables=False)
    assert empty["count"].to_list() == [0, 0, 0, 0]


def test_correlation_against_a_target_outside_the_analyzed_columns():
    lf = pl.LazyFrame({"s": ["a", "b", "c"] * 10, "t": [float(i) for i in range(30)]})
    out = xray_quietly(lf, include="string", corr_target="t", great_tables=False)
    assert "correlation" not in out.columns
    out = xray_quietly(
        lf.with_columns(u=pl.col("t") * -2), corr_target="t", great_tables=False
    )
    assert out.filter(pl.col("column") == "u")["correlation"][0] == -1.0


def test_reads_only_analyzed_columns():
    def boom(_value: int) -> str:
        raise RuntimeError("an unanalyzed column was evaluated")

    lf = pl.LazyFrame({"x": [1, 2, 3, 4]}).with_columns(
        never=pl.col("x").map_elements(boom, return_dtype=pl.String)
    )
    assert xray_quietly(lf, great_tables=False)["column"].to_list() == ["x"]


def test_scan_parquet(tmp_path):
    df = _mixed_frame().drop("lst", "dur")
    path = tmp_path / "data.parquet"
    df.write_parquet(path, row_group_size=7)
    from_file = xray_quietly(
        pl.scan_parquet(path), include="all", expanded=True, great_tables=False
    )
    from_memory = xray_quietly(df.lazy(), include="all", expanded=True, great_tables=False)
    for column in from_file.columns:
        for a, b in zip(from_file[column].to_list(), from_memory[column].to_list()):
            assert _same(a, b), (column, a, b)


def test_dates_and_categoricals():
    start = date(2020, 1, 1)
    lf = pl.LazyFrame(
        {
            "d": [start + timedelta(days=i % 30) for i in range(120)],
            "c": pl.Series([f"c{i % 3}" for i in range(120)], dtype=pl.Categorical),
        }
    )
    out = xray_quietly(lf, include="all", great_tables=False)
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    assert (rows["d"]["earliest"], rows["d"]["latest"]) == ("2020-01-01", "2020-01-30")
    assert (rows["c"]["top"], rows["c"]["top_freq"]) == ("c0", 40)


def test_infinite_values():
    # The z-score fences come from the finite values; inf is an outlier.
    values = [float(i % 13) for i in range(200)] + [math.inf, -math.inf, 1e6]
    df = pl.DataFrame({"x": values, "y": [float(i) for i in range(203)]})
    for kwargs in ({}, {"expanded": True}):
        exact = xray_quietly(df, great_tables=False, outlier_method="zscore", **kwargs)
        streamed = xray_quietly(df.lazy(), great_tables=False, **kwargs)
        for column in streamed.columns:
            for a, b in zip(exact[column].to_list(), streamed[column].to_list()):
                assert _same(a, b), (column, a, b)
        assert streamed["n_outliers"][0] == 3


@pytest.mark.parametrize("frame", FRAMES)
def test_batched_group_counts_equal_the_plain_ones(monkeypatch, frame):
    # Above _GROUP_BY_MAX_DISTINCT distinct values, distinct counts and string
    # value counts merge per-batch results. Forced here with
    # tiny batches and frequent merges; the result must not change.
    df = FRAMES[frame]()
    kwargs = {"include": "all", "expanded": True, "great_tables": False}
    plain_result = xray_quietly(df.lazy(), **kwargs)
    monkeypatch.setattr(xray_mod, "_GROUP_BY_MAX_DISTINCT", 0)
    monkeypatch.setattr(xray_mod, "_BATCH_ROWS", 7)
    monkeypatch.setattr(xray_mod, "_MERGE_ROWS", 5)
    batched = xray_quietly(df.lazy(), **kwargs)
    assert batched.equals(plain_result)
