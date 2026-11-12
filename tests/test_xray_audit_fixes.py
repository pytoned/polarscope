"""Regression tests for the xray() findings of the code audit (PS-xx IDs)."""

from __future__ import annotations

import importlib
import math
import re
from datetime import date, datetime, timedelta

import polars as pl
import pytest

import polarscope as ps

xray_mod = importlib.import_module("polarscope.xray")
NAN = float("nan")


def plain(frame, **kwargs) -> pl.DataFrame:
    # These regression tests compare full summaries: LazyFrames get the exact
    # one (the streaming summary has its own tests, test_lazyframe_streaming).
    if isinstance(frame, pl.LazyFrame):
        kwargs.setdefault("exact", True)
    return ps.xray(frame, great_tables=False, **kwargs)


@pytest.fixture(params=["DataFrame", "LazyFrame"])
def as_input(request):
    """Run a test with DataFrame and with LazyFrame input."""
    return (lambda df: df) if request.param == "DataFrame" else (lambda df: df.lazy())


class TestCrashes:
    def test_column_named_count(self, as_input):
        # PS-02: the default output of value_counts()/group_by().len().
        vc = pl.DataFrame({"city": ["Oslo", "Bergen", "Oslo", "Oslo"]})["city"].value_counts()
        out = plain(as_input(vc), include="all", expanded=True)
        assert out.filter(pl.col("column") == "count")["mean"][0] == 2.0
        dates = pl.DataFrame({"count": [date(2020, 1, 1), date(2020, 1, 2)]})
        assert plain(as_input(dates), include="all")["earliest"][0] == "2020-01-01"

    def test_more_than_100_columns_of_mixed_kinds(self, as_input):
        # PS-03: the summary schema used to be inferred from the first 100 rows.
        wide = pl.DataFrame({**{f"s{i}": ["a", "b"] for i in range(101)}, "f": [1.5, 9.5]})
        assert plain(as_input(wide), include="all").height == 102
        wide = pl.DataFrame({**{f"n{i}": [1, 2] for i in range(101)}, "s": ["x", "y"]})
        assert plain(as_input(wide), include="all").height == 102

    def test_object_and_array_columns(self, as_input):
        # PS-10: Polars panics are not Exceptions, so they escaped the guard.
        df = pl.DataFrame(
            {
                "o": pl.Series([object(), object()], dtype=pl.Object),
                "a": pl.Series([[1, 2], [1, 2]], dtype=pl.Array(pl.Int64, 2)),
                "x": [1, 2],
            }
        )
        frame = df if isinstance(as_input(df), pl.DataFrame) else df.drop("o").lazy()
        out = plain(frame, include="all", expanded=True)
        assert out.filter(pl.col("column") == "a")["n_unique"][0] == 1
        if "o" in out["column"]:
            assert out.filter(pl.col("column") == "o")["n_unique"][0] == 2

    @pytest.mark.parametrize("name", ["*", "^a.*$"])
    def test_pattern_like_column_names(self, as_input, name):
        # PS-26: pl.col() reads these as a wildcard / regular expression.
        df = pl.DataFrame({"abc": [10.0, 20.0, 35.0], name: [1.0, 2.0, 3.0]})
        out = plain(as_input(df), corr_target=name)
        row = out.filter(pl.col("column") == name).row(0, named=True)
        assert row["mean"] == 2.0
        assert out.filter(pl.col("column") == "abc")["mean"][0] == pytest.approx(21.67)


class TestDecimal:
    def test_money_like_decimals_with_inexact_outlier_fences(self, as_input):
        # 1.9.6 compared the Decimal column with float fences such as -0.2,
        # whose exact decimal expansion has ~54 digits: "ComputeError:
        # overflow in Python Decimal to Polars Decimal conversion". Decimal
        # columns are now analyzed as Float64.
        from decimal import Decimal

        values = ["0.10", "0.20", "0.30", "0.40", "9.90"]
        df = pl.DataFrame(
            {"price": pl.Series([Decimal(v) for v in values], dtype=pl.Decimal(38, 2))}
        )
        for method in ("iqr", "zscore"):
            out = plain(as_input(df), expanded=True, outlier_method=method)
            assert out.select("mean", "min", "max").row(0) == (2.18, 0.1, 9.9)
        assert plain(as_input(df))["n_outliers"][0] == 1
        assert ps.fix(df, outliers="iqr", verbose=False)["price"].null_count() == 1

    def test_price_columns_of_both_precisions(self, as_input):
        # Whole-number prices give exact IQR fences but inexact z-score fences;
        # 1.9.6 failed on the latter for decimal[10,2] and decimal[38,2] alike.
        from decimal import Decimal

        old = ["20.00", "150.00", "0.00", "150.00", "80.00", "250.00", "40.00"]
        new = ["25.00", "175.00", "0.00", "175.00", "95.00", "290.00", "55.00"]
        df = pl.DataFrame(
            {
                "list_price": pl.Series([Decimal(v) for v in old], dtype=pl.Decimal(10, 2)),
                "sale_price": pl.Series([Decimal(v) for v in new], dtype=pl.Decimal(38, 2)),
            }
        )
        for method in ("iqr", "zscore"):
            out = plain(as_input(df), outlier_method=method)
            assert out["max"].to_list() == [250.0, 290.0]
            assert out["n_outliers"].to_list() == [0, 0]


class TestNaN:
    def test_nan_is_missing_not_a_value(self, as_input):
        # PS-05: one NaN made the mean NaN and NaN fences flagged every value.
        df = pl.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, NAN, NAN, NAN]})
        row = plain(as_input(df)).row(0, named=True)
        assert row["count"] == 7
        assert row["nan_count"] == 3
        assert row["pct_missing"] == 30.0
        assert row["mean"] == 4.0
        assert row["50%"] == 4.0
        assert row["max"] == 7.0
        assert row["n_outliers"] == 0

    def test_zscore_with_one_nan(self, as_input):
        df = pl.DataFrame({"x": [float(i) for i in range(99)] + [NAN]})
        assert plain(as_input(df), outlier_method="zscore")["n_outliers"][0] == 0

    def test_nan_is_not_a_positive_value(self):
        out = plain(pl.DataFrame({"x": [-1.0, -2.0, NAN, NAN]}), expanded=True)
        assert out.select("pct_pos", "pct_neg").row(0) == (0.0, 100.0)

    def test_nan_count_column_only_when_needed(self):
        assert "nan_count" not in plain(pl.DataFrame({"x": [1.0, 2.0]})).columns
        assert "nan_count" in plain(pl.DataFrame({"x": [1.0, NAN]})).columns

    def test_infinite_values_are_outliers_without_poisoning_fences(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, math.inf]
        assert plain(pl.DataFrame({"x": values}))["n_outliers"][0] == 1
        assert plain(pl.DataFrame({"x": values}), outlier_method="zscore")["n_outliers"][0] == 1


class TestStatistics:
    def test_quantiles_are_linear(self, as_input):
        # PS-14: "nearest" made the 50% column differ from the median.
        out = plain(as_input(pl.DataFrame({"x": list(range(1, 11))})))
        assert out.select("25%", "50%", "75%").row(0) == (3.25, 5.5, 7.75)
        assert plain(as_input(pl.DataFrame({"x": [10, 20]})))["50%"][0] == 15.0

    def test_integer_min_max_are_exact(self, as_input):
        # PS-30: floats cannot tell these two values apart.
        df = pl.DataFrame({"x": [1234567890123456789, 1234567890123456788]})
        out = plain(as_input(df))
        assert out.select("min", "max").row(0) == (1234567890123456788, 1234567890123456789)

    def test_min_max_fall_back_to_float_with_float_columns(self):
        out = plain(pl.DataFrame({"i": [1, 2], "f": [0.5, 1.5]}))
        assert out.schema["min"] == pl.Float64

    def test_cardinality_is_exact_on_large_frames(self):
        # PS-41: the approximation could exceed the number of values.
        n = 300_001
        out = plain(pl.DataFrame({"x": pl.int_range(n, eager=True)}), expanded=True)
        assert out.select("n_unique", "uniqueness_ratio", "n_duplicates").row(0) == (n, 1.0, 0)

    def test_boolean_correlation_target(self, as_input):
        # PS-46: Booleans are analyzed as 0/1, so they work as a target too.
        df = pl.DataFrame({"t": [True, False, True, False], "x": [1.0, 2.0, 3.0, 5.0]})
        corr = plain(as_input(df), corr_target="t", decimals=12)["correlation"][0]
        expected = df.select(pl.corr(pl.col("t").cast(pl.UInt8), "x")).item()
        assert corr == pytest.approx(expected)

    def test_spearman_target_matches_polars(self):
        # PS-34: the target is ranked once; results must equal pl.corr.
        df = pl.DataFrame(
            {
                "t": [5, 3, 3, 9, 1, 7, 7, 2],
                "a": [1.0, 4.0, 4.0, 2.0, 8.0, 0.5, 3.0, 6.0],
                "b": [1, None, 2, 3, 4, 5, 6, 7],
            }
        )
        out = plain(df, corr_target="t", corr_method="spearman", decimals=12)
        for c in ("a", "b"):
            expected = df.select(pl.corr("t", c, method="spearman")).item()
            assert out.filter(pl.col("column") == c)["correlation"][0] == pytest.approx(expected)

    def test_all_null_numeric_label(self):
        out = plain(
            pl.DataFrame({"a": pl.Series([None] * 5, dtype=pl.Float64), "b": [1.0, 2, 3, 4, 5]}),
            expanded=True,
        )
        assert out["normality_test"][0] == "N/A (no values)"


class TestQualityFlags:
    def test_dates_and_timestamps_are_not_ids(self):
        # PS-13: near-unique temporal columns were flagged "Drop - likely an ID".
        n = 365
        df = pl.DataFrame(
            {
                "order_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
                "ts": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)],
                "order_id": list(range(n)),
            }
        )
        out = plain(df, include="all", expanded=True, model_usability=True)
        flags = dict(zip(out["column"], out["usability_flags"]))
        assert "ID" not in flags["order_date"]
        assert "ID" not in flags["ts"]
        assert "ID" in flags["order_id"]

    def test_opt_dtype_matches_fix(self):
        # PS-24/PS-25: opt_dtype is what fix()/convert_datatypes produces.
        df = pl.DataFrame(
            {
                "s": [f"v{i % 25}" for i in range(100)],
                "flag": [True, False] * 50,
                "small": [i % 2 for i in range(100)],
                "f": [1.5, 0.25] * 50,
                "p": [0.1, 0.123456789] * 50,
            }
        )
        out = plain(df, include="all", expanded=True)
        suggested = dict(zip(out["column"], out["opt_dtype"]))
        fixed = ps.convert_datatypes(df)
        for c in ("s", "flag", "small", "f", "p"):
            assert suggested[c] == str(fixed.schema[c]), c

    def test_categorical_label_is_version_independent(self):
        df = pl.DataFrame({"c": pl.Series(["a", "b", "a"], dtype=pl.Categorical)})
        out = plain(df, include="all", expanded=True)
        assert out.select("dtype", "opt_dtype").row(0) == ("Categorical", "Categorical")


class TestHtml:
    def test_user_text_is_escaped_exactly_once(self):
        # PS-09: Great Tables 0.x writes plain string cells into the HTML as
        # they are; 1.x escapes them itself, so xray() must not escape twice.
        df = pl.DataFrame({"status": ["<NA>"] * 3 + ["ok"], "<b>x</b>": [1, 2, 3, 4]})
        html = ps.xray(df, include="all", title="<i>t</i>", corr_target="<b>x</b>").as_raw_html()
        cells = re.findall(r"<td[^>]*>(.*?)</td>", html, re.S)
        assert "&lt;NA&gt;" in cells  # the top value of "status"
        assert "&lt;b&gt;x&lt;/b&gt;" in cells  # the column name
        assert "<NA>" not in html
        assert "&amp;lt;NA" not in html
        assert "&lt;i&gt;t&lt;/i&gt;" in html  # title, escaped by Great Tables
        assert "&amp;lt;i&amp;gt;" not in html

    def test_plain_output_is_not_escaped(self):
        df = pl.DataFrame({"status": ["<NA>", "<NA>", "ok"]})
        assert plain(df, include="all")["top"][0] == "<NA>"


class TestTests:
    def test_model_usability_alone_skips_uniformity(self, monkeypatch):
        # PS-16: the uniformity result was computed and thrown away.
        calls = []
        original = xray_mod._test_uniformity
        monkeypatch.setattr(
            xray_mod, "_test_uniformity", lambda *a, **k: calls.append(a) or original(*a, **k)
        )
        plain(pl.DataFrame({"x": [float(i % 17) for i in range(200)]}), model_usability=True)
        assert calls == []

    def test_large_and_small_paths_agree(self):
        # Above 5,000 values the KS uniformity statistic comes from Polars and
        # normality uses the stride sample; both must match SciPy on all values.
        if not xray_mod._check_scipy_availability():
            pytest.skip("scipy not installed")
        values = [(i * 7919) % 6007 / 7.0 for i in range(6000)]
        out = plain(pl.DataFrame({"x": values}), expanded=True)
        stats = xray_mod.stats
        import numpy as np

        arr = np.asarray(values)
        normalized = (arr - arr.min()) / (arr.max() - arr.min())
        expected = stats.kstest(normalized, "uniform", method="asymp").pvalue
        assert out["uniformity_test"][0] == xray_mod._uniformity_result(expected)
        assert out["normality_test"][0] == xray_mod._test_normality(values, "shapiro")


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


class TestExecutionPaths:
    """The large-data code paths, forced on small frames via the thresholds."""

    CONFIGS = [
        {"include": "all"},
        {"include": "all", "expanded": True, "model_usability": True},
        {"corr_target": "target"},
        {"corr_target": "target", "corr_method": "spearman", "expanded": True},
        {"outlier_method": "zscore", "expanded": True},
        {"include": "all", "expanded": True, "uniformity_test": "chi2", "normality_test": "anderson"},
    ]

    @pytest.mark.parametrize("config", CONFIGS)
    @pytest.mark.parametrize("columns_per_batch", [1, 2, 4])
    def test_lazyframe_collected_in_column_batches(
        self, monkeypatch, config, columns_per_batch
    ):
        # A LazyFrame above _COLLECT_CELLS cells is collected a batch of
        # columns at a time; the corr_target column is read once and joins
        # every batch, including one that holds the target itself.
        df = _mixed_with_nan()
        expected = plain(df, **config)
        monkeypatch.setattr(xray_mod, "_COLLECT_CELLS", df.height * columns_per_batch)
        assert plain(df.lazy(), **config).equals(expected)

    @pytest.mark.parametrize("config", CONFIGS)
    def test_large_frames_run_queries_one_at_a_time(self, monkeypatch, config):
        # Above _LARGE_DATASET_ROWS queries run sequentially (streamable ones on
        # the streaming engine) and the KS statistic gets a query of its own;
        # only the quasi-constant mode share is skipped there.
        df = _mixed_with_nan()
        skipped = ["shakiness_score", "quality_flag", *xray_mod._USABILITY_COLS]
        expected = plain(df, **config)
        monkeypatch.setattr(xray_mod, "_LARGE_DATASET_ROWS", 0)
        for frame in (df, df.lazy()):
            out = plain(frame, **config)
            keep = [c for c in expected.columns if c not in skipped]
            assert out.select(keep).equals(expected.select(keep))


class TestBoolean:
    @pytest.mark.parametrize("method", ["iqr", "zscore", "percentile"])
    def test_booleans_have_no_outliers(self, as_input, method):
        # With 12% True both quartiles are 0, so the IQR fences called every
        # True an outlier - and that pushed the column towards "SHAKY".
        df = pl.DataFrame({"churned": [i % 25 < 3 for i in range(400)]})
        bounds = [0.05, 0.95] if method == "percentile" else None
        out = plain(
            as_input(df),
            include="all",
            expanded=True,
            outlier_method=method,
            outlier_bounds=bounds,
        ).row(0, named=True)
        assert out["n_outliers"] is None and out["pct_outliers"] is None
        assert out["mean"] == 0.12
        # Only the skew (2.34 > 2) counts against it now.
        assert (out["shakiness_score"], out["quality_flag"]) == (1, "✓ OK")


class TestNormalityPValues:
    # Reference p-values from statsmodels' lilliefors(pvalmethod="approx"),
    # which implements the same Dallal-Wilkinson approximation independently.
    @pytest.mark.parametrize(
        "values, expected",
        [
            ([float(k * k) for k in range(1, 61)], 0.011342398180722019),
            (
                [float((i * 37) % 101) + 0.002 * ((i * 37) % 101) ** 2 for i in range(400)],
                0.00015602346913996676,
            ),
            ([-math.log(1 - (k + 0.5) / 3000) for k in range(3000)], 4.4537207017014385e-210),
        ],
    )
    def test_lilliefors_p_value_matches_reference(self, values, expected):
        if not xray_mod._check_scipy_availability():
            pytest.skip("scipy not installed")
        import numpy as np

        p = xray_mod._lilliefors_p_value(np.asarray(values))
        assert p == pytest.approx(expected, rel=1e-9)

    def test_large_columns_resolve_small_p_values(self):
        # The Monte Carlo p-value it replaces could not go below 1/200, so a
        # plainly skewed 20M-row column read "NON-NORMAL (p=0.005)".
        if not xray_mod._check_scipy_availability():
            pytest.skip("scipy not installed")
        skewed = [-math.log(1 - (k + 0.5) / 20_000) for k in range(20_000)]
        df = pl.DataFrame({"x": skewed})
        for test in ("shapiro", "ks"):
            out = plain(df, expanded=True, normality_test=test)
            assert out["normality_test"][0] == "NON-NORMAL (p<0.001)"

    def test_normal_data_keeps_large_p_values(self):
        if not xray_mod._check_scipy_availability():
            pytest.skip("scipy not installed")
        import numpy as np

        values = xray_mod.stats.norm.ppf((np.arange(6000) + 0.5) / 6000)
        assert xray_mod._test_normality(values, "ks") == "NORMAL (p=1.000)"
        assert xray_mod._test_normality([1.0, 2.0, 3.0, 5.0], "ks") == "N/A (insufficient data)"

    def test_anderson_table_bounds_are_shown_as_bounds(self):
        # SciPy clips its interpolated Anderson-Darling p-value to 1%-15%.
        if not xray_mod._check_scipy_availability():
            pytest.skip("scipy not installed")
        import inspect

        if "method" not in inspect.signature(xray_mod.stats.anderson).parameters:
            pytest.skip("this SciPy reports no Anderson-Darling p-value")
        skewed = [-math.log(1 - (k + 0.5) / 5000) for k in range(5000)]
        assert xray_mod._test_normality(skewed, "anderson") == "NON-NORMAL (p≤0.010)"
        normal = xray_mod.stats.norm.ppf([(k + 0.5) / 500 for k in range(500)])
        assert xray_mod._test_normality(normal, "anderson") == "NORMAL (p≥0.150)"


class TestDisplay:
    def test_missing_cells_show_a_dash(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 4.0, 8.0],
                "s": ["a", "b", "a", "c"],
                "d": [date(2020, 1, k) for k in range(1, 5)],
            }
        )
        for kwargs in ({}, {"expanded": True}, {"corr_target": "x"}):
            html = ps.xray(df, include="all", **kwargs).as_raw_html()
            cells = re.findall(r"<td[^>]*>(.*?)</td>", html, re.S)
            assert "None" not in html
            assert cells.count("-") >= 10
        # The plain DataFrame keeps real nulls.
        assert plain(df, include="all")["mean"].null_count() == 2

    @pytest.mark.parametrize(
        "dtype, label",
        [
            (pl.Decimal(10, 2), "Decimal(10, 2)"),
            (pl.Decimal(38, 2), "Decimal(38, 2)"),
            (pl.Datetime("us"), "Datetime(us)"),
            (pl.Datetime("ms", "Europe/Oslo"), "Datetime(ms, Europe/Oslo)"),
            (pl.Duration("ns"), "Duration(ns)"),
            (pl.List(pl.Datetime("us")), "List(Datetime(us))"),
            (pl.Array(pl.Float64, (2, 3)), "Array(Float64, 2x3)"),
            (pl.Struct({"a": pl.Int64, "b": pl.String}), "Struct(2 fields)"),
            (pl.Enum(["lo", "hi"]), "Enum(2 categories)"),
            (pl.Categorical, "Categorical"),
            (pl.Int64, "Int64"),
        ],
    )
    def test_short_dtype_labels(self, dtype, label):
        assert xray_mod._dtype_label(dtype) == label

    def test_dtype_and_opt_dtype_columns_use_short_labels(self):
        from decimal import Decimal

        df = pl.DataFrame(
            {
                "price": pl.Series([Decimal("20.00"), Decimal("25.50")], dtype=pl.Decimal(38, 2)),
                "ts": pl.Series([datetime(2024, 1, 1), datetime(2024, 1, 2)], dtype=pl.Datetime("us")),
            }
        )
        out = plain(df, include="all", expanded=True)
        assert out.select("dtype", "opt_dtype").rows() == [
            ("Decimal(38, 2)", "Decimal(38, 2)"),
            ("Datetime(us)", "Datetime(us)"),
        ]


class TestTinyRanges:
    @pytest.mark.parametrize("values", [[0.0, 5e-324], [-1e-310, 1e-310], [0.0, 1e-320, 3e-320]])
    def test_subnormal_ranges_are_one_bar(self, values):
        # Found by the property tests: n_bins / range overflowed to inf, and
        # the histogram cast failed ("conversion from f64 to u8 failed").
        df = pl.DataFrame({"x": values})
        for frame, kwargs in ((df, {}), (df.lazy(), {"exact": True}), (df.lazy(), {})):
            plot = ps.xray(frame, **kwargs)._tbl_data["distribution_plot"][0]
            assert list(plot) == [len(values)]
