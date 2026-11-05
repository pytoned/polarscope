"""Focused unit tests for internal logic and edge branches."""

from __future__ import annotations

import builtins
from datetime import datetime
import importlib
import sys
import types
import warnings

import numpy as np
import polars as pl
import pytest


clean_mod = importlib.import_module("polarscope.clean")
datasets_mod = importlib.import_module("polarscope.datasets")
plots_mod = importlib.import_module("polarscope.plots")
utils_mod = importlib.import_module("polarscope.utils")
xray_mod = importlib.import_module("polarscope.xray")


def test_clean_normalize_and_dedupe_paths() -> None:
    assert clean_mod._normalize("  naïve café ", ascii_only=True) == "naive_cafe"
    assert clean_mod._normalize("MiXeD Name", case="upper", ascii_only=False) == "MIXED_NAME"

    df = pl.DataFrame({"A A": [1], "A-A": [2], "A_A": [3], "": [4]})
    cleaned = clean_mod.clean_column_names(df, dedupe=True)
    assert cleaned.columns == ["a_a", "a_a_1", "a_a_2", "col"]

    with pytest.raises(pl.exceptions.DuplicateError):
        clean_mod.clean_column_names(df, dedupe=False, case="upper")

    # Regression: a deduped name must not collide with a pre-existing column.
    collide = pl.DataFrame({"A": [1], "a": [2], "a_1": [3]})
    cleaned_collide = clean_mod.clean_column_names(collide, dedupe=True)
    assert cleaned_collide.columns == ["a", "a_1", "a_1_1"]
    assert len(set(cleaned_collide.columns)) == cleaned_collide.width


def test_datasets_loaders_and_info() -> None:
    names = datasets_mod.list_datasets()
    assert set(names) == {"titanic", "diabetes", "cardio"}

    titanic_path = datasets_mod.load_titanic(return_polars=False)
    diabetes_path = datasets_mod.load_diabetes(return_polars=False)
    cardio_path = datasets_mod.load_cardio(return_polars=False)
    assert titanic_path.endswith("titanic.parquet")
    assert diabetes_path.endswith("diabetes.parquet")
    assert cardio_path.endswith("cardio.parquet")

    titanic_df = datasets_mod.load_titanic()
    diabetes_df = datasets_mod.load_diabetes()
    cardio_df = datasets_mod.load_cardio()
    assert isinstance(titanic_df, pl.DataFrame)
    assert isinstance(diabetes_df, pl.DataFrame)
    assert isinstance(cardio_df, pl.DataFrame)
    assert titanic_df.height > 0
    assert diabetes_df.height > 0
    assert cardio_df.height > 0

    # Top-level convenience aliases load the shipped data.
    import polarscope as ps
    assert ps.titanic().height == titanic_df.height
    assert ps.diabetes().height == diabetes_df.height
    assert ps.cardio().height == cardio_df.height

    assert "Titanic Dataset" in datasets_mod.dataset_info("titanic")
    assert "Diabetes Dataset" in datasets_mod.dataset_info("diabetes")
    assert "Cardiovascular Disease Dataset" in datasets_mod.dataset_info("cardio")
    with pytest.raises(ValueError):
        datasets_mod.dataset_info("unknown")


def test_utils_save_fig_branches_and_base64() -> None:
    class DummyMatplotlibFig:
        def __init__(self) -> None:
            self.saved_args = None

        def savefig(self, *args, **kwargs) -> None:
            self.saved_args = (args, kwargs)

    dummy_mpl = DummyMatplotlibFig()
    utils_mod.save_fig(dummy_mpl, "/tmp/out.png")
    assert dummy_mpl.saved_args is not None
    assert dummy_mpl.saved_args[1]["dpi"] == 300

    DummyPlotly = type("DummyPlotly", (), {})
    DummyPlotly.__module__ = "plotly.graph_objs"

    class PlotlyObj(DummyPlotly):  # type: ignore[misc]
        def __init__(self) -> None:
            self.html_called = False
            self.image_called = False

        def write_html(self, path: str, include_plotlyjs: str = "cdn") -> None:
            self.html_called = path.endswith(".html") and include_plotlyjs == "cdn"

        def write_image(self, path: str, scale: float = 1.0) -> None:
            self.image_called = path.endswith(".png") and scale == 2.0

    PlotlyObj.__module__ = "plotly.graph_objs"
    plotly_obj = PlotlyObj()
    utils_mod.save_fig(plotly_obj, "/tmp/out.html")
    utils_mod.save_fig(plotly_obj, "/tmp/out.png", scale=2.0)
    assert plotly_obj.html_called
    assert plotly_obj.image_called

    class DummyAltairChart:
        def __init__(self) -> None:
            self.saved = None

        def save(self, path: str) -> None:
            self.saved = path

    alt = DummyAltairChart()
    utils_mod.save_fig(alt, "/tmp/out.json")
    assert alt.saved == "/tmp/out.json"


def test_xray_quasi_constant_and_distribution_validation() -> None:
    import polarscope as ps

    # A value covering >= constant_threshold of the column is flagged quasi-constant.
    df_const = pl.DataFrame({"x": [1] * 99 + [2]})
    out = ps.xray(df_const, include="all", expanded=True, great_tables=False)
    assert out["quality_flag"][0] == "⚠ SHAKY"

    # The histogram is the only distribution plot, so there is no knob to turn.
    with pytest.raises(TypeError):
        ps.xray(df_const, distribution_plot="kde")

    # The rendered histogram column itself is unaffected by removing the argument.
    html = ps.xray(df_const, include="all", expanded=True).as_raw_html()
    assert "distribution_plot" in html


def test_xray_model_usability_with_non_numeric_columns() -> None:
    """Regression: model_usability + non-numeric columns must not raise."""
    import polarscope as ps

    df = pl.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "txt": ["a", "b", "a", "c"]})
    out = ps.xray(
        df, include="all", expanded=True, model_usability=True,
        corr_target="num", great_tables=False,
    )
    assert "usability_score" in out.columns
    assert out.height == 2


def test_xray_raises_on_no_matching_columns() -> None:
    """Regression: default include on a string-only frame must raise clearly."""
    import polarscope as ps

    df = pl.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
    with pytest.raises(ValueError, match="include='all'"):
        ps.xray(df)
    with pytest.raises(ValueError, match="No columns matched"):
        ps.xray(df, include="temporal")
    with pytest.raises(ValueError):
        ps.xray(pl.DataFrame())


def test_xray_usability_columns_in_minimal_mode() -> None:
    """model_usability=True appends its columns in minimal mode too."""
    import polarscope as ps

    df = pl.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "txt": ["a", "b", "a", "c"]})
    out = ps.xray(df, include="all", model_usability=True, great_tables=False)
    for col in ["usability_flags", "usability_score", "recommendation"]:
        assert col in out.columns
    # GT rendering with usability columns in minimal mode must not raise
    ps.xray(df, include="all", model_usability=True)


def test_model_usability_scores_match_recommendations_and_display_modes() -> None:
    import polarscope as ps

    constant = ps.xray(
        pl.DataFrame({"constant": [1] * 20}),
        model_usability=True,
        great_tables=False,
    ).row(0, named=True)
    assert constant["recommendation"] == "Drop - constant value"
    assert constant["usability_score"] <= 20

    binary = pl.DataFrame({"feature": [0.0, 1.0] * 50})
    minimal = ps.xray(
        binary,
        model_usability=True,
        great_tables=False,
    ).row(0, named=True)
    expanded = ps.xray(
        binary,
        expanded=True,
        model_usability=True,
        great_tables=False,
    ).row(0, named=True)
    assert minimal["usability_flags"] == expanded["usability_flags"]
    assert minimal["usability_score"] == expanded["usability_score"]


def test_id_detection_uses_name_tokens_and_dtype() -> None:
    common = {
        "n_unique": 200,
        "count": 1000,
        "null_count": 0,
        "uniqueness_ratio": 0.2,
        "dtype": "Int64",
    }
    assert not xray_mod._is_likely_id_column("humidity", common)
    assert xray_mod._is_likely_id_column("customer_id", common)
    assert xray_mod._is_likely_id_column("customerId", common)

    unique_float = {
        **common,
        "n_unique": 1000,
        "uniqueness_ratio": 1.0,
        "dtype": "Float64",
    }
    assert not xray_mod._is_likely_id_column("measurement", unique_float)


def test_balanced_low_cardinality_column_is_not_quasi_constant() -> None:
    score = xray_mod._calculate_shakiness_score(
        {
            "pct_missing": 0.0,
            "n_unique": 2,
            "uniqueness_ratio": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "pct_outliers": 0.0,
            "normality_test": "",
        },
        missing_threshold=0.3,
        constant_threshold=0.99,
        skew_threshold=2.0,
        kurtosis_threshold=7.0,
        outlier_threshold=0.05,
        mode_share=0.5,
    )
    assert score == 0


def test_xray_categorical_boolean_temporal_coverage() -> None:
    """Categorical/Enum get string stats; Boolean gets 0/1 numeric stats; temporal gets min/max."""
    import polarscope as ps

    df = pl.DataFrame({
        "cat": pl.Series(["a", "b", "a", "a"], dtype=pl.Categorical),
        "enum": pl.Series(["x", "y", "x", "y"], dtype=pl.Enum(["x", "y"])),
        "flag": [True, False, True, True],
        "when": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 4), "1d", eager=True),
    })

    out = ps.xray(df, include="all", expanded=True, great_tables=False).to_dicts()
    by_col = {r["column"]: r for r in out}

    # Categorical/Enum are analyzed as strings
    assert by_col["cat"]["top"] == "a" and by_col["cat"]["top_freq"] == 3
    assert by_col["enum"]["top_freq"] == 2

    # Boolean is analyzed as 0/1 numeric: Mean = share of True
    assert by_col["flag"]["mean"] == 0.75
    assert by_col["flag"]["min"] == 0.0 and by_col["flag"]["max"] == 1.0

    # Temporal columns report earliest/latest
    assert by_col["when"]["earliest"].startswith("2024-01-01")
    assert by_col["when"]["latest"].startswith("2024-01-04")

    # include='string' matches Categorical/Enum
    string_cols = ps.xray(df, include="string", great_tables=False)["column"].to_list()
    assert set(string_cols) == {"cat", "enum"}

    # GT rendering of the mixed frame must not raise in either mode
    ps.xray(df, include="all")
    ps.xray(df, include="all", expanded=True)


def test_xray_null_aware_duplicate_semantics() -> None:
    """Uniqueness/duplicates are computed among non-null values only."""
    import polarscope as ps

    # 4 valid all-unique values + 4 nulls: no duplicates, ratio 1.0
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None, None, None, None]})
    row = ps.xray(df, expanded=True, great_tables=False).to_dicts()[0]
    assert row["n_duplicates"] == 0
    assert row["uniqueness_ratio"] == 1.0


def test_normality_ks_fallback_for_large_samples() -> None:
    if not xray_mod._check_scipy_availability():
        pytest.skip("scipy not available")
    data = np.random.default_rng(0).normal(size=6000)
    result = xray_mod._test_normality(data, "shapiro")
    # Truly normal data must pass even via the large-sample KS fallback,
    # and the cell text is the compact "VERDICT (p=...)" format.
    assert result.startswith("NORMAL (p")


def test_anderson_normality_avoids_scipy_future_warning() -> None:
    if not xray_mod._check_scipy_availability():
        pytest.skip("scipy not available")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = xray_mod._test_normality([1.0, 2.0, 3.0, 4.0, 5.0], "anderson")

    assert result.startswith(("NORMAL", "NON-NORMAL"))


def test_statistical_test_cells_are_compact() -> None:
    if not xray_mod._check_scipy_availability():
        pytest.skip("scipy not available")
    rng = np.random.default_rng(0)

    # Tiny p-values render as "p<0.001" instead of a misleading "p=0.000".
    skewed = rng.exponential(1.0, size=2000)
    assert xray_mod._test_normality(skewed, "shapiro") == "NON-NORMAL (p<0.001)"
    assert xray_mod._test_uniformity(skewed, "ks") == "NON-UNIFORM (p<0.001)"

    # Truly uniform data passes the uniformity test at realistic sample sizes.
    uniform = rng.uniform(0.0, 1.0, size=2000)
    result = xray_mod._test_uniformity(uniform, "ks")
    assert result.startswith("UNIFORM (p")


def test_expanded_column_group_order() -> None:
    import polarscope as ps

    rng = np.random.default_rng(0)
    df = pl.DataFrame({
        "a": rng.exponential(1.0, 200),
        "b": rng.normal(0.0, 1.0, 200),
    })

    cols = ps.xray(df, expanded=True, great_tables=False, corr_target="b").columns
    # Plot payloads are Great Tables-only; plain output must not carry them.
    assert "distribution_plot" not in cols
    assert "correlation_plot" not in cols
    # Basic statistics lead, with opt_dtype right after dtype.
    assert cols[:8] == ["column", "dtype", "opt_dtype", "count", "mean", "std", "min", "max"]
    # Group order: quantiles, counts, outliers, distribution, correlation,
    # tests, quality assessment.
    assert cols[-4:] == ["normality_test", "uniformity_test", "shakiness_score", "quality_flag"]
    assert (
        cols.index("25%")
        < cols.index("null_count")
        < cols.index("n_outliers")
        < cols.index("mad")
        < cols.index("correlation")
        < cols.index("normality_test")
    )


def test_compact_count_honors_decimals() -> None:
    """Compact integer columns keep decimal precision (12,345 -> 12.35K)."""
    import polarscope as ps

    df = pl.DataFrame({"x": [float(i) for i in range(12345)]})
    html = ps.xray(df, compact=True, decimals=2).as_raw_html()
    assert "12.35K" in html


def test_execution_time_formatting() -> None:
    assert xray_mod._format_execution_time(870) == "870 ms"
    assert xray_mod._format_execution_time(12219) == "12.2 s"
    assert xray_mod._format_execution_time(93000) == "1.6 min"


def test_string_stats_and_numeric_column_hiding() -> None:
    import polarscope as ps

    df = pl.DataFrame({"prov": ["a", "bb", "ccc", "a", "a"] * 20, "val": list(range(100))})

    # Expanded string view exposes the richer string stats...
    out = ps.xray(df, include="string", expanded=True, great_tables=False)
    for col in ["min_length", "median_length", "max_length", "mode_share", "top_3", "sample_vals"]:
        assert col in out.columns
    # ...and drops numeric-only stats entirely when no numeric column is analyzed.
    for col in ["mean", "std", "skew", "n_outliers", "normality_test", "25%"]:
        assert col not in out.columns

    # Minimal string view shows length defaults but not the expanded-only extras.
    minimal = ps.xray(df, include="string", great_tables=False)
    assert {"min_length", "median_length", "max_length"} <= set(minimal.columns)
    assert "mode_share" not in minimal.columns
    assert "top_3" not in minimal.columns

    # A row's stats are correct.
    prov = out.filter(pl.col("column") == "prov")
    assert prov["min_length"][0] == 1
    assert prov["max_length"][0] == 3
    assert prov["mode_share"][0] == 60.0  # "a" appears 60 of 100


def test_iqr_column_omitted_when_percentiles_exclude_quartiles() -> None:
    """IQR needs both the 25th and 75th percentile; without them it must not appear.

    The non-numeric branch used to seed IQR with None unconditionally, so a
    string column resurrected an all-null IQR column that numeric-only frames
    correctly omitted.
    """
    import polarscope as ps

    df = pl.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0], "txt": ["a", "b", "a", "c", "b"]})

    out = ps.xray(df, include="all", percentiles=[0.1, 0.9], great_tables=False)
    assert "iqr" not in out.columns

    # ...but the default percentiles still produce a real IQR.
    default = ps.xray(df, include="all", great_tables=False)
    assert "iqr" in default.columns
    assert default.filter(pl.col("column") == "num")["iqr"][0] == 2.0


def test_correlation_columns_omitted_when_nothing_can_correlate() -> None:
    """corr_target must not emit all-null Correlation columns.

    String columns are never correlated with the target, so include='string'
    (or a frame whose only numeric column is the target itself) produced an
    entirely empty Correlation / Correlation_Plot pair.
    """
    import polarscope as ps

    df = pl.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "txt": ["a", "b", "a", "c", "b"],
            "target": [0, 1, 0, 1, 1],
        }
    )

    # No numeric column is analyzed -> nothing to correlate.
    strings = ps.xray(df, include="string", corr_target="target", great_tables=False)
    assert "correlation" not in strings.columns
    assert "correlation_plot" not in strings.columns

    # Target is the only numeric column -> still nothing to correlate, both modes.
    only_target = pl.DataFrame({"target": [0, 1, 0, 1], "txt": ["a", "b", "a", "c"]})
    for expanded in (False, True):
        out = ps.xray(
            only_target, include="all", expanded=expanded,
            corr_target="target", great_tables=False,
        )
        assert "correlation" not in out.columns, f"expanded={expanded}"
        assert "correlation_plot" not in out.columns, f"expanded={expanded}"

    # A genuine correlation is still reported.
    real = ps.xray(df, include="all", corr_target="target", great_tables=False)
    assert "correlation" in real.columns
    assert real.filter(pl.col("column") == "num")["correlation"][0] is not None


def test_corr_target_must_exist_and_be_numeric() -> None:
    import polarscope as ps

    df = pl.DataFrame({"num": [1.0, 2.0, 3.0], "txt": ["a", "b", "c"]})

    with pytest.raises(ValueError, match="not found in DataFrame"):
        ps.xray(df, include="all", corr_target="missing", great_tables=False)

    with pytest.raises(ValueError, match="must be numeric"):
        ps.xray(df, include="all", corr_target="txt", great_tables=False)


def test_corr_method_requires_corr_target() -> None:
    """corr_method has nothing to act on without a target, so it must be rejected."""
    import polarscope as ps

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 5.0, 9.0]})

    with pytest.raises(ValueError, match="without a valid corr_target"):
        ps.xray(df, corr_method="pearson", great_tables=False)

    # The pairing is checked before the value is, so an unsupported method
    # without a target still reports the missing target.
    with pytest.raises(ValueError, match="without a valid corr_target"):
        ps.xray(df, corr_method="kendall", great_tables=False)

    # Omitting both remains the happy path.
    assert "correlation" not in ps.xray(df, great_tables=False).columns


def test_corr_method_validated_against_a_real_target() -> None:
    import polarscope as ps

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 5.0, 9.0]})

    for method in ("pearson", "spearman"):
        out = ps.xray(
            df, corr_target="target", corr_method=method, great_tables=False
        )
        assert out.filter(pl.col("column") == "a")["correlation"][0] is not None

    with pytest.raises(ValueError, match="corr_method must be"):
        ps.xray(df, corr_target="target", corr_method="kendall", great_tables=False)


def test_corr_method_defaults_to_pearson_and_spearman_differs() -> None:
    """The method must reach pl.corr, not merely survive validation."""
    import polarscope as ps

    # Monotonic but strongly non-linear: Spearman is exactly 1, Pearson is not.
    df = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "target": [1.0, 2.0, 3.0, 4.0, 100.0]}
    )

    def corr_of(**kwargs: str) -> float:
        out = ps.xray(df, corr_target="target", great_tables=False, **kwargs)
        return out.filter(pl.col("column") == "a")["correlation"][0]

    implicit = corr_of()
    pearson = corr_of(corr_method="pearson")
    spearman = corr_of(corr_method="spearman")

    assert implicit == pytest.approx(pearson)
    assert spearman == pytest.approx(1.0)
    assert pearson < 0.9


def _corr_cells(frame: pl.DataFrame) -> dict[str, str]:
    """{column name: rendered Correlation_Plot cell HTML} for an xray table."""
    import re

    import polarscope as ps

    html = ps.xray(frame, include="all", corr_target="target").as_raw_html()
    cells = {}
    for row in re.findall(r"<tr.*?</tr>", html, re.S):
        name = re.search(r'<td[^>]*>([A-Za-z_]\w*)</td>', row)
        track = re.search(r'<div class="ps-corr-track".*?</div></div>', row, re.S)
        if name:
            cells[name.group(1)] = track.group(0) if track else row
    return cells


def _bar_px(cell: str) -> tuple[float, float, str]:
    """(track width, bar width, bar colour) parsed out of a rendered bar cell."""
    import re

    track_w = float(re.search(r'width:([\d.]+)px;height', cell).group(1))
    bar = re.search(r'class="ps-corr-bar"[^>]*?width:([\d.]+)px[^>]*?background:(#\w+)', cell)
    return track_w, float(bar.group(1)), bar.group(2)


def _xray_frame(df: pl.DataFrame) -> pl.DataFrame:
    import polarscope as ps

    return ps.xray(df, include="all", corr_target="target", great_tables=False)


def test_correlation_bar_has_fixed_width_track_and_linear_scale() -> None:
    """Bars sit on a constant-width track spanning exactly -1..1.

    fmt_nanoplot could not do this: it renders a scalar as a stub whose size
    depends on the other values in the column, which made every correlation
    look identical and unreadable.
    """
    n = 40
    df = pl.DataFrame(
        {
            "target": [float(i) for i in range(n)],
            "perfect": [float(i) for i in range(n)],
            "half": [i * 0.5 + (i % 7) * 4.0 for i in range(n)],
            "negative": [float(-i) for i in range(n)],
        }
    )
    xr = _xray_frame(df)
    cells = _corr_cells(df)

    tracks = {c: _bar_px(cells[c])[0] for c in ("perfect", "half", "negative")}
    assert len(set(tracks.values())) == 1, f"track width must be constant, got {tracks}"
    track_w = next(iter(tracks.values()))

    # A perfect correlation fills exactly half the track (centre -> edge).
    assert _bar_px(cells["perfect"])[1] == pytest.approx(track_w / 2)

    # Bar length is linear in |corr| against that same fixed half-track.
    for col in ("perfect", "half", "negative"):
        corr = xr.filter(pl.col("column") == col)["correlation"][0]
        assert _bar_px(cells[col])[1] == pytest.approx(abs(corr) * track_w / 2, abs=1.0)

    # Sign is encoded by colour.
    assert _bar_px(cells["perfect"])[2] == "#4A90E2"
    assert _bar_px(cells["negative"])[2] == "#E24A4A"


def test_correlation_plot_renders_no_none_text_and_shares_the_spanner() -> None:
    """The target's own row shows a dash, and one spanner covers both columns."""
    import re

    import polarscope as ps

    df = pl.DataFrame(
        {
            "target": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "feat": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        }
    )
    html = ps.xray(df, include="all", corr_target="target").as_raw_html()

    # The literal string "None" must never be rendered into a cell.
    assert not re.search(r"<td[^>]*>\s*None\s*</td>", html)

    # The correlation spanner covers Correlation *and* Correlation_Plot.
    spanner = re.search(
        r'<th[^>]*colspan="(\d+)"[^>]*>\s*(?:<span[^>]*>)?\s*Correlation with', html
    )
    assert spanner is not None, "correlation spanner not found"
    assert int(spanner.group(1)) == 2, f"spanner spans {spanner.group(1)} column(s), want 2"


def test_correlation_bar_scale_is_independent_of_other_columns() -> None:
    """The same correlation renders identically whatever its neighbours are.

    The nanoplot autoscaled to the observed range, so a weak correlation drawn
    beside a strong one came out exactly the same length.
    """
    weak = pl.DataFrame(
        {
            "target": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "feat": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2, 4.0],
        }
    )
    strong = weak.with_columns((pl.col("target") * 10 + 0.1).alias("strongfeat"))

    beside = _corr_cells(strong)
    alone_px = _bar_px(_corr_cells(weak)["feat"])[1]
    beside_px = _bar_px(beside["feat"])[1]
    strong_px = _bar_px(beside["strongfeat"])[1]

    assert alone_px == pytest.approx(beside_px)
    assert beside_px < strong_px / 2


def test_all_produced_column_names_are_lowercase() -> None:
    """Every statistic xray() produces is named in lowercase.

    The output used to mix Title_Case ('pct_missing'), ALLCAPS ('iqr') and
    lowercase ('skew', 'std'), which reads as inconsistent.
    """
    from datetime import date

    import polarscope as ps

    df = pl.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "txt": ["a", "b", "a", "c", "b", "a"],
            "flag": [True, False, True, True, False, True],
            "when": [date(2020, 1, i + 1) for i in range(6)],
            "target": [0, 1, 0, 1, 1, 0],
        }
    )
    for expanded in (False, True):
        for usability in (False, True):
            out = ps.xray(
                df, include="all", expanded=expanded, model_usability=usability,
                corr_target="target", great_tables=False,
            )
            bad = [c for c in out.columns if c != c.lower()]
            assert bad == [], f"expanded={expanded} usability={usability}: {bad}"


def test_correlation_plot_column_keeps_its_own_name() -> None:
    """The correlation bar column is labelled correlation_plot, not an axis legend."""
    import re

    import polarscope as ps

    df = pl.DataFrame({"target": [0.0, 1.0, 0.0, 1.0], "feat": [1.0, 2.0, 3.0, 1.5]})
    html = ps.xray(df, include="all", corr_target="target").as_raw_html()

    header = re.search(r'id="correlation_plot"[^>]*>(.*?)</th>', html, re.S)
    assert header is not None, "correlation_plot column header not found"
    assert re.sub(r"<[^>]+>", "", header.group(1)).strip() == "correlation_plot"


def test_expanded_mode_drops_unpopulated_string_and_temporal_stats() -> None:
    """Expanded mode must not render stat columns that nothing populated.

    String/temporal keys are seeded as None on every column so the summary
    schema stays stable. A numeric-only frame populates none of them, which
    left 11 columns of literal "None" in the expanded table.
    """
    import polarscope as ps

    numeric = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    out = ps.xray(numeric, expanded=True, great_tables=False)

    empty = [c for c in out.columns if out[c].null_count() == out.height]
    assert empty == [], f"all-null columns rendered: {empty}"
    for col in ("top", "top_freq", "mode_share", "sample_vals", "earliest", "latest"):
        assert col not in out.columns

    # A frame that actually has string data keeps its string stats.
    mixed = numeric.with_columns(pl.Series("txt", ["a", "b", "a", "c"]))
    mixed_out = ps.xray(mixed, include="all", expanded=True, great_tables=False)
    assert {"top", "top_freq", "mode_share"} <= set(mixed_out.columns)
    # ...but still no temporal columns, since there is no temporal column.
    assert "earliest" not in mixed_out.columns


def test_number_formatting_drops_trailing_zeros() -> None:
    """Whole numbers render bare, but genuine decimals keep their precision."""
    import re

    import polarscope as ps

    df = pl.DataFrame({"whole": [1.0, 2.0, 3.0, 4.0, 5.0]})
    for expanded in (False, True):
        html = ps.xray(df, expanded=expanded).as_raw_html()
        cells = {c.strip() for c in re.findall(r"<td[^>]*>([^<]*)</td>", html)}

        # mean/median/min/max of 1..5 are whole - no padding zeros
        assert "3.00" not in cells, f"expanded={expanded}"
        assert "1.00" not in cells, f"expanded={expanded}"
        assert {"1", "3", "5"} <= cells, f"expanded={expanded}: {sorted(cells)}"

        # std is 1.5811..., which must still honour decimals=2
        assert "1.58" in cells, f"expanded={expanded}"


def test_xray_internal_helpers() -> None:
    df = pl.DataFrame(
        {
            "num": [1, 2, 3],
            "txt": ["a", "b", "c"],
            "dt": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 3), eager=True),
            "ts": pl.Series(
                [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                ],
                dtype=pl.Datetime("us"),
            ),
        }
    )
    assert xray_mod._get_columns_to_analyze(df, None) == ["num"]
    assert set(xray_mod._get_columns_to_analyze(df, "all")) == {"num", "txt", "dt", "ts"}
    assert xray_mod._get_columns_to_analyze(df, "string") == ["txt"]
    assert xray_mod._get_columns_to_analyze(df, "temporal") == ["dt", "ts"]
    assert xray_mod._get_columns_to_analyze(df, ["Int64"]) == ["num"]
    assert xray_mod._get_columns_to_analyze(df, ["Datetime"]) == ["ts"]
    with pytest.raises(ValueError):
        xray_mod._get_columns_to_analyze(df, "invalid")

    assert xray_mod._percentile_to_label(0.5) == "50%"
    assert xray_mod._percentile_to_label(0.1) == "10%"
    quantiles = xray_mod._calculate_quantiles(pl.Series([1.0, 2.0, 3.0, 4.0]), [0.25, 0.5, 0.75])
    assert set(quantiles) == {"25%", "50%", "75%"}

    # String-like dtype detection covers Categorical and Enum
    assert xray_mod._is_stringy(pl.Series(["a"], dtype=pl.Categorical).dtype)
    assert xray_mod._is_stringy(pl.Series(["x"], dtype=pl.Enum(["x"])).dtype)
    assert not xray_mod._is_stringy(pl.Series([1]).dtype)

    assert xray_mod._suggest_optimal_dtype(pl.Series([1.0, 2.0, float("nan")]), pl.Float64) == "Int64"
    assert xray_mod._suggest_optimal_dtype(pl.Series([1.1, 2.2]), pl.Float64) == "Float32"
    # Int32 must check both ends of the range; a huge max used to be ignored.
    assert xray_mod._suggest_optimal_dtype(
        pl.Series([0, 3_000_000_000], dtype=pl.Int64), pl.Int64
    ) == "Int64"
    assert xray_mod._suggest_optimal_dtype(
        pl.Series([-2_000_000_000, 2_000_000_000], dtype=pl.Int64), pl.Int64
    ) == "Int32"
    # Float64 -> Float32 is suggested only when lossless (round-trip error
    # <= 1e-6), mirroring the shrink policy fix()/convert_datatypes applies.
    assert xray_mod._suggest_optimal_dtype(
        pl.Series([307491.47, 12345.678]), pl.Float64
    ) == "Float64"
    assert xray_mod._suggest_optimal_dtype(pl.Series([float("nan"), float("nan")]), pl.Float64) == "Float64"
    assert xray_mod._suggest_optimal_dtype(pl.Series(["x", "x", "x", "x", "y"]), pl.String) == "Categorical"

    outlier_series = pl.Series([1.0, 2.0, 3.0, 100.0])
    assert xray_mod._count_outliers(outlier_series, "iqr", None) >= 1
    assert xray_mod._count_outliers(outlier_series, "percentile", [0.25, 0.75]) >= 1
    assert xray_mod._count_outliers(outlier_series, "zscore", None) >= 0
    assert xray_mod._count_outliers(pl.Series([5.0]), "zscore", None) == 0

    nano_df = pl.DataFrame(
        {
            "distribution_plot": [[1, 2, 3], [], [5], None],
            "correlation_plot": [0.1, None, float("nan"), float("inf")],
        }
    )
    nano_df, has_hist = xray_mod._sanitize_nanoplot_column(
        nano_df, "distribution_plot", list_payload=True
    )
    nano_df, has_corr = xray_mod._sanitize_nanoplot_column(
        nano_df, "correlation_plot", list_payload=False
    )
    assert has_hist is True
    assert has_corr is True
    assert nano_df["distribution_plot"].to_list()[1] is None
    assert nano_df["correlation_plot"].to_list()[2] is None


def test_plots_core_helpers_and_drop_missing() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, None],
            "b": [1, None, 3],
            "c": ["x", "y", "z"],
        }
    )

    assert plots_mod._numeric_columns(df) == ["a", "b"]
    assert plots_mod._ensure_columns(df, None) == ["a", "b"]
    assert plots_mod._ensure_columns(df, ["a", "c", "missing"]) == ["a"]

    dropped_rows = plots_mod.drop_missing(df, axis="rows")
    assert dropped_rows.height <= df.height
    dropped_cols = plots_mod.drop_missing(df, axis="columns", thresh=0.67)
    assert dropped_cols.width <= df.width
    subset_rows = plots_mod.drop_missing(df, axis="rows", thresh=0.5, subset=["a", "b"])
    assert subset_rows.height <= df.height

    with pytest.raises(ValueError):
        plots_mod.drop_missing(df, axis="diagonal")

    with pytest.raises(ValueError):
        plots_mod.drop_missing(df, axis="rows", thresh=1.5)

    with pytest.raises(ValueError):
        plots_mod.drop_missing(df, axis="rows", subset=["missing_col"])

    with pytest.raises(ValueError, match="only supported when axis='rows'"):
        plots_mod.drop_missing(df, axis="columns", subset=["a"])

    # Threshold uses ceil semantics; with one subset column and thresh=0.5,
    # rows with nulls in that subset should be removed.
    subset_df = pl.DataFrame({"a": [1, None], "b": [10, 20]})
    kept = plots_mod.drop_missing(subset_df, axis="rows", thresh=0.5, subset=["a"])
    assert kept.height == 1

    # Column dropping should handle empty dataframes without division-by-zero.
    empty_df = pl.DataFrame(schema={"x": pl.Int64, "y": pl.String})
    kept_cols = plots_mod.drop_missing(empty_df, axis="columns", thresh=0.8)
    assert kept_cols.columns == ["x", "y"]

    assert plots_mod.drop_missing(pl.DataFrame(), axis="rows").shape == (0, 0)


def test_missingval_plot_sort_validation() -> None:
    df = pl.DataFrame({"a": [1, None], "b": [None, 2]})
    with pytest.raises(ValueError):
        plots_mod.missingval_plot(df, sort="invalid", backend="plotly")


def test_missingval_plot_counts_nan_as_missing() -> None:
    figure = plots_mod.missingval_plot(
        pl.DataFrame({"value": [1.0, float("nan"), None]}),
        backend="plotly",
    )
    assert list(figure.data[0].x) == [2]


def test_altair_backend_has_actionable_missing_extra_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_altair(name, *args, **kwargs):
        if name == "altair":
            raise ModuleNotFoundError("No module named 'altair'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_altair)

    with pytest.raises(ImportError, match=r"polarscope\[altair\]"):
        plots_mod.dist_plot(
            pl.DataFrame({"value": [1, 2, 3]}),
            backend="altair",
        )


def test_cat_plot_true_bottom_values_and_validation() -> None:
    values = (
        ["A"] * 50
        + ["B"] * 40
        + ["C"] * 30
        + ["D"] * 20
        + ["E"] * 10
        + ["F"] * 1
    )
    df = pl.DataFrame({"cat": values})

    fig = plots_mod.cat_plot(df, top=2, bottom=2, backend="plotly")
    trace_x = list(fig.data[0].x)
    assert trace_x == ["A", "B", "E", "F"]

    with pytest.raises(ValueError):
        plots_mod.cat_plot(df, top=-1, bottom=2, backend="plotly")


def test_cat_plot_supports_enum_columns() -> None:
    frame = pl.DataFrame(
        {
            "category": pl.Series(
                ["a", "b", "a"],
                dtype=pl.Enum(["a", "b"]),
            )
        }
    )
    figure = plots_mod.cat_plot(frame, backend="plotly")
    assert len(figure.data) == 1


def test_dist_plot_preserves_large_integer_values() -> None:
    values = [2**53 + 1, 2**53 + 3]
    figure = plots_mod.dist_plot(
        pl.DataFrame({"value": values}),
        backend="plotly",
    )
    assert list(figure.data[0].x) == values


def test_corr_plot_respects_explicit_altair_backend() -> None:
    pytest.importorskip("altair")
    frame = pl.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    chart = plots_mod.corr_plot(frame, backend="altair")
    assert type(chart).__module__.startswith("altair.")


def test_corr_plot_cluster_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "c": [1.0, 3.0, 2.0]})

    # Provide fake scipy modules that trigger import-time attribute errors in clustering block.
    fake_hierarchy = types.ModuleType("scipy.cluster.hierarchy")
    fake_distance = types.ModuleType("scipy.spatial.distance")
    monkeypatch.setitem(sys.modules, "scipy.cluster.hierarchy", fake_hierarchy)
    monkeypatch.setitem(sys.modules, "scipy.spatial.distance", fake_distance)

    fig = plots_mod.corr_plot(df, clustered=True, backend="plotly", interactive=True)
    assert fig is not None


def test_seaborn_backend_removed() -> None:
    """Verify seaborn backend was removed and raises clear error."""
    assert "seaborn" not in plots_mod._VALID_BACKENDS


def test_as_dataframe_accepts_lazyframe_and_rejects_other_types() -> None:
    import polarscope as ps

    frame = pl.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    lazy = frame.lazy()

    out = ps.xray(lazy, great_tables=False)
    assert out.height == 2
    assert ps.fix(lazy, verbose=False).height == frame.height
    assert ps.dist_plot(lazy) is not None

    with pytest.raises(TypeError, match="Polars DataFrame or LazyFrame"):
        ps.xray(object(), great_tables=False)

    with pytest.raises(TypeError, match="pl.from_pandas"):
        ps.fix([1, 2, 3], verbose=False)


def test_xray_threshold_validation() -> None:
    import polarscope as ps

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing_threshold"):
        ps.xray(df, missing_threshold=1.5, great_tables=False)
    with pytest.raises(ValueError, match="constant_threshold"):
        ps.xray(df, constant_threshold=-0.1, great_tables=False)
    with pytest.raises(ValueError, match="outlier_threshold"):
        ps.xray(df, outlier_threshold=float("nan"), great_tables=False)
    with pytest.raises(ValueError, match="skew_threshold"):
        ps.xray(df, skew_threshold=-1, great_tables=False)
    with pytest.raises(ValueError, match="shakiness_threshold"):
        ps.xray(df, shakiness_threshold=-1, great_tables=False)
    with pytest.raises(ValueError, match="shakiness_threshold"):
        ps.xray(df, shakiness_threshold=True, great_tables=False)


def test_xray_footnote_source_note_and_theme() -> None:
    from great_tables import GT

    import polarscope as ps

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

    styled = ps.xray(df, theme=2).as_raw_html()
    assert styled != ps.xray(df).as_raw_html()
    assert "Source: unit test" in ps.xray(df, source_note="Source: unit test").as_raw_html()

    if hasattr(GT, "tab_footnote"):
        html = ps.xray(
            df,
            footnote="Excludes nulls.",
            source_note="Source: unit test",
            theme="gray",
        ).as_raw_html()
        assert "Excludes nulls." in html
        assert "Source: unit test" in html
    else:
        with pytest.raises(ValueError, match="great_tables>=0.22"):
            ps.xray(df, footnote="Excludes nulls.")

    with pytest.raises(ValueError, match="great_tables=True"):
        ps.xray(df, theme="gray", great_tables=False)
    with pytest.raises(ValueError, match="theme"):
        ps.xray(df, theme="purple")
    with pytest.raises(ValueError, match="style"):
        ps.xray(df, theme=9)
    with pytest.raises(ValueError, match="theme dict keys"):
        ps.xray(df, theme={"colorscale": "blue"})


def test_correlation_plots_treat_undefined_as_missing() -> None:
    """Constant columns yield undefined correlations; those must not render as nan."""
    df = pl.DataFrame(
        {
            "const": [1.0, 1.0, 1.0, 1.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [4.0, 3.0, 2.0, 1.0],
        }
    )
    matrix = plots_mod._correlation_matrix(df, ["const", "b", "c"], "pearson")
    assert matrix[0][1] is None
    assert matrix[1][2] == pytest.approx(-1.0)

    heatmap = plots_mod.corr_heatmap(df, split="high", threshold=0.3, backend="plotly")
    assert heatmap is not None

    target = plots_mod.corr_heatmap(df, target="b", backend="plotly")
    texts = [cell for row in target.data[0].text for cell in row]
    assert "nan" not in texts
    assert any(cell == "" for cell in texts)

    plot = plots_mod.corr_plot(df, backend="plotly")
    plot_texts = [cell for row in plot.data[0].text for cell in row]
    assert "nan" not in plot_texts


def test_save_fig_rejects_unknown_objects() -> None:
    with pytest.raises(TypeError, match="Don't know how to save"):
        utils_mod.save_fig(object(), "/tmp/out.png")
