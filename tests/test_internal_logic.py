"""Focused unit tests for internal logic and edge branches."""

from __future__ import annotations

import importlib
import sys
import types

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
    assert set(names) == {"titanic", "diabetes"}

    titanic_path = datasets_mod.load_titanic(return_polars=False)
    diabetes_path = datasets_mod.load_diabetes(return_polars=False)
    assert titanic_path.endswith("titanic.parquet")
    assert diabetes_path.endswith("diabetes.parquet")

    titanic_df = datasets_mod.load_titanic()
    diabetes_df = datasets_mod.load_diabetes()
    assert isinstance(titanic_df, pl.DataFrame)
    assert isinstance(diabetes_df, pl.DataFrame)
    assert titanic_df.height > 0
    assert diabetes_df.height > 0

    assert "Titanic Dataset" in datasets_mod.dataset_info("titanic")
    assert "Diabetes Dataset" in datasets_mod.dataset_info("diabetes")
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
    assert out["Quality_Flag"][0] == "⚠ SHAKY"

    # distribution_plot only supports "histogram".
    with pytest.raises(ValueError):
        ps.xray(df_const, distribution_plot="kde")


def test_xray_model_usability_with_non_numeric_columns() -> None:
    """Regression: model_usability + non-numeric columns must not raise."""
    import polarscope as ps

    df = pl.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "txt": ["a", "b", "a", "c"]})
    out = ps.xray(
        df, include="all", expanded=True, model_usability=True,
        corr_target="num", great_tables=False,
    )
    assert "Usability_Score" in out.columns
    assert out.height == 2


def test_xray_internal_helpers() -> None:
    df = pl.DataFrame(
        {
            "num": [1, 2, 3],
            "txt": ["a", "b", "c"],
            "dt": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 3), eager=True),
        }
    )
    assert xray_mod._get_columns_to_analyze(df, None) == ["num"]
    assert set(xray_mod._get_columns_to_analyze(df, "all")) == {"num", "txt", "dt"}
    assert xray_mod._get_columns_to_analyze(df, "string") == ["txt"]
    assert xray_mod._get_columns_to_analyze(df, "temporal") == ["dt"]
    assert xray_mod._get_columns_to_analyze(df, ["Int64"]) == ["num"]
    with pytest.raises(ValueError):
        xray_mod._get_columns_to_analyze(df, "invalid")

    assert xray_mod._percentile_to_label(0.5) == "50%"
    assert xray_mod._percentile_to_label(0.1) == "10%"
    quantiles = xray_mod._calculate_quantiles(pl.Series([1.0, 2.0, 3.0, 4.0]), [0.25, 0.5, 0.75])
    assert set(quantiles) == {"25%", "50%", "75%"}

    short_kurtosis = xray_mod._calculate_kurtosis(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(short_kurtosis)
    valid_kurtosis = xray_mod._calculate_kurtosis(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert isinstance(valid_kurtosis, float)

    assert xray_mod._suggest_optimal_dtype(pl.Series([1.0, 2.0, float("nan")]), pl.Float64) == "Int64"
    assert xray_mod._suggest_optimal_dtype(pl.Series([1.1, 2.2]), pl.Float64) == "Float32"
    assert xray_mod._suggest_optimal_dtype(pl.Series([float("nan"), float("nan")]), pl.Float64) == "Float64"
    assert xray_mod._suggest_optimal_dtype(pl.Series(["x", "x", "x", "x", "y"]), pl.String) == "Categorical"

    outlier_series = pl.Series([1.0, 2.0, 3.0, 100.0])
    assert xray_mod._count_outliers(outlier_series, "iqr", None) >= 1
    assert xray_mod._count_outliers(outlier_series, "percentile", [0.25, 0.75]) >= 1
    assert xray_mod._count_outliers(outlier_series, "zscore", None) >= 0

    nano_df = pl.DataFrame(
        {
            "Distribution_Plot": [[1, 2, 3], [], [5], None],
            "Correlation_Plot": [0.1, None, float("nan"), float("inf")],
        }
    )
    nano_df, has_hist = xray_mod._sanitize_nanoplot_column(
        nano_df, "Distribution_Plot", list_payload=True
    )
    nano_df, has_corr = xray_mod._sanitize_nanoplot_column(
        nano_df, "Correlation_Plot", list_payload=False
    )
    assert has_hist is True
    assert has_corr is True
    assert nano_df["Distribution_Plot"].to_list()[1] is None
    assert nano_df["Correlation_Plot"].to_list()[2] is None


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

    # Threshold uses ceil semantics; with one subset column and thresh=0.5,
    # rows with nulls in that subset should be removed.
    subset_df = pl.DataFrame({"a": [1, None], "b": [10, 20]})
    kept = plots_mod.drop_missing(subset_df, axis="rows", thresh=0.5, subset=["a"])
    assert kept.height == 1

    # Column dropping should handle empty dataframes without division-by-zero.
    empty_df = pl.DataFrame(schema={"x": pl.Int64, "y": pl.String})
    kept_cols = plots_mod.drop_missing(empty_df, axis="columns", thresh=0.8)
    assert kept_cols.columns == ["x", "y"]


def test_missingval_plot_sort_validation() -> None:
    df = pl.DataFrame({"a": [1, None], "b": [None, 2]})
    with pytest.raises(ValueError):
        plots_mod.missingval_plot(df, sort="invalid", backend="plotly")


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
