"""xray(exact=True) on LazyFrames: same results as DataFrames, without collecting."""

from __future__ import annotations

import importlib
import math
import warnings
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

import polarscope as ps

xray_mod = importlib.import_module("polarscope.xray")


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


_CONFIGS = [
    {},
    {"include": "all"},
    {"include": "all", "expanded": True, "model_usability": True},
    {"include": "all", "corr_target": "i"},
    {"include": "all", "corr_target": "i", "corr_method": "spearman", "expanded": True},
    {"include": "all", "outlier_method": "zscore"},
    {
        "include": "all",
        "outlier_method": "percentile",
        "outlier_bounds": [0.1, 0.9],
        "percentiles": [0.1, 0.5, 0.9],
    },
    {"include": "all", "percentiles": [0.5]},
    {"include": "string", "expanded": True},
    {"include": "temporal", "expanded": True},
]


def _assert_frames_match(left: pl.DataFrame, right: pl.DataFrame) -> None:
    assert left.columns == right.columns
    for row_l, row_r in zip(left.iter_rows(), right.iter_rows()):
        for a, b in zip(row_l, row_r):
            if isinstance(a, float) and isinstance(b, float):
                assert (math.isnan(a) and math.isnan(b)) or a == b
            else:
                assert a == b


@pytest.mark.parametrize("config", _CONFIGS)
def test_lazyframe_matches_dataframe(config: dict) -> None:
    # With exact=True a LazyFrame gets the DataFrame's summary.
    df = _mixed_frame()
    eager = ps.xray(df, great_tables=False, **config)
    lazy = ps.xray(df.lazy(), great_tables=False, exact=True, **config)
    _assert_frames_match(eager, lazy)

    # The nanoplot payloads (dropped from the plain output) match too.
    eager_gt = ps.xray(df, **config)._tbl_data
    lazy_gt = ps.xray(df.lazy(), exact=True, **config)._tbl_data
    assert (
        eager_gt["distribution_plot"].to_list()
        == lazy_gt["distribution_plot"].to_list()
    )


def test_lazyframe_large_frame_matches_dataframe(tmp_path) -> None:
    # Above 300k rows DataFrames run their per-column queries one at a time,
    # and this LazyFrame is collected one column at a time. Results must not
    # depend on the input type or on how the file is chunked.
    n = 300_001
    start = date(2000, 1, 1)
    df = pl.DataFrame(
        {
            "x": [k % 977 for k in range(n)],
            "y": [k / 7 for k in range(n)],
            "s": [f"k{k % 50}" for k in range(n)],
            "d": pl.date_range(start, start + timedelta(days=n - 1), eager=True),
        }
    )
    eager = ps.xray(df, include="all", expanded=True, great_tables=False)
    lazy = ps.xray(
        df.lazy(), include="all", expanded=True, great_tables=False, exact=True
    )
    # Cardinality is exact at every size.
    assert "n_unique" in eager.columns
    assert eager.filter(pl.col("column") == "y")["n_unique"][0] == n
    _assert_frames_match(eager, lazy)

    path = tmp_path / "large.parquet"
    df.write_parquet(path, row_group_size=50_000)
    scanned = ps.xray(
        pl.scan_parquet(path),
        include="all",
        expanded=True,
        great_tables=False,
        exact=True,
    )
    _assert_frames_match(eager, scanned)


def test_scan_parquet_is_supported(tmp_path) -> None:
    df = _mixed_frame().drop("lst", "dur")
    path = tmp_path / "data.parquet"
    df.write_parquet(path)

    eager = ps.xray(df, include="all", expanded=True, great_tables=False)
    lazy = ps.xray(
        pl.scan_parquet(path),
        include="all",
        expanded=True,
        great_tables=False,
        exact=True,
    )
    _assert_frames_match(eager, lazy)


def test_lazyframe_reads_only_analyzed_columns() -> None:
    def boom(_value: int) -> str:
        raise RuntimeError("an unanalyzed column was evaluated")

    lf = pl.LazyFrame({"x": [1, 2, 3, 4]}).with_columns(
        never=pl.col("x").map_elements(boom, return_dtype=pl.String)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The default include analyzes numeric columns only, so the String
        # column must never be computed - collecting the frame would raise.
        result = ps.xray(lf, great_tables=False)
    assert result["column"].to_list() == ["x"]


@pytest.mark.parametrize("cells", [0, 600, 10_000_000])
def test_lazyframe_reads_each_column_once(monkeypatch, cells: int) -> None:
    # However many statistics come from a column, and whether the frame is
    # collected whole (default), two columns at a time (600 cells) or one
    # column at a time (0), each analyzed column is read once - the
    # corr_target column too.
    monkeypatch.setattr(xray_mod, "_COLLECT_CELLS", cells)
    reads: Counter[str] = Counter()

    def counted(name: str):
        def read(series: pl.Series) -> pl.Series:
            reads[name] += 1
            return series

        return read

    base = pl.LazyFrame(
        {
            "x": [float(i % 17) for i in range(300)],
            "y": [i % 5 for i in range(300)],
            "s": [f"v{i % 4}" for i in range(300)],
        }
    )
    lf = base.with_columns(
        pl.col(c).map_batches(counted(c), return_dtype=t)
        for c, t in base.collect_schema().items()
    )
    # xray() first counts the rows. Polars up to at least 1.36 evaluates the
    # first column of a plan like this one to do that (1.43 does not); that
    # is not a statistics read, so it is measured and allowed for.
    lf.select(pl.len()).collect(engine="streaming")
    row_count_reads = Counter(reads)
    reads.clear()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps.xray(
            lf,
            include="all",
            expanded=True,
            corr_target="y",
            great_tables=False,
            exact=True,
        )
    # The memory guard sizes the String column from its first rows (fixed-width
    # columns need no sample), which is one more (short) read of it.
    guard_reads = {"x": 0, "y": 0, "s": 1}
    assert reads == {c: 1 + row_count_reads[c] + guard_reads[c] for c in ("x", "y", "s")}


def test_lazyframe_subtitle_and_rendering() -> None:
    df = ps.titanic()
    lazy_gt = ps.xray(df.lazy(), include="all", corr_target="Survived")
    assert "(LazyFrame)" in lazy_gt._heading.subtitle
    assert "in memory" not in lazy_gt._heading.subtitle
    assert f"{df.height:,} rows × {df.width} columns" in lazy_gt._heading.subtitle
    lazy_gt.as_raw_html()

    eager_gt = ps.xray(df)
    assert "in memory" in eager_gt._heading.subtitle


def test_xray_rejects_non_frames() -> None:
    with pytest.raises(TypeError, match="DataFrame or LazyFrame"):
        ps.xray({"x": [1, 2, 3]})


def test_collect_falls_back_when_streaming_engine_fails() -> None:
    # A query the streaming engine cannot run is retried on the default engine.
    class StreamingUnsupported:
        def __init__(self) -> None:
            self.engines: list[str] = []

        def collect(self, engine: str = "auto") -> pl.DataFrame:
            self.engines.append(engine)
            if engine == "streaming":
                raise pl.exceptions.InvalidOperationError("not supported")
            return pl.DataFrame({"x": [1]})

    query = StreamingUnsupported()
    assert xray_mod._collect(query, streaming=True).item() == 1
    assert query.engines == ["streaming", "auto"]


def test_histogram_bins_are_right_closed() -> None:
    # 13 values 0..12 in 5 bins of width 2.4: edges 2.4, 4.8, 7.2, 9.6.
    df = pl.DataFrame({"v": [float(k) for k in range(13)]})
    plot = ps.xray(df.lazy())._tbl_data["distribution_plot"].to_list()[0]
    assert plot == [3.0, 2.0, 3.0, 2.0, 3.0]

    # A value exactly on an edge belongs to the lower bin: 0..10 in 5 bins of
    # width 2 puts 2, 4, 6 and 8 in the bins they close.
    df = pl.DataFrame({"v": [float(k) for k in range(11)]})
    plot = ps.xray(df)._tbl_data["distribution_plot"].to_list()[0]
    assert plot == [3.0, 2.0, 2.0, 2.0, 2.0]


def test_string_ties_are_deterministic() -> None:
    df = pl.DataFrame({"s": ["b", "a", "c", "a", "b", "c", "d"]})
    for frame in (df, df.lazy()):
        out = ps.xray(frame, include="string", expanded=True, great_tables=False)
        assert out["top"][0] == "a"
        assert out["top_3"][0] == "a (2), b (2), c (2)"
        assert out["sample_vals"][0] == "a, c, d"


def test_decimal_columns_get_full_numeric_treatment() -> None:
    df = pl.DataFrame(
        {
            "dec": pl.Series(
                [Decimal(f"{k}.5") for k in range(30)], dtype=pl.Decimal(10, 2)
            )
        }
    )
    # zscore outliers on a Decimal column used to fail with a Decimal overflow.
    out = ps.xray(df, outlier_method="zscore", great_tables=False)
    assert out["n_outliers"][0] == 0
    assert out["mean"][0] == pytest.approx(15.0)
    plot = ps.xray(df.lazy())._tbl_data["distribution_plot"].to_list()[0]
    assert plot is not None and sum(plot) == 30


def test_infinite_values_get_no_histogram() -> None:
    df = pl.DataFrame({"v": [0.0, 1.0, 2.0, float("inf")] * 5})
    plot = ps.xray(df.lazy())._tbl_data["distribution_plot"].to_list()[0]
    assert plot is None


@pytest.mark.parametrize(
    "values",
    [[3], [1, 2], [5, 1, 4], [1, 1, 2, 9], [2, 2, 2, 7, 7, 8], [10, 1, 3, 3, 5, 6, 6]],
)
def test_median_from_counts_matches_polars(values: list[int]) -> None:
    series = pl.Series(values)
    counts = series.value_counts()
    pairs = zip(counts[:, 0].to_list(), counts["count"].to_list())
    assert xray_mod._median_from_counts(pairs) == series.median()
