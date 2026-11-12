"""Tests for polarscope.plots and polarscope.utils.save_fig.

Every chart is serialized (Altair ``to_json()``, Plotly ``to_json()``) because
several backend failures - Altair's 5,000-row limit, invalid encoding
shorthand - only surface then, which is when a notebook displays the chart.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import polars as pl
import pytest

import polarscope as ps
from polarscope import plots

try:
    import altair as alt
except ImportError:  # optional dependency
    alt = None

needs_altair = pytest.mark.skipif(alt is None, reason="altair is not installed")
BACKENDS = ["plotly", pytest.param("altair", marks=needs_altair)]
# Names pl.col() does not take literally, and names Vega-Lite shorthand misreads.
PATTERN_NAMES = ["*", "^a.*$"]
SPECIAL_NAMES = ["x:y", "a.b", "c[0]", "back\\slash", *PATTERN_NAMES]


def _reject_constant(token: str):
    raise ValueError(f"{token} is not valid JSON")


def serialize(chart) -> dict:
    """Serialize a figure the way it is displayed and return the spec."""
    if hasattr(chart, "to_plotly_json"):
        return json.loads(chart.to_json())
    # Altair: also insist on strict JSON (no NaN/Infinity), which Vega needs.
    return json.loads(chart.to_json(), parse_constant=_reject_constant)


def altair_rows(spec: dict) -> list[dict]:
    """The single inline dataset of an Altair spec (layers share it)."""
    datasets = spec["datasets"]
    assert len(datasets) == 1
    return next(iter(datasets.values()))


def encoding_fields(node) -> list[str]:
    """Every ``field`` referenced anywhere in a Vega-Lite spec."""
    if isinstance(node, list):
        return [field for item in node for field in encoding_fields(item)]
    if not isinstance(node, dict):
        return []
    fields = []
    for key, value in node.items():
        if key == "datasets":
            continue
        if key == "field" and isinstance(value, str):
            fields.append(value)
        else:
            fields.extend(encoding_fields(value))
    return fields


def assert_fields_resolve(spec: dict) -> None:
    """Each encoded field must name one data column as Vega-Lite reads it.

    Vega-Lite treats unescaped ``.`` and ``[]`` as nested access and ``\\`` as
    an escape character.
    """
    keys = set().union(*(row.keys() for row in altair_rows(spec)))
    for field in encoding_fields(spec):
        assert not re.search(r"(?<!\\)[.\[\]]", field), field
        assert re.sub(r"\\(.)", r"\1", field) in keys, field


def histogram_bars(chart, backend: str) -> list[tuple[float, float, int]]:
    """(bin start, bin end, count) for each bar of a dist_plot figure."""
    if backend == "plotly":
        trace = chart.data[0]
        assert trace.type == "bar"
        half = trace.width / 2
        return [(x - half, x + half, y) for x, y in zip(trace.x, trace.y)]
    return [
        (row["bin_start"], row["bin_end"], row["count"])
        for row in altair_rows(serialize(chart))
    ]


def same_correlation(got, want) -> bool:
    if got is None or want is None:
        return got is want
    if math.isnan(want):
        return math.isnan(got)
    return math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12)


# ---------- dist_plot: pre-binned histograms (PS-08) ----------


@pytest.mark.parametrize("backend", BACKENDS)
def test_dist_plot_embeds_bin_counts_not_raw_values(backend):
    df = pl.DataFrame({"v": pl.int_range(200_000, eager=True).cast(pl.Float64).sin()})
    chart = ps.dist_plot(df, bins=30, backend=backend)

    spec = serialize(chart)
    assert len(json.dumps(spec)) < 50_000
    bars = histogram_bars(chart, backend)
    assert 1 <= len(bars) <= 31
    assert sum(count for _, _, count in bars) == df.height


@needs_altair
def test_dist_plot_altair_handles_more_than_5000_rows():
    df = ps.cardio()
    chart = ps.dist_plot(df, column="weight", backend="altair")
    rows = altair_rows(serialize(chart))
    assert len(rows) <= 31
    assert sum(row["count"] for row in rows) == df.height


@pytest.mark.parametrize("backend", BACKENDS)
def test_dist_plot_uses_nice_bins_bounded_by_bins(backend):
    df = pl.DataFrame({"v": [0.05 * i for i in range(21)]})
    bars = histogram_bars(ps.dist_plot(df, bins=10, backend=backend), backend)

    assert [round(start, 9) for start, _, _ in bars] == [round(0.1 * i, 9) for i in range(10)]
    assert all(math.isclose(end - start, 0.1) for start, end, _ in bars)
    # The maximum (1.0) sits on the last edge and is counted in the last bin.
    assert [count for _, _, count in bars] == [2] * 9 + [3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_dist_plot_integer_bins_are_centred_on_integers(backend):
    df = pl.DataFrame({"v": [0, 1, 1, 2, 10]})
    bars = histogram_bars(ps.dist_plot(df, backend=backend), backend)

    assert [(start + end) / 2 for start, end, _ in bars] == list(range(11))
    # 10 gets its own bin rather than being merged into the one for 9.
    assert [count for _, _, count in bars] == [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_dist_plot_ignores_null_nan_and_infinite_values(backend):
    df = pl.DataFrame({"v": [1.0, None, float("nan"), float("inf"), -float("inf"), 2.0]})
    chart = ps.dist_plot(df, backend=backend)
    serialize(chart)
    assert sum(count for _, _, count in histogram_bars(chart, backend)) == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_dist_plot_without_values_has_no_bars(backend):
    df = pl.DataFrame({"v": [None, None]}, schema={"v": pl.Float64})
    chart = ps.dist_plot(df, backend=backend)
    serialize(chart)
    assert histogram_bars(chart, backend) == []


@pytest.mark.parametrize(
    "dtype", [pl.Int8, pl.UInt64, pl.Float32, pl.Decimal(10, 2)], ids=str
)
def test_dist_plot_numeric_dtypes(dtype):
    df = pl.DataFrame({"v": [1, 2, 2, 3, 7]}).with_columns(pl.col("v").cast(dtype))
    chart = ps.dist_plot(df)
    serialize(chart)
    assert sum(count for _, _, count in histogram_bars(chart, "plotly")) == 5


def test_dist_plot_large_integers_are_all_counted():
    values = [2**53 + 1, 2**53 + 3]
    bars = histogram_bars(ps.dist_plot(pl.DataFrame({"value": values})), "plotly")
    assert sum(count for _, _, count in bars) == 2
    assert bars[0][0] <= min(values) and max(values) <= bars[-1][1]


def test_dist_plot_rejects_non_positive_bins():
    with pytest.raises(ValueError, match="bins"):
        ps.dist_plot(pl.DataFrame({"v": [1.0, 2.0]}), bins=0)


# ---------- Altair: large aggregated charts, global state (finding 2) ----------


def _wide_frame(n_cols: int, n_rows: int = 12) -> pl.DataFrame:
    return pl.DataFrame(
        {f"c{i}": [float((i * j) % 7 + j) for j in range(n_rows)] for i in range(n_cols)}
    )


@needs_altair
def test_altair_correlation_charts_with_many_columns():
    wide = _wide_frame(80)

    spec = serialize(ps.corr_plot(wide, backend="altair"))
    assert len(altair_rows(spec)) == 80 * 80

    spec = serialize(ps.corr_heatmap(wide, backend="altair"))  # layered with text
    assert len(altair_rows(spec)) == 80 * 80


@needs_altair
def test_altair_charts_do_not_change_global_row_limit():
    before = (alt.data_transformers.active, dict(alt.data_transformers.options))
    serialize(ps.corr_plot(_wide_frame(80), backend="altair"))
    assert (alt.data_transformers.active, dict(alt.data_transformers.options)) == before

    too_long = pl.DataFrame({"a": list(range(5001))})
    with pytest.raises(alt.MaxRowsError):
        alt.Chart(too_long).mark_point().encode(x="a:Q").to_dict()


@needs_altair
def test_altair_missingval_plot_with_many_columns():
    df = pl.DataFrame({f"c{i}": [1.0, None] for i in range(5001)})
    rows = altair_rows(serialize(ps.missingval_plot(df, backend="altair")))
    assert len(rows) == 5001


# ---------- correlations: Spearman ranking (PS-09) ----------


def _recording_corr(monkeypatch) -> list[str]:
    """Record the method of every pl.corr call made from now on."""
    methods: list[str] = []
    real_corr = pl.corr

    def corr(a, b, *, method="pearson", **kwargs):
        methods.append(method)
        return real_corr(a, b, method=method, **kwargs)

    monkeypatch.setattr(pl, "corr", corr)
    return methods


def test_spearman_ranks_null_free_columns_once(monkeypatch):
    df = _wide_frame(6, 30).with_columns(
        pl.Series("gappy", [None] + [float(j % 5) for j in range(29)])
    )
    methods = _recording_corr(monkeypatch)
    plots._correlation_matrix(df, df.columns, "spearman")

    # 7 columns -> 28 pairs. Only the 7 pairs involving the column with a null
    # need pl.corr(method="spearman"); the rest correlate precomputed ranks.
    assert methods.count("spearman") == 7
    assert methods.count("pearson") == 21


def test_spearman_target_mode_ranks_once(monkeypatch):
    df = _wide_frame(5, 30)
    methods = _recording_corr(monkeypatch)
    ps.corr_heatmap(df, target="c0", method="spearman")
    assert methods == ["pearson"] * 4


def test_spearman_matrix_matches_pairwise_pl_corr():
    nan = float("nan")
    df = pl.DataFrame(
        {
            "ties": [1, 2, 2, 3, 3, 3, 4, 5, 5, 6],
            "nan": [0.5, nan, 1.5, 2.0, -1.0, 3.0, nan, 0.0, 4.0, 2.5],
            "nulls": [3.0, None, 1.0, 4.0, None, 9.0, 2.0, 6.0, 5.0, 3.0],
            "const": [7.0] * 10,
            "rev": [10.0, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    for method in ("spearman", "pearson"):
        got = plots._correlation_matrix(df, df.columns, method)
        for i, a in enumerate(df.columns):
            for j, b in enumerate(df.columns):
                want = df.select(pl.corr(a, b, method=method)).item()
                assert same_correlation(got[i][j], want), (method, a, b)


# ---------- pattern-like column names (PS-26) ----------


def _pattern_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "*": [1.0, 2.0, 3.0, 4.0, 5.0],
            "^a.*$": [2.0, 1.0, None, 3.0, 5.0],
            "a1": [5.0, 3.0, 4.0, 1.0, 2.0],
        }
    )


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlations_with_pattern_like_names(method):
    df = _pattern_frame()
    plain = df.rename({"*": "star", "^a.*$": "regex"})
    want = plots._correlation_matrix(plain, plain.columns, method)
    assert plots._correlation_matrix(df, df.columns, method) == want

    fig = ps.corr_plot(df, method=method)
    serialize(fig)
    assert list(fig.data[0].x) == df.columns
    assert [list(row) for row in fig.data[0].z] == want

    fig = ps.corr_heatmap(df, target="*", method=method)
    serialize(fig)
    assert list(fig.data[0].y) == ["*"]
    assert list(fig.data[0].z[0]) == want[0][1:]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("column", PATTERN_NAMES)
def test_dist_plot_with_pattern_like_name(backend, column):
    df = _pattern_frame()
    chart = ps.dist_plot(df, column=column, backend=backend)
    serialize(chart)
    bars = histogram_bars(chart, backend)
    assert sum(count for _, _, count in bars) == df.height - df[column].null_count()


@pytest.mark.parametrize("backend", BACKENDS)
def test_missingval_plot_with_pattern_like_names(backend):
    df = pl.DataFrame(
        {"*": [None, 1.0, 2.0], "^a.*$": [None, None, 1.0], "a1": [1.0, 2.0, 3.0]}
    )
    chart = ps.missingval_plot(df, sort="none", backend=backend)
    spec = serialize(chart)
    if backend == "plotly":
        assert list(chart.data[0].y) == df.columns
        assert list(chart.data[0].x) == [1, 2, 0]
    else:
        assert [(r["column"], r["count"]) for r in altair_rows(spec)] == [
            ("*", 1), ("^a.*$", 2), ("a1", 0)
        ]


def test_cat_plot_with_pattern_like_names():
    df = pl.DataFrame({"*": ["x", "x", "y"], "^a.*$": ["p", "q", "q"], "a1": ["m"] * 3})
    fig = ps.cat_plot(df, top=1, bottom=1)
    serialize(fig)
    assert [list(trace.x) for trace in fig.data] == [["x", "y"], ["q", "p"], ["m"]]
    assert [list(trace.y) for trace in fig.data] == [[2, 1], [2, 1], [3]]


# ---------- Altair field names (PS-28) ----------


@needs_altair
@pytest.mark.parametrize("name", SPECIAL_NAMES)
def test_altair_dist_plot_with_special_column_names(name):
    df = pl.DataFrame({name: [1.0, 2.0, 2.0, 4.0]})
    spec = serialize(ps.dist_plot(df, column=name, backend="altair"))
    assert_fields_resolve(spec)
    assert spec["encoding"]["x"]["title"] == name
    assert sum(row["count"] for row in altair_rows(spec)) == 4


@needs_altair
def test_altair_charts_with_special_column_names():
    numeric = pl.DataFrame(
        {name: [float((i * 3 + k) % 5) for k in range(6)] for i, name in enumerate(SPECIAL_NAMES)}
    )
    strings = numeric.select(pl.all().cast(pl.String))
    for chart in (
        ps.corr_plot(numeric, backend="altair"),
        ps.corr_heatmap(numeric, backend="altair"),
        ps.missingval_plot(numeric, backend="altair"),
        ps.cat_plot(strings, backend="altair"),
    ):
        spec = serialize(chart)
        assert_fields_resolve(spec)
        shown = {value for row in altair_rows(spec) for value in row.values()}
        assert set(SPECIAL_NAMES) <= shown


# ---------- cat_plot (PS-13, PS-02, PS-37) ----------


@pytest.mark.parametrize("backend", BACKENDS)
def test_cat_plot_many_string_columns(backend):
    df = pl.DataFrame({f"s{i}": ["a", "b", "a"] for i in range(15)})
    chart = ps.cat_plot(df, backend=backend)
    spec = serialize(chart)
    if backend == "plotly":
        assert len(chart.data) == 15
    else:
        assert {row["column"] for row in altair_rows(spec)} == set(df.columns)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cat_plot_with_columns_named_like_its_fields(backend):
    df = pl.DataFrame(
        {"count": ["a", "b", "a"], "value": ["x", "x", "y"], "len": ["k", "k", "k"]}
    )
    chart = ps.cat_plot(df, backend=backend)
    spec = serialize(chart)
    if backend == "plotly":
        assert [list(trace.x) for trace in chart.data] == [["a", "b"], ["x", "y"], ["k"]]
        assert [list(trace.y) for trace in chart.data] == [[2, 1], [2, 1], [3]]
    else:
        rows = altair_rows(spec)
        assert [(r["column"], r["value"], r["count"]) for r in rows if r["type"] == "top"] == [
            ("count", "a", 2), ("count", "b", 1),
            ("value", "x", 2), ("value", "y", 1),
            ("len", "k", 3),
        ]


def test_cat_plot_orders_tied_counts_by_value():
    labels = [f"v{i:02d}" for i in range(40)]
    shuffled = labels[1::2] + labels[-2::-2]  # every label once, in a scrambled order
    df = pl.DataFrame({"cat": ["zz"] * 3 + shuffled})

    fig = ps.cat_plot(df, top=5, bottom=3)
    serialize(fig)
    assert list(fig.data[0].x) == ["zz", "v00", "v01", "v02", "v03", "v37", "v38", "v39"]


def test_cat_plot_only_converts_plotted_values(monkeypatch):
    lengths: list[int] = []
    real_to_list = pl.Series.to_list

    def to_list(self, *args, **kwargs):
        lengths.append(len(self))
        return real_to_list(self, *args, **kwargs)

    monkeypatch.setattr(pl.Series, "to_list", to_list)
    df = pl.DataFrame({"cat": [f"v{i}" for i in range(5000)]})
    fig = ps.cat_plot(df, top=3, bottom=2)
    assert lengths and max(lengths) <= 3
    assert len(fig.data[0].x) == 5


# ---------- missingval_plot (PS-18) ----------


def test_missingval_plot_reads_all_counts_in_one_select(monkeypatch):
    calls: list[int] = []
    real_select = pl.DataFrame.select

    def select(self, *args, **kwargs):
        calls.append(1)
        return real_select(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "select", select)
    df = pl.DataFrame({f"c{i}": [1.0, None, float("nan")] for i in range(300)})
    fig = ps.missingval_plot(df)
    assert len(calls) <= 2
    assert list(fig.data[0].x) == [2] * 300


# ---------- corr_heatmap threshold (PS-47) ----------


def _weakly_correlated() -> pl.DataFrame:
    df = pl.DataFrame(
        {"a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [2.0, 7, 1, 8, 2, 8, 1, 8, 2, 8]}
    )
    assert 0 < abs(df.select(pl.corr("a", "b")).item()) < 0.3
    return df


def test_corr_heatmap_honours_explicit_zero_threshold():
    df = _weakly_correlated()

    fig = ps.corr_heatmap(df, split="high", threshold=0.0)
    serialize(fig)
    assert fig.data[0].text[0][1] != ""

    # Not set: high/low still default to 0.3, which hides the weak pair.
    fig = ps.corr_heatmap(df, split="high")
    serialize(fig)
    assert fig.data[0].text[0][1] == ""


def test_corr_heatmap_threshold_validation():
    df = _weakly_correlated()
    with pytest.raises(ValueError, match="threshold"):
        ps.corr_heatmap(df, threshold=1.5)
    serialize(ps.corr_heatmap(df, threshold=None))
    fig = ps.corr_heatmap(df, split="low", threshold=0.0)
    assert "threshold=0.0" in fig.layout.title.text


# ---------- input validation (PS-45) ----------


@pytest.mark.parametrize(
    "func",
    [ps.corr_heatmap, ps.corr_plot, ps.dist_plot, ps.missingval_plot, ps.cat_plot],
)
def test_plot_functions_reject_lazyframes(func):
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0], "s": ["x", "y", "x"]})
    with pytest.raises(TypeError, match=r"\.collect\(\)"):
        func(df.lazy())
    with pytest.raises(TypeError, match="Polars DataFrame"):
        func(df.to_dict(as_series=False))


# ---------- save_fig (PS-29) ----------


@pytest.mark.parametrize("as_path", [str, Path], ids=["str", "Path"])
def test_save_fig_writes_plotly_html(tmp_path, as_path):
    fig = ps.dist_plot(pl.DataFrame({"v": [1.0, 2.0, 2.5, 4.0]}))
    target = tmp_path / "plot.html"
    ps.save_fig(fig, as_path(target))

    html = target.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert re.search(r'<script[^>]* src="https://cdn\.plot\.ly/', html)  # default: CDN


def test_save_fig_can_embed_plotlyjs(tmp_path):
    fig = ps.dist_plot(pl.DataFrame({"v": [1.0, 2.0, 2.5, 4.0]}))
    target = tmp_path / "PLOT.HTML"
    ps.save_fig(fig, target, include_plotlyjs=True)

    html = target.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert not re.search(r'<script[^>]* src="https://cdn\.plot\.ly/', html)
    assert len(html) > 1_000_000  # plotly.js itself is inlined


@needs_altair
@pytest.mark.parametrize("as_path", [str, Path], ids=["str", "Path"])
def test_save_fig_writes_altair_json(tmp_path, as_path):
    chart = ps.dist_plot(pl.DataFrame({"v": [1.0, 2.0, 2.5, 4.0]}), backend="altair")
    target = tmp_path / "chart.json"
    ps.save_fig(chart, as_path(target))
    assert json.loads(target.read_text(encoding="utf-8")) == serialize(chart)


def test_save_fig_static_png(tmp_path):
    """Static export through kaleido, when it is installed."""
    pytest.importorskip("kaleido")
    fig = ps.dist_plot(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    path = tmp_path / "plot.png"
    try:
        ps.save_fig(fig, path)
    except Exception as error:  # kaleido 1.x needs a Chrome/Chromium install
        if "chrome" in str(error).lower():
            pytest.skip(f"kaleido cannot find a browser: {error}")
        raise
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
