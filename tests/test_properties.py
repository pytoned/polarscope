"""Property-based tests: xray() and the cleaning functions on random frames.

Hypothesis generates frames with every dtype xray() handles and the values
that broke it before (NaN, +/-inf, nulls, extremes, Decimals, Booleans,
empty frames, pattern-like and ID-like names). The statistics are checked
against answers computed independently from the raw Python values, the
three engines (DataFrame, LazyFrame exact=True, LazyFrame streaming) are
checked against each other, and every result must render.

Runs are deterministic (derandomize) so CI is reproducible. For a deeper
local search: POLARSCOPE_HYPOTHESIS=explore pytest tests/test_properties.py
"""

from __future__ import annotations

import importlib
import math
import os
import re
import statistics
import warnings
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import polarscope as ps  # noqa: E402

xray_mod = importlib.import_module("polarscope.xray")

settings.register_profile(
    "ci",
    max_examples=40,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile(
    "explore",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile(os.environ.get("POLARSCOPE_HYPOTHESIS", "ci"))

NAN, INF = float("nan"), float("inf")

# ── Strategies ───────────────────────────────────────────────────────────────

FINITE = st.floats(
    min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False
)
SPECIAL = st.sampled_from([NAN, INF, -INF, 0.0, -0.0])
NAMES = [
    "x", "value", "count", "*", "^a.*$", "order_id", "Order ID", "kundenummer",
    "HIDE", "price kr", "naïve", "id_type", "a b", "col.1", "Survived",
]


def _nullable(values, null_rate: int = 6):
    """Values with a null now and then (1 in ``null_rate``)."""
    return st.one_of(*([values] * (null_rate - 1)), st.none())


def _column(kind: str, n: int, *, finite_only: bool):
    floats = FINITE if finite_only else st.one_of(FINITE, FINITE, FINITE, SPECIAL)
    if kind == "float":
        return st.lists(_nullable(floats), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.Float64)
        )
    if kind == "float32":
        return st.lists(_nullable(st.floats(-1e6, 1e6, width=32)), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.Float32)
        )
    if kind == "int":
        values = st.one_of(st.integers(-5, 5), st.integers(-(2**40), 2**40))
        return st.lists(_nullable(values), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.Int64)
        )
    if kind == "int8":
        return st.lists(_nullable(st.integers(-128, 127)), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.Int8)
        )
    if kind == "uint":
        return st.lists(_nullable(st.integers(0, 2**32 - 1)), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.UInt32)
        )
    if kind == "decimal":
        cents = st.integers(-(10**8), 10**8).map(lambda c: Decimal(c) / 100)
        return st.lists(_nullable(cents), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.Decimal(12, 2))
        )
    if kind == "bool":
        return st.lists(_nullable(st.booleans()), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.Boolean)
        )
    if kind == "string":
        text = st.one_of(
            st.sampled_from(["a", "b", "ç", "日本", "", "a b"]),
            st.text(max_size=6),
        )
        return st.lists(_nullable(text), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.String)
        )
    if kind == "category":
        return st.lists(_nullable(st.sampled_from(["lo", "mid", "hi"])), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.String).cast(pl.Categorical)
        )
    if kind == "date":
        days = st.integers(0, 50_000).map(lambda d: date(1970, 1, 1) + timedelta(days=d))
        return st.lists(_nullable(days), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.Date)
        )
    if kind == "datetime":
        secs = st.integers(0, 4 * 10**9).map(
            lambda s: datetime(1970, 1, 1) + timedelta(seconds=s)
        )
        return st.lists(_nullable(secs), min_size=n, max_size=n).map(
            lambda v: pl.Series(v, dtype=pl.Datetime("us"))
        )
    raise ValueError(kind)


ALL_KINDS = [
    "float", "float32", "int", "int8", "uint", "decimal", "bool", "string",
    "category", "date", "datetime",
]
NUMERIC_KINDS = ["float", "int", "int8", "uint", "decimal", "bool"]


@st.composite
def frames(draw, kinds=ALL_KINDS, *, max_rows=40, finite_only=False, min_rows=0):
    n = draw(st.integers(min_rows, max_rows))
    k = draw(st.integers(1, 4))
    names = draw(st.lists(st.sampled_from(NAMES), min_size=k, max_size=k, unique=True))
    columns = {}
    for name in names:
        kind = draw(st.sampled_from(kinds))
        columns[name] = draw(_column(kind, n, finite_only=finite_only)).alias(name)
    return pl.DataFrame(list(columns.values()))


def _quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


def _plain(frame, **kwargs):
    return _quiet(ps.xray, frame, great_tables=False, decimals=12, **kwargs)


def _numeric_targets(df: pl.DataFrame) -> list[str]:
    return [
        c
        for c, dt in df.schema.items()
        if (dt.is_numeric() or dt == pl.Boolean)
        and not xray_mod._is_id_column(c, dt)
    ]


def _close(a, b, rel=1e-9, abs_=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b or math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    return a == b


# ── Every frame, every engine: no crash, renders, invariants hold ───────────


@given(
    df=frames(),
    expanded=st.booleans(),
    use_target=st.booleans(),
    method=st.sampled_from([None, "zscore", "iqr"]),
)
def test_every_engine_runs_renders_and_keeps_invariants(df, expanded, use_target, method):
    targets = _numeric_targets(df)
    kwargs = {"include": "all", "expanded": expanded}
    if use_target and targets:
        kwargs["corr_target"] = targets[0]
    if method:
        kwargs["outlier_method"] = method
    variants = {
        "DataFrame": df,
        "LazyFrame exact": df.lazy(),
        "LazyFrame streaming": df.lazy(),
    }
    for engine, frame in variants.items():
        extra = {"exact": True} if engine == "LazyFrame exact" else {}
        if engine == "LazyFrame streaming" and method == "iqr":
            with pytest.raises(ValueError, match="exact=True"):
                ps.xray(frame, **kwargs)
            continue
        html = _quiet(ps.xray, frame, **kwargs, **extra).as_raw_html()
        cells = re.findall(r"<td[^>]*>(.*?)</td>", html, re.S)
        assert "None" not in cells, engine
        out = _plain(frame, **kwargs, **extra)
        assert out["column"].to_list() == df.columns, engine
        for row in out.iter_rows(named=True):
            _check_invariants(row, df.height, engine)


def _check_invariants(row: dict, n_rows: int, engine: str) -> None:
    where = f"{engine}: {row['column']}"
    missing = row["null_count"] + (row.get("nan_count") or 0)
    assert row["count"] + missing == n_rows, where
    assert 0 <= row["pct_missing"] <= 100, where
    if row.get("n_unique") is not None:
        assert 0 <= row["n_unique"] <= row["count"], where
    if row.get("uniqueness_ratio") is not None:
        assert 0 <= row["uniqueness_ratio"] <= 1, where
    if row.get("n_outliers") is not None:
        assert 0 <= row["n_outliers"] <= row["count"], where
    ordered = [row.get(k) for k in ("min", "25%", "50%", "75%", "max")]
    ordered = [v for v in ordered if v is not None and not math.isnan(v)]
    assert ordered == sorted(ordered), where


# ── Statistics against independent answers ──────────────────────────────────


def _python_values(series: pl.Series) -> list:
    """Present values as Python numbers: no null, no NaN; Decimal as float."""
    out = []
    for v in series.to_list():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out.append(float(v) if isinstance(v, Decimal) else int(v) if isinstance(v, bool) else v)
    return out


def _stdev(values: list) -> float:
    """Sample standard deviation, two-pass (Python 3.9's statistics.stdev
    trips an internal assertion on tiny values such as -1e-162)."""
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _linear_quantile(sorted_values: list, p: float) -> float:
    position = p * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


@given(df=frames(NUMERIC_KINDS, finite_only=True), method=st.sampled_from(["iqr", "zscore"]))
def test_numeric_statistics_match_independent_answers(df, method):
    out = _plain(df, include="all", expanded=True, outlier_method=method)
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    for name in df.columns:
        series, row = df.get_column(name), rows[name]
        if xray_mod._is_id_column(name, series.dtype):
            assert row.get("mean") is None and row.get("std") is None
            continue
        values = _python_values(series)
        assert row["count"] == len(values), name
        assert row["null_count"] == series.null_count(), name
        if not values:
            continue
        assert _close(row["min"], min(values)) and _close(row["max"], max(values)), name
        assert _close(row["mean"], statistics.fmean(values), rel=1e-7, abs_=1e-6), name
        if len(values) > 1:
            assert _close(row["std"], _stdev(values), rel=1e-6, abs_=1e-6), name
        ordered = sorted(values)
        for p, label in ((0.25, "25%"), (0.5, "50%"), (0.75, "75%")):
            assert _close(row[label], _linear_quantile(ordered, p), rel=1e-9, abs_=1e-6), (name, label)
        assert row["n_unique"] == len(set(values)), name
        if series.dtype == pl.Boolean:
            assert row["n_outliers"] is None, name
            continue
        if method == "iqr":
            q1, q3 = _linear_quantile(ordered, 0.25), _linear_quantile(ordered, 0.75)
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        else:
            if len(values) < 2 or _stdev(values) == 0:
                assert row["n_outliers"] == 0, name
                continue
            m, s = statistics.fmean(values), _stdev(values)
            lo, hi = m - 3 * s, m + 3 * s
        expected = sum(1 for v in values if v < lo or v > hi)
        # A value within float rounding of a fence may land on either side.
        near = sum(1 for v in values if math.isclose(v, lo, rel_tol=1e-9) or math.isclose(v, hi, rel_tol=1e-9))
        assert abs(row["n_outliers"] - expected) <= near, (name, row["n_outliers"], expected)


@given(df=frames(NUMERIC_KINDS, finite_only=True, min_rows=1))
def test_histograms_count_every_value(df):
    gt = _quiet(ps.xray, df, include="all")
    plots = dict(zip(gt._tbl_data["column"], gt._tbl_data["distribution_plot"]))
    for name in df.columns:
        series = df.get_column(name)
        if xray_mod._is_id_column(name, series.dtype):
            continue
        values = _python_values(series)
        if values:
            assert sum(plots[name]) == len(values), name


@given(df=frames(["string", "category", "date", "datetime"], min_rows=1))
def test_string_and_temporal_statistics_match_independent_answers(df):
    out = _plain(df, include="all", expanded=True)
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    for name in df.columns:
        series, row = df.get_column(name), rows[name]
        values = [v for v in series.to_list() if v is not None]
        assert row["count"] == len(values), name
        if not values:
            continue
        if series.dtype.is_temporal():
            assert (row["earliest"], row["latest"]) == (str(min(values)), str(max(values))), name
            continue
        lengths = [len(v) for v in values]
        assert (row["min_length"], row["max_length"]) == (min(lengths), max(lengths)), name
        if xray_mod._is_id_column(name, series.dtype):
            continue
        counts = Counter(values)
        top_count = max(counts.values())
        assert row["top_freq"] == top_count, name
        assert row["top"] == min(v for v, c in counts.items() if c == top_count), name
        assert row["n_unique"] == len(counts), name


# ── The engines agree ────────────────────────────────────────────────────────


@given(df=frames(), expanded=st.booleans())
def test_lazyframe_exact_equals_dataframe(df, expanded):
    # Same values; floats may differ in the last digit, because a column read
    # through the lazy engine can be chunked differently and floating-point
    # sums depend on the order.
    kwargs = {"include": "all", "expanded": expanded}
    exact, full = _plain(df.lazy(), exact=True, **kwargs), _plain(df, **kwargs)
    assert exact.columns == full.columns and exact.dtypes == full.dtypes
    for column in full.columns:
        for a, b in zip(full[column].to_list(), exact[column].to_list()):
            assert _close(a, b, rel=1e-12, abs_=1e-12), (column, a, b)


@given(df=frames(), expanded=st.booleans())
def test_streaming_equals_dataframe_where_both_report(df, expanded):
    kwargs = {"include": "all", "expanded": expanded}
    full = _plain(df, outlier_method="zscore", **kwargs)
    streamed = _plain(df.lazy(), **kwargs)
    assert set(streamed.columns) <= set(full.columns)
    for column in streamed.columns:
        for a, b in zip(full[column].to_list(), streamed[column].to_list()):
            assert _close(a, b, rel=1e-6, abs_=1e-6), (column, a, b)


# ── The memory guard ─────────────────────────────────────────────────────────


@given(df=frames(min_rows=1))
def test_guard_streams_instead_of_failing(df):
    full = xray_mod._available_memory
    xray_mod._available_memory = lambda: 0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = ps.xray(df, include="all", great_tables=False)
        assert any("summarized by streaming" in str(w.message) for w in caught)
        assert not {"25%", "50%", "75%", "iqr"} & set(out.columns)
        with pytest.raises(MemoryError):
            ps.xray(df, include="all", percentiles=[0.5])
    finally:
        xray_mod._available_memory = full


# ── Cleaning functions ───────────────────────────────────────────────────────


@given(df=frames(["float", "float32", "int", "int8", "uint", "string", "bool"]))
def test_convert_datatypes_is_lossless_and_idempotent(df):
    converted = _quiet(ps.convert_datatypes, df)
    assert converted.columns == df.columns and converted.height == df.height
    for name in df.columns:
        before, after = df.get_column(name), converted.get_column(name)
        back = after.cast(before.dtype)
        assert back.null_count() == before.null_count(), name
        assert back.equals(before, check_names=False) or (
            before.dtype.is_float()
            and back.is_nan().equals(before.is_nan())
            and back.filter(~before.is_nan()).equals(before.filter(~before.is_nan()))
        ), name
    again = _quiet(ps.convert_datatypes, converted)
    assert again.schema == converted.schema


@given(df=frames(), outliers=st.sampled_from([None, "iqr", "zscore"]))
def test_fix_never_adds_rows_or_drops_named_columns(df, outliers):
    fixed = _quiet(ps.fix, df, outliers=outliers, verbose=False)
    assert fixed.height <= df.height
    assert set(fixed.columns) <= set(ps.clean_column_names(df).columns) | set(df.columns)


# ── ID names ─────────────────────────────────────────────────────────────────

WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=8).filter(
    lambda w: w not in xray_mod._ID_WORDS
    and w not in xray_mod._ID_NUMBER_WORDS
    and not w.endswith("nummer")
)


@given(word=WORD, sep=st.sampled_from(["_", " ", "-"]), id_word=st.sampled_from(["id", "ID", "Id"]))
def test_names_ending_in_a_separate_id_word_are_ids(word, sep, id_word):
    assert xray_mod._is_id_column(f"{word}{sep}{id_word}", pl.Int64)
    assert xray_mod._is_id_column(f"{word}{sep}{id_word}", pl.String)
    assert not xray_mod._is_id_column(f"{word}{sep}{id_word}", pl.Float64)


@given(word=WORD)
def test_single_words_are_not_ids(word):
    # "hide", "valid", "paid", ...: "id" inside a word is not an ID.
    assert not xray_mod._is_id_column(word, pl.Int64)
    assert not xray_mod._is_id_column(word.upper(), pl.Int64)
