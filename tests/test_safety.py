"""Guardrails: xray() stays within the machine's memory, and skips ID columns."""

from __future__ import annotations

import importlib
import inspect
import re
import warnings

import polars as pl
import pytest

import polarscope as ps

xray_mod = importlib.import_module("polarscope.xray")


def _frame(n: int = 400) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [float(i % 37) for i in range(n)],
            "y": [i % 11 for i in range(n)],
            "s": [f"v{i % 4}" for i in range(n)],
        }
    )


def _note_texts(html: str) -> list[str]:
    return re.findall(r'class="gt_sourcenote"[^>]*>(?:<[^>]+>)*([^<]*)', html)


# ── Memory guard ─────────────────────────────────────────────────────────────


def test_available_memory_is_known_here():
    available = xray_mod._available_memory()
    assert isinstance(available, int) and available > 0
    headroom = xray_mod._cgroup_headroom()
    assert headroom is None or headroom >= 0


@pytest.fixture
def little_memory(monkeypatch):
    """A machine with 1 KB free: no full summary fits."""
    monkeypatch.setattr(xray_mod, "_available_memory", lambda: 1024)


@pytest.mark.parametrize("make", [lambda df: df, lambda df: df.lazy()])
def test_full_summary_that_does_not_fit_is_streamed(little_memory, make):
    df = _frame()
    with pytest.warns(UserWarning, match="summarized by streaming instead"):
        out = ps.xray(make(df), include="all", great_tables=False, exact=True)
    assert not {"25%", "50%", "75%", "iqr"} & set(out.columns)
    # The same values the streaming summary gives.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert out.equals(ps.xray(df.lazy(), include="all", great_tables=False))
        html = ps.xray(make(df), include="all", exact=True).as_raw_html()
    notes = _note_texts(html)
    assert any("GB of memory" in n and "free" in n for n in notes)
    assert any("Summarized by streaming" in n for n in notes)
    # exact=True was just refused: the notes must not suggest it.
    assert not any("Pass exact=True" in n for n in notes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outlier_method": "iqr"},
        {"outlier_method": "percentile", "outlier_bounds": [0.05, 0.95]},
        {"percentiles": [0.1, 0.9]},
        {"corr_target": "x", "corr_method": "spearman"},
    ],
)
@pytest.mark.parametrize("make", [lambda df: df, lambda df: df.lazy()])
def test_explicit_full_summary_options_raise_instead(little_memory, make, kwargs):
    with pytest.raises(MemoryError, match="GB of memory"):
        ps.xray(make(_frame()), exact=True, **kwargs)


def test_no_warning_when_the_full_summary_fits():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ps.xray(_frame(), great_tables=False)
    assert "25%" in out.columns


def test_the_guard_cannot_be_switched_off():
    # No parameter raises a memory limit or skips the check.
    names = set(inspect.signature(ps.xray).parameters)
    assert not {n for n in names if any(k in n for k in ("memory", "force", "unsafe", "limit"))}


def test_memory_estimate_scales_with_rows():
    lf = pl.LazyFrame({"v": [1.5, 2.5], "s": ["abcdefghijklmnopq", "x"]})
    schema = lf.collect_schema()
    kinds = {"v": "numeric", "s": "string"}
    need = xray_mod._exact_memory_need(lf, schema, ["v", "s"], kinds, 200_000_000)
    # 200M Float64 rows: the summary's working set is about 96 bytes a row.
    assert need >= 200_000_000 * 96
    small = xray_mod._exact_memory_need(lf, schema, ["v"], {"v": "numeric"}, 1_000)
    assert small < 10 * 1024**2


# ── ID columns ───────────────────────────────────────────────────────────────

ID_NAMES = [
    "ORDER_ID", "order id", "Order-ID", "customerId", "CustomerID", "PassengerId", "id",
    "ID", "uuid", "id_kunde", "kundenummer", "Kundennummer", "kunde_nr",
    "order_no", "customer_number", "subscription_id", "customer_key",
]
NOT_ID_NAMES = [
    "HIDE", "width", "valid", "paid", "hidden", "id_type", "ID_COUNT",
    "number_of_children", "ITEM_COUNT", "Survived", "zip_code", "idle_time", "video",
]


@pytest.mark.parametrize("name", ID_NAMES)
def test_id_names(name):
    assert xray_mod._is_id_column(name, pl.Int64)
    assert xray_mod._is_id_column(name, pl.String)


@pytest.mark.parametrize("name", NOT_ID_NAMES)
def test_not_id_names(name):
    assert not xray_mod._is_id_column(name, pl.Int64)


def test_id_dtypes():
    assert xray_mod._is_id_column("ORDER_ID", pl.Decimal(38, 0))
    assert xray_mod._is_id_column("ORDER_ID", pl.Categorical())
    assert not xray_mod._is_id_column("price_id", pl.Decimal(10, 2))
    assert not xray_mod._is_id_column("ratio_id", pl.Float64)
    assert not xray_mod._is_id_column("yes_no", pl.Boolean)


def _with_ids() -> pl.DataFrame:
    n = 300
    return pl.DataFrame(
        {
            "ORDER_ID": [i % 250 for i in range(n)],
            "customer_uuid": [f"c-{i:04d}" for i in range(n)],
            "HIDE": [float(i % 7) for i in range(n)],
            "target_id": [float(i % 5) for i in range(n)],
        }
    )


@pytest.mark.parametrize(
    "make",
    [lambda df: df, lambda df: df.lazy(), lambda df: df.lazy()],
    ids=["DataFrame", "LazyFrame-streaming", "LazyFrame-exact"],
)
def test_id_columns_get_counts_only(make, request):
    kwargs = {"exact": True} if request.node.callspec.id == "LazyFrame-exact" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ps.xray(
            make(_with_ids()), include="all", expanded=True, great_tables=False, **kwargs
        )
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    for name in ("ORDER_ID", "customer_uuid"):
        row = rows[name]
        assert row["count"] == 300
        assert row["mean"] is None and row["std"] is None and row["skew"] is None
        assert row["normality_test"] == "N/A (ID column)"
    assert rows["ORDER_ID"]["n_unique"] == 250
    assert rows["ORDER_ID"]["opt_dtype"] == "Int32"
    assert rows["customer_uuid"]["n_unique"] == 300
    assert (rows["customer_uuid"]["min_length"], rows["customer_uuid"]["max_length"]) == (6, 6)
    assert rows["HIDE"]["mean"] == pytest.approx(2.99)  # "HIDE" is not an ID


def test_id_columns_are_named_in_a_note_and_get_no_plot():
    gt = ps.xray(_with_ids(), include="all")
    assert any(
        "Identifier columns (by name), counted but not summarized: ORDER_ID, customer_uuid."
        in n
        for n in _note_texts(gt.as_raw_html())
    )
    plots = dict(zip(gt._tbl_data["column"], gt._tbl_data["distribution_plot"]))
    assert not plots["ORDER_ID"]


def test_an_id_named_correlation_target_is_analyzed():
    out = ps.xray(_with_ids(), corr_target="target_id", great_tables=False)
    rows = {r["column"]: r for r in out.iter_rows(named=True)}
    assert rows["target_id"]["mean"] == 2.0
    assert rows["HIDE"]["correlation"] is not None


# ── Reading the free memory (fake system files) ──────────────────────────────


def _fake_files(monkeypatch, files: dict[str, str]) -> None:
    import io

    def fake_open(path, *args, **kwargs):
        if path in files:
            return io.StringIO(files[path])
        raise FileNotFoundError(path)

    monkeypatch.setattr(xray_mod, "open", fake_open, raising=False)


def test_cgroup_v2_headroom_is_the_tightest_level(monkeypatch):
    _fake_files(
        monkeypatch,
        {
            "/proc/self/cgroup": "0::/app/worker\n",
            "/sys/fs/cgroup/app/worker/memory.max": "max\n",
            "/sys/fs/cgroup/app/worker/memory.current": "100\n",
            "/sys/fs/cgroup/app/memory.max": "1000\n",
            "/sys/fs/cgroup/app/memory.current": "400\n",
            "/sys/fs/cgroup/memory.max": "5000\n",
            "/sys/fs/cgroup/memory.current": "1000\n",
        },
    )
    assert xray_mod._cgroup_headroom() == 600  # app: 1000 - 400


def test_cgroup_v1_headroom_ignores_unlimited(monkeypatch):
    _fake_files(
        monkeypatch,
        {
            "/proc/self/cgroup": "4:memory:/pod\n3:cpu:/pod\n",
            "/sys/fs/cgroup/memory/pod/memory.limit_in_bytes": "2048\n",
            "/sys/fs/cgroup/memory/pod/memory.usage_in_bytes": "48\n",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes": "10\n",
        },
    )
    assert xray_mod._cgroup_headroom() == 2000


def test_no_cgroup_limit_reads_as_unknown(monkeypatch):
    _fake_files(monkeypatch, {"/proc/self/cgroup": "0::/\n", "/sys/fs/cgroup/memory.max": "max\n"})
    assert xray_mod._cgroup_headroom() is None
    _fake_files(monkeypatch, {})
    assert xray_mod._cgroup_headroom() is None


def test_meminfo_is_read_without_psutil(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil fails
    _fake_files(monkeypatch, {"/proc/meminfo": "MemTotal: 8192 kB\nMemAvailable: 2048 kB\n"})
    assert xray_mod._os_available_memory() == 2048 * 1024


def test_available_memory_takes_the_smaller_and_has_a_default(monkeypatch):
    monkeypatch.setattr(xray_mod, "_os_available_memory", lambda: 8 * 1024**3)
    monkeypatch.setattr(xray_mod, "_cgroup_headroom", lambda: 3 * 1024**3)
    assert xray_mod._available_memory() == 3 * 1024**3
    monkeypatch.setattr(xray_mod, "_os_available_memory", lambda: None)
    monkeypatch.setattr(xray_mod, "_cgroup_headroom", lambda: None)
    assert xray_mod._available_memory() == xray_mod._UNKNOWN_MEMORY
