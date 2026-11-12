from __future__ import annotations

import functools
import inspect
import math
import os
import re
import sys
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from typing import TYPE_CHECKING, Any, NamedTuple, Union

import polars as pl
import polars.selectors as cs

from ._helpers import literal_name_map
from .clean import _shrink_target

if TYPE_CHECKING:
    from great_tables import GT

# Optional scipy imports - lazy loaded to avoid import warnings
SCIPY_AVAILABLE = None  # Will be checked when needed
stats = None
_NORMALITY_MAX_SAMPLE_SIZE = 3000
# SciPy's Anderson-Darling p-value is interpolated in a table of 1%-15%.
_ANDERSON_P_RANGE = (0.01, 0.15)


def _check_scipy_availability():
    """Check if SciPy is available and import it if needed."""
    global SCIPY_AVAILABLE, stats

    if SCIPY_AVAILABLE is None:
        try:
            from scipy import stats as scipy_stats

            stats = scipy_stats
            SCIPY_AVAILABLE = True
        except (ImportError, ValueError):
            # Handle both import errors and binary incompatibility
            SCIPY_AVAILABLE = False
            stats = None

    return SCIPY_AVAILABLE


def _format_memory_usage(df: pl.DataFrame) -> str:
    """Format a DataFrame's estimated in-memory size with a sensible unit."""
    size_mb = df.estimated_size(unit="mb")
    if size_mb >= 1000:  # >= 1 GB
        return f"{size_mb / 1024:.1f} GB"
    elif size_mb >= 1.0:  # >= 1 MB
        return f"{size_mb:.1f} MB"
    return f"{size_mb * 1024:.1f} KB"


def _format_execution_time(execution_ms: float) -> str:
    """Format an execution time with appropriate units, like memory usage."""
    if execution_ms < 1000:
        return f"{execution_ms:.0f} ms"
    seconds = execution_ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def _is_stringy(dt) -> bool:
    """True for dtypes analyzed with string statistics (String/Categorical/Enum)."""
    return dt == pl.String or isinstance(dt, (pl.Categorical, pl.Enum))


def _get_columns_to_analyze(
    df: pl.DataFrame | pl.LazyFrame | Mapping[str, pl.DataType],
    include: str | list[str] | None,
) -> list[str]:
    """
    Filter columns based on include parameter.

    Parameters
    ----------
    df : pl.DataFrame, pl.LazyFrame, or schema mapping
        The frame (or its schema) to analyze
    include : str, list[str], or None
        Which data types to include

    Returns
    -------
    list[str]
        List of column names to analyze
    """
    schema = df.collect_schema() if isinstance(df, (pl.DataFrame, pl.LazyFrame)) else df
    items = list(schema.items())

    if include is None or include == "numeric":
        # Default behavior: only numeric columns
        return [c for c, dt in items if dt.is_numeric()]

    elif include == "all":
        # Include all columns
        return [c for c, _ in items]

    elif include == "string":
        # String-like columns (String, Categorical, Enum)
        return [c for c, dt in items if _is_stringy(dt)]

    elif include == "temporal":
        # Only date/datetime columns
        return [c for c, dt in items if dt.is_temporal()]

    elif isinstance(include, list):
        # Match exact dtypes and parameterized dtype families such as Datetime.
        include_types = {str(dtype) for dtype in include}
        return [
            c
            for c, dt in items
            if str(dt) in include_types or str(dt.base_type()) in include_types
        ]

    else:
        raise ValueError(
            f"Invalid include parameter: {include}. "
            f"Must be None, 'all', 'numeric', 'string', 'temporal', or list of dtype names."
        )


# ── Statistics engine ────────────────────────────────────────────────────────
#
# Every statistic is computed with Polars expressions, and each analyzed
# column gets its own small queries:
#
# - numeric: one query for everything that needs the column's values (moments,
#   quantiles from a single sort, cardinality, MAD, correlation and the inputs
#   of the SciPy tests), then one group-by that counts histogram bins and
#   outliers against the bounds the first query found;
# - string: group-bys for the value counts and the length distribution
#   (which yields min/median/mean/max length);
# - temporal and other dtypes: one query each.
#
# A DataFrame is first split into one-column frames, so planning a query does
# not get slower as the frame gets wider; small frames run their queries
# concurrently on a thread pool.
#
# A LazyFrame is summarized by streaming by default (_gather_streaming_facts):
# one streaming aggregation over all analyzed columns, then a small group-by
# per column against its results, so no column is ever held in memory and
# memory grows with neither the number of rows nor of columns. Statistics
# that need all of a column's values at once (quantiles, MAD) are left out.
#
# With exact=True a LazyFrame is collected in batches of analyzed columns of
# at most _COLLECT_CELLS cells (a single column when one column alone is
# larger), and each batch is summarized exactly like a DataFrame and released
# before the next one is read. Every column is read once, so this is as fast
# as collecting the frame first, at several times the memory of one batch
# instead of the whole frame. (Measured on a 20M-row, 12-column Parquet file:
# 6.5 s and 1.6 GB, against 6.7 s and 3.6 GB for collect-then-xray.)

# Above this many rows the numeric mode share (quasi-constant detection) is
# skipped, and queries run one at a time instead of concurrently.
_LARGE_DATASET_ROWS = 300_000

# A LazyFrame is collected in batches of analyzed columns of at most this many
# cells (rows x columns), or one column when a column alone holds more.
_COLLECT_CELLS = 10_000_000

# The SciPy tests use every value up to this many; above it, normality uses a
# _NORMALITY_MAX_SAMPLE_SIZE stride sample and the KS uniformity statistic is
# computed in Polars.
_TEST_EXACT_MAX = 5000

# ── Memory guard ──
# The full (exact) summary holds a column in memory, sorted and copied: on a
# 4-core machine, one 100M-row Float64 column (0.8 GB) peaked at 7.8 GB. Before
# computing it, xray() estimates that peak and compares it with the memory
# the machine (or container) has free. Above _MEMORY_SHARE of it, the summary
# is computed by streaming instead, which never holds a column. There is no
# option to override this.
_MEMORY_SHARE = 0.5
# Assumed when the free memory cannot be read (an unknown platform).
_UNKNOWN_MEMORY = 2 * 1024**3
# Peak bytes per row of the full summary, by kind of column, measured on
# 100M rows (Float64 72, Decimal 76, Int64 34, Datetime 11, String 11) and
# rounded up for the cases measured less favourably.
_EXACT_BYTES_PER_ROW = {"numeric": 96, "temporal": 24, "boolean": 24, "other": 128}
# A LazyFrame's string columns are sized from their first rows.
_STRING_SIZE_SAMPLE_ROWS = 10_000


def _os_available_memory() -> int | None:
    """Memory the operating system can still hand out, in bytes."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=2
            ).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            free = sum(
                int(re.search(rf"{key}:\s+(\d+)", out).group(1))
                for key in ("Pages free", "Pages inactive", "Pages speculative")
            )
            return free * page
        except Exception:
            pass
    return None


def _cgroup_headroom() -> int | None:
    """Memory left under this process's container (cgroup) limits, in bytes.

    A container's limit is invisible to /proc/meminfo, which reports the
    host. Every level of the cgroup path counts; the tightest one wins.
    """
    try:
        with open("/proc/self/cgroup") as cgroups:
            lines = cgroups.read().splitlines()
    except OSError:
        return None
    headroom = None

    def tighten(limit_file: str, usage_file: str) -> None:
        nonlocal headroom
        try:
            with open(limit_file) as fh:
                limit = fh.read().strip()
            with open(usage_file) as fh:
                usage = int(fh.read().strip())
        except (OSError, ValueError):
            return
        if limit == "max" or not limit.isdigit() or int(limit) >= 2**60:
            return
        left = max(0, int(limit) - usage)
        headroom = left if headroom is None else min(headroom, left)

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":  # cgroup v2
            base, limit, usage = "/sys/fs/cgroup", "memory.max", "memory.current"
        elif "memory" in controllers.split(","):  # cgroup v1
            base = "/sys/fs/cgroup/memory"
            limit, usage = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        parts = [p for p in path.split("/") if p]
        for depth in range(len(parts), -1, -1):
            directory = "/".join([base, *parts[:depth]])
            tighten(f"{directory}/{limit}", f"{directory}/{usage}")
    return headroom


def _available_memory() -> int:
    """Bytes this process can still use: the free memory, capped by any
    container limit; _UNKNOWN_MEMORY when neither can be read."""
    known = [m for m in (_os_available_memory(), _cgroup_headroom()) if m is not None]
    return min(known) if known else _UNKNOWN_MEMORY


def _exact_bytes_per_row(
    dtype: pl.DataType, kind: str, value_bytes: float | None
) -> float:
    """Peak bytes per row of the full summary of one column (see above).

    ``value_bytes`` is the in-memory size of one value, needed for strings
    and nested dtypes (a String value is 16 bytes plus its text beyond 12).
    """
    if kind == "id":
        # Counting distinct values holds a hash set of them.
        return 3 * max(value_bytes or 16, 16)
    if kind == "string":
        return 2.5 * max(value_bytes or 16, 16)
    if dtype == pl.Boolean:
        return _EXACT_BYTES_PER_ROW["boolean"]
    if kind in ("numeric", "temporal"):
        return _EXACT_BYTES_PER_ROW[kind]
    return max(_EXACT_BYTES_PER_ROW["other"], 4 * (value_bytes or 0))


# In-memory bytes per value of fixed-width dtypes (Categorical/Enum: codes).
_FIXED_WIDTH = {
    pl.Boolean: 1, pl.Int8: 1, pl.UInt8: 1, pl.Int16: 2, pl.UInt16: 2,
    pl.Int32: 4, pl.UInt32: 4, pl.Float32: 4, pl.Date: 4, pl.Categorical: 4,
    pl.Enum: 4, pl.Int64: 8, pl.UInt64: 8, pl.Float64: 8, pl.Datetime: 8,
    pl.Duration: 8, pl.Time: 8, pl.Decimal: 16,
}
if hasattr(pl, "Int128"):
    _FIXED_WIDTH[pl.Int128] = 16


def _exact_memory_need(
    df: pl.DataFrame | pl.LazyFrame,
    schema: Mapping[str, pl.DataType],
    columns: list[str],
    kinds: Mapping[str, str],
    n_rows: int,
) -> int:
    """Estimated peak bytes of the full summary of ``columns`` (see above).

    A DataFrame's columns are already in memory: the peak is the largest
    column's working set, or the concurrent ones' for frames small enough to
    run their queries in parallel. A LazyFrame also holds the batch of
    columns it reads (see _gather_facts).
    """
    value_bytes = {c: _FIXED_WIDTH.get(type(schema[c])) for c in columns}
    variable = [c for c in columns if value_bytes[c] is None]
    if isinstance(df, pl.DataFrame):
        value_bytes.update(
            {c: df.get_column(c).estimated_size() / max(df.height, 1) for c in variable}
        )
    elif variable:
        # Strings, lists, ... are sized from their first rows.
        sample = _collect(
            df.select(variable).head(_STRING_SIZE_SAMPLE_ROWS), streaming=True
        )
        rows_sampled = max(sample.height, 1)
        value_bytes.update(
            {c: sample.get_column(c).estimated_size() / rows_sampled for c in variable}
        )
    peaks = sorted(
        (n_rows * _exact_bytes_per_row(schema[c], kinds[c], value_bytes[c]) for c in columns),
        reverse=True,
    )
    if n_rows <= _LARGE_DATASET_ROWS:
        concurrent = min(len(peaks), (os.cpu_count() or 1) + 4)  # the thread pool
        need = sum(peaks[:concurrent])
    else:
        need = peaks[0]
    if isinstance(df, pl.LazyFrame):
        per_batch = max(1, _COLLECT_CELLS // max(n_rows, 1))
        batch_bytes = sorted((n_rows * value_bytes[c] for c in columns), reverse=True)
        need += sum(batch_bytes[:per_batch])
    return int(need)


# Errors a query can raise; a Rust panic is not an Exception subclass.
_QUERY_ERRORS = (Exception, pl.exceptions.PanicException)


def _collect(query: pl.LazyFrame, *, streaming: bool) -> pl.DataFrame:
    """Collect a query, on the streaming engine when requested."""
    if streaming:
        try:
            return query.collect(engine="streaming")
        except (pl.exceptions.PolarsError, pl.exceptions.PanicException):
            # The streaming engine does not support every operation in every
            # Polars release; the default engine gives the same answer.
            pass
    return query.collect()


# Identifier columns are recognized by name (see _is_id_column) and get only
# counts: a mean or standard deviation of IDs means nothing, and computing it
# costs time and memory.
_ID_WORDS = {"id", "ids", "uuid", "guid"}
_ID_NUMBER_WORDS = {"nr", "no", "num", "number", "nummer", "key"}
# Descriptive suffixes that make an "id_..." name an attribute, not an ID.
_ID_ATTRIBUTE_WORDS = {
    "type", "count", "cnt", "flag", "status", "source", "length", "len",
    "format", "kind", "category", "name", "valid", "is", "has",
}


def _name_tokens(name: str) -> list[str]:
    """Lowercase words of a column name: split on separators and camelCase."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.findall(r"[a-z0-9]+", spaced.lower())


def _is_id_column(name: str, dtype: pl.DataType) -> bool:
    """Whether a column's name and dtype say it holds identifiers.

    Matches whole words only, so "ORDER_ID", "Order ID", "order-id", "customerId",
    "id_kunde", "uuid", "kundenummer", "Kundennummer", "kunde_nr" and
    "order_no" are IDs while "HIDE", "valid", "paid", "id_type" and
    "number_of_children" are not:

    - "id", "ids", "uuid" or "guid" as the last word, or as the first word of
      a name not ending in a descriptive word ("id_type", "ID_COUNT");
    - a word ending in "nummer" (kundenummer, ordrenummer);
    - "nr", "no", "num", "number", "nummer" or "key" as the last of two or
      more words.

    Only dtypes IDs are stored in count: integers, Decimals without
    decimals, and strings (String, Categorical, Enum). A Boolean "yes_no" or
    a Float64 "ratio_id" is analyzed as usual.
    """
    if not (
        dtype.is_integer()
        or (isinstance(dtype, pl.Decimal) and dtype.scale == 0)
        or _is_stringy(dtype)
    ):
        return False
    words = _name_tokens(name)
    if not words:
        return False
    if words[-1] in _ID_WORDS:
        return True
    if words[0] in _ID_WORDS and words[-1] not in _ID_ATTRIBUTE_WORDS:
        return True
    if any(w.endswith("nummer") for w in words):
        return True
    return len(words) > 1 and words[-1] in _ID_NUMBER_WORDS


def _column_kind(dtype: pl.DataType) -> str:
    """Classify a dtype into the analysis branch xray() applies to it."""
    if dtype.is_numeric() or dtype == pl.Boolean:
        return "numeric"
    if _is_stringy(dtype):
        return "string"
    if dtype.is_temporal():
        return "temporal"
    return "other"


def _dtype_label(dtype: pl.DataType) -> str:
    """A short dtype name, the same on every supported Polars version.

    Parameters are positional and reprs that vary by release are normalized:
    Decimal(10, 2), Datetime(us), Datetime(us, UTC), Duration(ms),
    List(Int64), Array(Int64, 3), Struct(2 fields), Categorical,
    Enum(3 categories).
    """
    if isinstance(dtype, pl.Decimal):
        if dtype.precision is None:
            return f"Decimal(scale={dtype.scale})"
        return f"Decimal({dtype.precision}, {dtype.scale})"
    if isinstance(dtype, pl.Datetime):
        parts = [dtype.time_unit or "us"] + ([dtype.time_zone] if dtype.time_zone else [])
        return f"Datetime({', '.join(parts)})"
    if isinstance(dtype, pl.Duration):
        return f"Duration({dtype.time_unit or 'us'})"
    if isinstance(dtype, pl.Array):
        # A multidimensional Array nests Arrays; shape holds every dimension.
        inner = dtype.inner
        while isinstance(inner, pl.Array):
            inner = inner.inner
        shape = "x".join(str(size) for size in dtype.shape)
        return f"Array({_dtype_label(inner)}, {shape})"
    if isinstance(dtype, pl.List):
        return f"List({_dtype_label(dtype.inner)})"
    if isinstance(dtype, pl.Struct):
        n = len(dtype.fields)
        return f"Struct({n} field{'s' if n != 1 else ''})"
    if isinstance(dtype, pl.Enum):
        n = len(dtype.categories)
        return f"Enum({n} categor{'ies' if n != 1 else 'y'})"
    if dtype == pl.Categorical:
        return "Categorical"
    return str(dtype)


def _numeric_expr(
    column: str, dtype: pl.DataType, *, nan_as_null: bool = True
) -> pl.Expr:
    """A column as analyzed numerically.

    Boolean becomes 0/1 and Decimal becomes Float64. With ``nan_as_null``,
    NaN becomes null: Polars orders NaN above every number, so a single NaN
    would otherwise turn the mean into NaN, shift the quantiles and make every
    value an outlier. Pass False for columns known to hold no NaN: the
    conversion copies the column, which doubles the memory a query needs.
    """
    expr = pl.col(column)
    if dtype == pl.Boolean:
        return expr.cast(pl.UInt8)
    if isinstance(dtype, pl.Decimal):
        return expr.cast(pl.Float64)
    if nan_as_null and dtype.is_float():
        return expr.fill_nan(None)
    return expr


def _string_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    """A string-like column (String, Categorical, Enum) as String values."""
    expr = pl.col(column)
    return expr if dtype == pl.String else expr.cast(pl.String)


def _groupable_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    """A column Polars can group: Polars 1.25 cannot group Array, so as List."""
    expr = pl.col(column)
    if isinstance(dtype, pl.Array):
        return expr.cast(pl.List(dtype.inner))
    return expr


def _mode_count_expr(expr: pl.Expr) -> pl.Expr:
    """Frequency of the most common non-null value."""
    # value_counts on fixed field names: its default count field "count"
    # collides with a column named "count", and unique_counts() after
    # drop_nulls() raises a ShapeError on Polars 2 (and on the streaming
    # engine of 1.x) when the column holds nulls.
    counts = expr.drop_nulls().alias("__ps_value").value_counts(name="__ps_count")
    return counts.struct.field("__ps_count").max()


def _suggested_dtype(dtype: pl.DataType, facts: dict[str, Any], n_rows: int) -> str:
    """opt_dtype: the dtype fix()/convert_datatypes gives the column.

    The one addition is Int64 for float columns whose values are all whole,
    finite and within the Int64 range, a conversion fix() never makes.
    """
    count = facts.get("count") or 0
    vmin, vmax = facts.get("min"), facts.get("max")
    if dtype.is_float() and count:
        if (
            facts.get("nan_count") == 0
            and facts.get("n_finite") == count
            and facts.get("int_bad") == 0
            and -(2.0**63) <= vmin
            and vmax < 2.0**63
        ):
            return "Int64"
    target = _shrink_target(
        dtype,
        n_rows=n_rows,
        n_unique=facts.get("n_unique"),
        vmin=vmin if dtype.is_integer() else None,
        vmax=vmax if dtype.is_integer() else None,
        n_finite=facts.get("n_finite"),
        f32_exact=facts.get("f32_bad") == 0 if "f32_bad" in facts else None,
    )
    return _dtype_label(target if target is not None else dtype)


def _outlier_bounds(
    method: str,
    quantile: Any,
    bounds: list[float] | None,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[float, float] | None:
    """Outlier fences, or None when there can be no outliers.

    ``quantile`` maps a probability to the column's quantile; ``mean`` and
    ``std`` are those of the column's finite values.

    - ``"iqr"`` (the ``xray`` default): values outside
      [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    - ``"percentile"``: values outside the ``bounds`` quantiles
    - ``"zscore"``: values outside mean ± 3*std
    """
    if method == "iqr":
        q25, q75 = quantile(0.25), quantile(0.75)
        if q25 is None or q75 is None:
            return None
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    elif method == "percentile":
        lower, upper = quantile(bounds[0]), quantile(bounds[1])
        if lower is None or upper is None:
            return None
    else:  # zscore
        if mean is None or std is None or std == 0:
            return None
        lower, upper = mean - 3 * std, mean + 3 * std
    # inf - inf (quartiles of a mostly infinite column) leaves that side open.
    if math.isnan(lower):
        lower = -math.inf
    if math.isnan(upper):
        upper = math.inf
    return lower, upper


def _histogram_bins(n_valid: int) -> int:
    """Histogram bin count: grows with the data, capped at 12 for nanoplots."""
    return min(12, max(5, math.isqrt(n_valid)))


def _histogram_bin_expr(x: pl.Expr, lo: float, hi: float, n_bins: int) -> pl.Expr:
    """0-based equal-width bin index over [lo, hi].

    Bins are right-closed with the first one also holding ``lo``, matching
    ``Series.hist(bin_count=n_bins)``. Multiplying before dividing keeps values
    that sit exactly on a bin edge exact.
    """
    scaled = (x - lo) * n_bins / (hi - lo)
    return (scaled.ceil() - 1).clip(0, n_bins - 1)


def _median_from_counts(counts: Iterable[tuple[float, int]]) -> float | None:
    """Median of a distribution given as (value, frequency) pairs.

    Same convention as Polars' median: the mean of the two middle values when
    the total count is even.
    """
    pairs = sorted((v, c) for v, c in counts if v is not None)
    total = sum(c for _, c in pairs)
    if total == 0:
        return None
    lo_rank, hi_rank = (total - 1) // 2, total // 2
    lo_val = hi_val = None
    seen = 0
    for value, count in pairs:
        seen += count
        if lo_val is None and seen > lo_rank:
            lo_val = value
        if seen > hi_rank:
            hi_val = value
            break
    return (lo_val + hi_val) / 2


def _numeric_job_exprs(
    column: str,
    dtype: pl.DataType,
    *,
    quantile_points: list[float],
    need_extra: bool,
    mode_count: bool,
    zscore: bool,
    corr_expr: pl.Expr | None,
    want_tests: bool,
    want_ks: bool,
    want_all_values: bool,
    nan_as_null: bool,
) -> list[pl.Expr]:
    """Everything about a numeric column that needs its values in memory."""
    raw = pl.col(column)
    x = _numeric_expr(column, dtype, nan_as_null=nan_as_null)
    # Sort once: the quantiles, the median for MAD and the cardinality all
    # read the sorted copy instead of each selecting over the column again.
    ordered = x.drop_nulls().sort()
    nan_count = raw.is_nan().sum() if dtype.is_float() else pl.lit(0, dtype=pl.UInt32)
    exprs = [
        raw.null_count().alias("null_count"),
        nan_count.alias("nan_count"),
        x.count().alias("count"),
        x.mean().alias("mean"),
        x.std().alias("std"),
        x.min().alias("min"),
        x.max().alias("max"),
        x.skew().alias("skew"),
        x.kurtosis().alias("kurtosis"),
        (x == 0).sum().alias("n_zero"),
        (x > 0).sum().alias("n_pos"),
        (x < 0).sum().alias("n_neg"),
        ordered.n_unique().alias("n_unique"),
    ]
    exprs += [
        ordered.quantile(p, "linear").alias(f"q_{j}")
        for j, p in enumerate(quantile_points)
    ]
    if mode_count:
        exprs.append(_mode_count_expr(ordered).alias("mode_count"))
    if need_extra:
        exprs.append((ordered - ordered.median()).abs().median().alias("mad"))
    if zscore:
        finite = pl.when(x.is_finite()).then(x)
        exprs += [finite.mean().alias("finite_mean"), finite.std().alias("finite_std")]
    if corr_expr is not None:
        exprs.append(corr_expr.alias("correlation"))
    if want_tests:
        # Finite values in their original order: the normality sample
        # strides through them (same indices as _lilliefors_p_value).
        values = x.filter(x.is_finite())
        n = values.len()
        k = pl.when(n <= _TEST_EXACT_MAX).then(n).otherwise(_NORMALITY_MAX_SAMPLE_SIZE)
        index = (pl.int_range(0, k, dtype=pl.Int64) * (n / k)).floor().cast(pl.Int64)
        exprs += [
            n.alias("test_n"),
            values.gather(index).implode().alias("test_sample"),
        ]
    if want_tests and want_ks:
        exprs += _ks_uniform_exprs(x)
    if want_tests and want_all_values:
        exprs.append(values.implode().alias("test_values"))
    return exprs


def _ks_uniform_exprs(x: pl.Expr) -> list[pl.Expr]:
    """Two-sided KS statistic against the uniform distribution, in Polars.

    On the min-max normalized sorted finite values u_i, as scipy.stats.kstest:
    D = max(max_i(i/n - u_i), max_i(u_i - (i-1)/n)). Written with one shared
    array w_i = n*u_i - i, D = max(max(-w), max(w) + 1) / n, it needs half
    the column-sized buffers. Sorting the finite values directly is also
    several times faster than filtering an already sorted column.
    """
    values = x.filter(x.is_finite())
    n = values.len()
    lo, hi = values.min(), values.max()
    w = (values.sort() - lo) * (n / (hi - lo)) - pl.int_range(1, n + 1, dtype=pl.Int64)
    d_stat = pl.max_horizontal((-w).max(), w.max() + 1) / n
    return [d_stat.alias("test_ks_d"), lo.alias("test_lo"), hi.alias("test_hi")]


def _distribution_query(
    frame: pl.LazyFrame,
    column: str,
    dtype: pl.DataType,
    facts: dict[str, Any],
    *,
    outlier_method: str,
    outlier_bounds: list[float] | None,
    need_extra: bool,
) -> pl.LazyFrame | None:
    """Histogram bins, outlier count and opt_dtype float facts in one query.

    Sets ``histogram``/``n_bins`` and the fallback ``n_outliers`` in
    ``facts``; returns None when nothing needs the data.
    """
    x = _numeric_expr(column, dtype, nan_as_null=bool(facts.get("nan_count")))
    n_valid = facts["count"]

    # Like xray() always has: a constant column is one bar, a failed
    # conversion counts as constant, and a non-finite range gets no plot.
    try:
        lo, hi = float(facts["min"]), float(facts["max"])
    except (TypeError, ValueError):
        lo = hi = 0.0
    facts["min_float"], facts["max_float"] = lo, hi
    # Bin and outlier flag travel as one small integer key,
    # bin + n_bins * outlier: grouping on one UInt8 is about a quarter faster
    # than on the pair. _read_distribution decodes it.
    key = None
    if lo == hi:
        facts["histogram"] = [n_valid]
    elif math.isfinite(lo) and math.isfinite(hi):
        facts["n_bins"] = _histogram_bins(n_valid)
        key = _histogram_bin_expr(x, lo, hi, facts["n_bins"]).cast(pl.UInt8)
    else:
        facts["histogram"] = []

    # A Boolean has two values and no outliers: with 12% True, the IQR fences
    # (0, 0) would call every True an outlier.
    fences = None
    facts["n_outliers"] = None
    if dtype != pl.Boolean:
        fences = _outlier_bounds(
            outlier_method,
            lambda p, q=facts["quantiles"]: _to_float(q[p]),
            outlier_bounds,
            mean=_to_float(facts.get("finite_mean")),
            std=_to_float(facts.get("finite_std")),
        )
        facts["n_outliers"] = 0
    if fences is not None:
        lower, upper = fences
        outlier = ((x < lower) | (x > upper)).cast(pl.UInt8)
        key = outlier * facts.get("n_bins", 1) if key is None else key + outlier * facts["n_bins"]

    aggs = []
    if need_extra and dtype.is_float():
        v = pl.col("__ps_value")
        roundtrip = v.cast(pl.Float32).cast(pl.Float64)
        aggs = [
            (roundtrip != v).sum().alias("f32_bad"),
            (v != v.round(0)).sum().alias("int_bad"),
            v.is_finite().sum().alias("n_finite"),
        ]
    if key is None and not aggs:
        return None
    if key is None:
        return frame.select(x.alias("__ps_value")).select(aggs)
    if not aggs:
        return frame.select(key.alias("__ps_key")).group_by("__ps_key").agg(
            pl.len().alias("__ps_n")
        )
    values = frame.select(key.alias("__ps_key"), x.alias("__ps_value"))
    return values.group_by("__ps_key").agg(pl.len().alias("__ps_n"), *aggs)


def _read_distribution(facts: dict[str, Any], result: pl.DataFrame) -> None:
    """Histogram, outlier count and float facts from a _distribution_query."""
    rows = result.to_dicts()
    for fact in ("f32_bad", "int_bad", "n_finite"):
        if fact in result.columns:
            facts[fact] = sum(int(r[fact] or 0) for r in rows)
    if "__ps_key" not in result.columns:
        return
    n_bins = facts.get("n_bins")
    if n_bins:
        facts["histogram"] = [0] * n_bins
    for r in rows:
        key = r["__ps_key"]
        if key is None:  # a missing value: no bin, not an outlier
            continue
        if n_bins:
            facts["histogram"][key % n_bins] += r["__ps_n"]
        if facts["n_outliers"] is not None and key >= (n_bins or 1):
            facts["n_outliers"] += r["__ps_n"]


class _Job(NamedTuple):
    """A per-column query whose single result frame feeds one statistic group."""

    column: str
    name: str
    query: pl.LazyFrame
    # The query runs in bounded memory on the streaming engine and gives the
    # same exact answer there; otherwise it needs its column in memory anyway
    # and the default engine is faster.
    streamable: bool = False
    may_fail: bool = False


def _run_job(job: _Job, *, streaming: bool) -> pl.DataFrame | None:
    """Collect one job; a failing ``may_fail`` job yields None."""
    try:
        return _collect(job.query, streaming=streaming)
    except _QUERY_ERRORS:
        if job.may_fail:
            return None
        raise


def _run_jobs(jobs: list[_Job], *, parallel: bool) -> list[pl.DataFrame | None]:
    """Run per-column queries; failing ``may_fail`` jobs yield None.

    ``parallel`` runs them concurrently (small frames); otherwise one at a
    time, streamable ones on the streaming engine, so only about one column's
    working set is alive at once (for a large DataFrame, the streamable
    group-bys would otherwise add hash tables the size of a column on top of
    the data already in memory).
    """
    if parallel:
        # Threads rather than pl.collect_all(): Polars releases the GIL while
        # collecting, and collect_all's combined plan panics on these queries
        # in supported releases (1.25).
        with ThreadPoolExecutor() as pool:
            return list(pool.map(lambda job: _run_job(job, streaming=False), jobs))
    return [_run_job(job, streaming=job.streamable) for job in jobs]


def _has_nan(frame: pl.DataFrame, name: str) -> bool:
    """Whether a column holds NaN (only float columns can)."""
    return frame.schema[name].is_float() and bool(frame.get_column(name).is_nan().any())


def _spearman_target_rank(target: pl.DataFrame, column: str) -> pl.Series | None:
    """The corr_target column's ranks, computed once for every Spearman pair.

    Ranking the target once instead of in every column's query is exact only
    while the pair has no missing values (Polars drops incomplete pairs before
    ranking), so None when the target has any; each pair then ranks itself.
    """
    x = _numeric_expr(column, target.schema[column], nan_as_null=_has_nan(target, column))
    values = target.select(x.alias("t")).to_series()
    if values.null_count():
        return None
    return values.rank("average").cast(pl.Float64).alias("__ps_target_rank")


def _gather_facts(
    df: pl.DataFrame | pl.LazyFrame,
    schema: Mapping[str, pl.DataType],
    columns: list[str],
    *,
    n_rows: int,
    corr_target: str | None,
    corr_method: str,
    **options: Any,
) -> dict[str, dict[str, Any]]:
    """Per-column facts of a DataFrame or LazyFrame.

    A LazyFrame is collected in batches of analyzed columns of at most
    _COLLECT_CELLS cells (one column when a column alone is larger); each batch
    is summarized like a DataFrame and released before the next is read. The
    corr_target column is read once and stays loaded. ``options`` are passed
    on to _gather_column_facts.
    """
    if isinstance(df, pl.DataFrame):
        load = df.select
        batch_size = len(columns)
    else:
        def load(names: list[str]) -> pl.DataFrame:
            return _collect(df.select(names), streaming=True)

        batch_size = max(1, _COLLECT_CELLS // max(n_rows, 1))

    target = load([corr_target]) if corr_target else None
    target_rank = None
    if target is not None and corr_method == "spearman":
        target_rank = _spearman_target_rank(target, corr_target)

    facts: dict[str, dict[str, Any]] = {}
    for start in range(0, len(columns), batch_size):
        batch = columns[start : start + batch_size]
        names = [c for c in batch if c != corr_target]
        if target is None:
            frame = load(names)
        elif names:
            frame = load(names).hstack(target)
        else:
            frame = target
        facts.update(
            _gather_column_facts(
                frame,
                schema,
                batch,
                n_rows=n_rows,
                corr_target=corr_target,
                corr_method=corr_method,
                target_rank=target_rank,
                **options,
            )
        )
        del frame
    return facts


def _gather_column_facts(
    source: pl.DataFrame,
    schema: Mapping[str, pl.DataType],
    columns: list[str],
    *,
    n_rows: int,
    percentiles: list[float],
    outlier_method: str,
    outlier_bounds: list[float] | None,
    corr_target: str | None,
    corr_method: str,
    target_rank: pl.Series | None,
    need_extra: bool,
    want_tests: bool,
    want_ks: bool,
    want_all_values: bool,
    id_columns: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """Compute the raw per-column facts xray() turns into its summary.

    ``source`` holds the analyzed columns (and corr_target) in memory; it is
    split into one-column frames. ``target_rank`` is the target's Spearman
    ranks (see _spearman_target_rank). ``need_extra`` adds MAD and the
    opt_dtype facts (expanded mode or model usability); ``want_tests`` adds
    the inputs of the SciPy tests, ``want_ks`` the KS uniformity statistic and
    ``want_all_values`` every finite value for the tests that need them all.
    """
    kinds = {c: "id" if c in id_columns else _column_kind(schema[c]) for c in columns}
    is_large = n_rows > _LARGE_DATASET_ROWS
    parallel = not is_large

    target_expr = None
    if corr_target:
        target_expr = _numeric_expr(
            corr_target,
            schema[corr_target],
            nan_as_null=_has_nan(source, corr_target),
        )

    def frame(*names: str, extra: pl.Series | None = None) -> pl.LazyFrame:
        series = [source.get_column(n) for n in dict.fromkeys(names)]
        if extra is not None:
            series.append(extra)
        return pl.DataFrame(series).lazy()

    # Quantiles each numeric column needs: the requested percentiles plus the
    # outlier fences' own.
    quantile_points = list(percentiles)
    if outlier_method == "iqr":
        quantile_points += [0.25, 0.75]
    elif outlier_method == "percentile":
        quantile_points += list(outlier_bounds)
    quantile_points = sorted(set(quantile_points))

    def numeric_job(c: str, *, nan_as_null: bool) -> _Job:
        dtype = schema[c]
        corr_expr = None
        extra = None
        if corr_target and c != corr_target:
            x = _numeric_expr(c, dtype, nan_as_null=nan_as_null)
            clean = (
                target_rank is not None
                and source.get_column(c).null_count() == 0
                and not nan_as_null
            )
            if clean:
                extra = target_rank
                corr_expr = pl.corr(
                    pl.col("__ps_target_rank"),
                    x.rank("average").cast(pl.Float64),
                    method="pearson",
                )
            else:
                corr_expr = pl.corr(target_expr, x, method=corr_method)
        job_exprs = _numeric_job_exprs(
            c,
            dtype,
            quantile_points=quantile_points,
            need_extra=need_extra,
            mode_count=not is_large,
            zscore=outlier_method == "zscore" and dtype != pl.Boolean,
            corr_expr=corr_expr,
            want_tests=want_tests,
            want_ks=want_ks and parallel,
            want_all_values=want_all_values,
            nan_as_null=nan_as_null,
        )
        names = (c, corr_target) if corr_expr is not None and extra is None else (c,)
        return _Job(c, "numeric", frame(*names, extra=extra).select(job_exprs))

    def ks_job(c: str, *, nan_as_null: bool) -> _Job:
        # Run one at a time, the KS statistic's buffers would add to the
        # numeric query's; as a query of its own, peak memory stays near one
        # column's working set.
        x = _numeric_expr(c, schema[c], nan_as_null=nan_as_null)
        return _Job(c, "ks", frame(c).select(_ks_uniform_exprs(x)))

    # ── Round 1: per-column statistics ──
    jobs: list[_Job] = []
    for c in columns:
        dtype, kind = schema[c], kinds[c]

        if kind == "numeric":
            # fill_nan copies the column, so only columns holding NaN get it.
            nan_as_null = _has_nan(source, c)
            jobs.append(numeric_job(c, nan_as_null=nan_as_null))
            if want_tests and want_ks and not parallel:
                jobs.append(ks_job(c, nan_as_null=nan_as_null))

        elif kind == "string":
            # Categorical/Enum columns are grouped on their categories and
            # only the grouped values are cast to String.
            grouped_on = pl.col(c) if dtype != pl.String else _string_expr(c, dtype)
            counts = (
                frame(c)
                .select(grouped_on.alias("value"))
                .group_by("value")
                .agg(pl.len().alias("count"))
                .with_columns(pl.col("value").cast(pl.String))
            )
            # Value counts ranked by frequency (ties broken by value so the
            # result is deterministic), trimmed to the ranks the summary uses:
            # the top 12 for the bar plot plus the middle and rarest values for
            # the sample. The count of distinct values rides along.
            ranked = (
                counts.drop_nulls("value")
                .sort(["count", "value"], descending=[True, False])
                .with_row_index("rank")
                .with_columns(pl.len().alias("distinct"))
                .filter(
                    (pl.col("rank") < 12)
                    | (pl.col("rank") == pl.col("distinct") // 2)
                    | (pl.col("rank") == pl.col("distinct") - 1)
                )
            )
            jobs.append(_Job(c, "value_counts", ranked, streamable=True))
            # The (small) distribution of lengths gives the null/non-null
            # counts and every length statistic without holding the strings.
            if dtype == pl.String:
                lengths = (
                    frame(c)
                    .select(pl.col(c).str.len_chars().alias("length"))
                    .group_by("length")
                    .agg(pl.len().alias("count"))
                )
            else:
                lengths = (
                    counts.select(
                        pl.col("value").str.len_chars().alias("length"), "count"
                    )
                    .group_by("length")
                    .agg(pl.col("count").sum())
                )
            jobs.append(_Job(c, "length_counts", lengths, streamable=True))

        elif kind == "id":
            raw = pl.col(c)
            exprs = [
                raw.null_count().alias("null_count"),
                raw.count().alias("count"),
                _groupable_expr(c, dtype).drop_nulls().n_unique().alias("n_unique"),
            ]
            if need_extra and dtype.is_integer():
                # For opt_dtype only: the range decides the integer width.
                exprs += [raw.min().alias("min"), raw.max().alias("max")]
            jobs.append(_Job(c, "id", frame(c).select(exprs)))
            if dtype == pl.String:
                lengths = (
                    frame(c)
                    .select(pl.col(c).str.len_chars().alias("length"))
                    .group_by("length")
                    .agg(pl.len().alias("count"))
                )
                jobs.append(_Job(c, "length_counts", lengths, streamable=True))

        elif kind == "temporal":
            raw = pl.col(c)
            ordered = raw.drop_nulls().sort()
            job_exprs = [
                raw.null_count().alias("null_count"),
                raw.count().alias("count"),
                raw.min().alias("min"),
                raw.max().alias("max"),
                ordered.n_unique().alias("n_unique"),
            ]
            if not is_large:
                job_exprs.append(_mode_count_expr(ordered).alias("mode_count"))
            jobs.append(_Job(c, "temporal", frame(c).select(job_exprs)))

        else:
            # Nested/binary/other dtypes: not every operation supports every
            # dtype, so the cardinality queries run on their own and may fail.
            raw = pl.col(c)
            counts = frame(c).select(
                raw.null_count().alias("null_count"), raw.count().alias("count")
            )
            jobs.append(_Job(c, "counts", counts, streamable=True))
            groupable = _groupable_expr(c, dtype)
            n_unique = frame(c).select(
                groupable.drop_nulls().n_unique().alias("n_unique")
            )
            jobs.append(_Job(c, "n_unique", n_unique, may_fail=True))
            if not is_large:
                mode = frame(c).select(_mode_count_expr(groupable).alias("mode_count"))
                jobs.append(_Job(c, "mode_count", mode, may_fail=True))

    results = _run_jobs(jobs, parallel=parallel)

    facts: dict[str, dict[str, Any]] = {
        c: {"correlation": None, "mode_count": None, "nan_count": None} for c in columns
    }
    for job, result in zip(jobs, results):
        f = facts[job.column]

        if job.name == "numeric":
            res = result.row(0, named=True)
            f.update(
                {k: v for k, v in res.items() if not k.startswith(("q_", "test_"))}
            )
            f["quantiles"] = {p: res[f"q_{j}"] for j, p in enumerate(quantile_points)}
            if want_tests:
                f["tests"] = {
                    "n": res["test_n"],
                    "sample": result.get_column("test_sample")[0],
                    "ks_d": res.get("test_ks_d"),
                    "lo": res.get("test_lo"),
                    "hi": res.get("test_hi"),
                    "values": result.get_column("test_values")[0]
                    if want_all_values
                    else None,
                }

        elif job.name == "ks":
            res = result.row(0, named=True)
            f["tests"].update(
                ks_d=res["test_ks_d"], lo=res["test_lo"], hi=res["test_hi"]
            )

        elif job.name == "value_counts":
            ranks = result.sort("rank")
            f["n_unique"] = int(ranks["distinct"][0]) if ranks.height else 0
            f["ranked_values"] = dict(
                zip(
                    ranks["rank"].to_list(),
                    zip(ranks["value"].to_list(), ranks["count"].to_list()),
                )
            )

        elif job.name == "length_counts":
            pairs = list(zip(result["length"].to_list(), result["count"].to_list()))
            present = [(length, count) for length, count in pairs if length is not None]
            n_valid = sum(count for _, count in present)
            f["count"] = n_valid
            f["null_count"] = sum(count for length, count in pairs if length is None)
            if n_valid:
                f["min_length"] = min(length for length, _ in present)
                f["max_length"] = max(length for length, _ in present)
                f["avg_length"] = (
                    sum(length * count for length, count in present) / n_valid
                )
                f["median_length"] = _median_from_counts(present)

        elif job.name in ("temporal", "counts", "id"):
            f.update(result.row(0, named=True))

        elif job.name == "n_unique":
            if result is None:
                # Last resort for dtypes without n_unique support (e.g.
                # Object): count the distinct rows; unknown if that fails too.
                distinct = (
                    frame(job.column)
                    .select(pl.col(job.column))
                    .drop_nulls()
                    .unique()
                    .select(pl.len())
                )
                result = _run_job(
                    _Job(job.column, "distinct", distinct, may_fail=True),
                    streaming=False,
                )
            f["n_unique"] = int(result.item()) if result is not None else None

        elif job.name == "mode_count":
            f["mode_count"] = result.item() if result is not None else None

    for f in facts.values():
        f["count"] = int(f["count"])
        f["null_count"] = int(f["null_count"])
        if not f["count"]:
            f["n_unique"] = 0
        elif f.get("n_unique") is not None:
            f["n_unique"] = int(f["n_unique"])

    # ── Round 2: histogram, outliers and opt_dtype float facts ──
    # All are counts against bounds now known (the column's range and
    # quantiles), so one group-by per column yields them, streaming for a
    # LazyFrame.
    jobs = []
    for c in columns:
        f = facts[c]
        if kinds[c] != "numeric" or f["count"] == 0:
            continue
        query = _distribution_query(
            frame(c),
            c,
            schema[c],
            f,
            outlier_method=outlier_method,
            outlier_bounds=outlier_bounds,
            need_extra=need_extra,
        )
        if query is not None:
            jobs.append(_Job(c, "distribution", query, streamable=True))

    for job, result in zip(jobs, _run_jobs(jobs, parallel=parallel)):
        _read_distribution(facts[job.column], result)

    return facts



# String columns of a LazyFrame summarized by streaming get their value counts
# (top value, top 3, sample, bar plot, exact distinct count) when they hold at
# most this many distinct values; the count's hash table is bounded by it.
_TOP_VALUES_MAX_DISTINCT = 100_000

# A streaming summary counts distinct values exactly up to about this many;
# above it, from the values whose hash falls in a slice of the hash range
# holding about this many, scaled up (error about 1/sqrt of it: 0.1%).
_DISTINCT_SAMPLE = 1_000_000


# A streamed group-by keeps a partial table per chunk of rows until the end, so
# with many distinct keys its memory grows with the number of rows (50,000
# distinct values: 0.4 GB at 20M rows, 2.7 GB at 200M). Above this many
# distinct values (a rough count), a streaming summary therefore groups each
# batch of _BATCH_ROWS rows itself and merges the results as they come
# (LazyFrame.collect_batches, the reason Polars 1.34 is the minimum): memory
# is one batch plus the distinct values.
_GROUP_BY_MAX_DISTINCT = 10_000
_BATCH_ROWS = 1_000_000
# Batch results are merged once they outgrow 4x the merged result (at least
# this many rows), so the merging work stays proportional to what is new.
_MERGE_ROWS = 250_000


def _count_groups(
    lf: pl.LazyFrame, key: str, approx: int, *, counts: bool, as_string: bool = False
) -> pl.DataFrame:
    """The distinct values of ``key`` (with a ``count`` column when ``counts``).

    ``approx`` is a rough distinct count that picks the method (see
    _GROUP_BY_MAX_DISTINCT); ``as_string`` casts the key to String, after
    grouping on it as it is (category codes group faster than their strings).
    """

    def group(frame):
        grouped = frame.group_by(key)
        out = grouped.agg(pl.len().cast(pl.Int64).alias("count")) if counts else grouped.agg()
        return out.with_columns(pl.col(key).cast(pl.String)) if as_string else out

    batched = approx > _GROUP_BY_MAX_DISTINCT
    if not batched:
        return _collect(group(lf), streaming=True)

    def merge(frames: list[pl.DataFrame]) -> pl.DataFrame:
        combined = pl.concat(frames)
        if counts:
            return combined.group_by(key).agg(pl.col("count").sum())
        return combined.unique()

    merged: pl.DataFrame | None = None
    pending: list[pl.DataFrame] = []
    pending_rows = 0
    batches = lf.collect_batches(
        chunk_size=_BATCH_ROWS, maintain_order=False, engine="streaming"
    )
    for batch in batches:
        part = group(batch)
        pending.append(part)
        pending_rows += part.height
        if pending_rows > 4 * max(merged.height if merged is not None else 0, _MERGE_ROWS):
            merged = merge(([merged] if merged is not None else []) + pending)
            pending, pending_rows = [], 0
    frames = ([merged] if merged is not None else []) + pending
    if not frames:
        return group(lf.head(0).collect())
    return merge(frames)


def _streaming_stats_exprs(
    column: str, dtype: pl.DataType, kind: str, *, need_extra: bool
) -> list[pl.Expr]:
    """Pass 1 of a streaming summary: one column's streaming aggregations.

    The rough distinct count (approx_n_unique, which may count null as a
    value) only sizes later queries. It must not follow drop_nulls(): next to
    the other aggregations that makes Polars hold the whole column (2 GB for
    a 200M-row string column, against 0.7 GB without it).
    """
    raw = pl.col(column)
    exprs = [raw.null_count().alias("null_count")]
    if kind == "numeric":
        x = _numeric_expr(column, dtype)  # NaN as null; Boolean/Decimal as numbers
        exprs += [
            x.count().alias("count"),
            x.mean().alias("mean"),
            x.std().alias("std"),
            x.min().alias("min"),
            x.max().alias("max"),
            x.skew().alias("skew"),
            x.kurtosis().alias("kurtosis"),
        ]
        if dtype.is_float():
            exprs.append(raw.is_nan().sum().alias("nan_count"))
        if need_extra:
            # The zero/sign shares are only shown in expanded mode (and read
            # by model usability, which implies it).
            exprs += [
                (x == 0).sum().alias("n_zero"),
                (x > 0).sum().alias("n_pos"),
                (x < 0).sum().alias("n_neg"),
                x.approx_n_unique().alias("approx_n_unique"),
            ]
    elif kind == "string":
        # Counted on the physical values (category codes, day numbers...):
        # the same count, and Polars 1.25 cannot count Categorical or Date.
        exprs += [
            raw.count().alias("count"),
            raw.to_physical().approx_n_unique().alias("approx_n_unique"),
        ]
    elif kind == "temporal":
        exprs += [raw.count().alias("count"), raw.min().alias("min"), raw.max().alias("max")]
        if need_extra:
            physical = raw.to_physical()
            exprs.append(physical.approx_n_unique().alias("approx_n_unique"))
    elif kind == "id":
        exprs.append(raw.count().alias("count"))
        if need_extra:
            exprs.append(raw.to_physical().approx_n_unique().alias("approx_n_unique"))
            if dtype.is_integer():
                # For opt_dtype only: the range decides the integer width.
                exprs += [raw.min().alias("min"), raw.max().alias("max")]
    else:
        exprs.append(raw.count().alias("count"))
    return exprs


def _streaming_distinct_count(
    lf: pl.LazyFrame, values: pl.Expr, approx: int, n_rows: int
) -> int:
    """Distinct non-null values: exact up to about _DISTINCT_SAMPLE, else estimated.

    Values are hashed; above the threshold only those whose hash falls in the
    lowest fraction p of the hash range (about _DISTINCT_SAMPLE of them) are
    counted, and the count is divided by p. ``approx`` (a rough HyperLogLog
    count) picks p. The set being counted never exceeds about
    _DISTINCT_SAMPLE values, however large the column.
    """
    # Nulls are dropped as rows: an expression-level drop_nulls() changes the
    # column's length, which makes Polars hold the whole column in memory.
    hashes = (
        lf.select(values.alias("v"))
        .drop_nulls()
        .select(pl.col("v").hash(seed=0).alias("h"))
    )
    fraction = min(1.0, _DISTINCT_SAMPLE / max(approx, 1))
    if fraction < 1.0:
        cutoff = pl.lit(int(fraction * 2.0**64), dtype=pl.UInt64)
        hashes = hashes.filter(pl.col("h") < cutoff)
    # unique() would gather every row first (1.6 GB for a 200M-row column);
    # _count_groups streams.
    rough = min(approx, _DISTINCT_SAMPLE)
    count = _count_groups(hashes, "h", rough, counts=False).height
    return round(count / fraction)


def _string_facts_from_counts(f: dict[str, Any], counts: pl.DataFrame) -> None:
    """Value-count facts of a string column: ranks, distinct count, lengths."""
    values = counts.drop_nulls("value").sort(
        ["count", "value"], descending=[True, False]
    )
    distinct = values.height
    f["n_unique"] = distinct
    keep = set(range(min(12, distinct))) | {distinct // 2, distinct - 1}
    f["ranked_values"] = {
        rank: (value, count)
        for rank, (value, count) in enumerate(values.iter_rows())
        if rank in keep
    }
    lengths = values.select(
        pl.col("value").str.len_chars().alias("length"), "count"
    ).rows()
    _set_length_facts(f, lengths)


def _set_length_facts(f: dict[str, Any], lengths: list[tuple[int, int]]) -> None:
    """min/median/avg/max length from (length, count) pairs of present values."""
    n_valid = sum(count for _, count in lengths)
    if n_valid:
        f["min_length"] = min(length for length, _ in lengths)
        f["max_length"] = max(length for length, _ in lengths)
        f["avg_length"] = sum(length * count for length, count in lengths) / n_valid
        f["median_length"] = _median_from_counts(lengths)


def _streaming_correlation(
    lf: pl.LazyFrame,
    x: pl.Expr,
    x_mean: float,
    y: pl.Expr,
    y_mean: float,
) -> float:
    """Pearson correlation from streamed sums over the rows with both values.

    The values are shifted by their means first, which keeps the sums small
    and the subtraction in the formula exact enough.
    """
    both = x.is_not_null() & y.is_not_null()
    a = pl.when(both).then(x.cast(pl.Float64) - x_mean)
    b = pl.when(both).then(y.cast(pl.Float64) - y_mean)
    sums = _collect(
        lf.select(
            both.sum().alias("n"),
            a.sum().alias("a"),
            b.sum().alias("b"),
            (a * a).sum().alias("aa"),
            (b * b).sum().alias("bb"),
            (a * b).sum().alias("ab"),
        ),
        streaming=True,
    ).row(0, named=True)
    n = sums["n"]
    if n < 2:
        return math.nan  # what pl.corr gives without two complete pairs
    sa, sb = sums["a"] or 0.0, sums["b"] or 0.0
    cov = (sums["ab"] or 0.0) - sa * sb / n
    var_a = (sums["aa"] or 0.0) - sa * sa / n
    var_b = (sums["bb"] or 0.0) - sb * sb / n
    if var_a <= 0 or var_b <= 0:
        return math.nan  # a constant side, as pl.corr
    return max(-1.0, min(1.0, cov / math.sqrt(var_a * var_b)))


def _gather_streaming_facts(
    lf: pl.LazyFrame,
    schema: Mapping[str, pl.DataType],
    columns: list[str],
    *,
    corr_target: str | None,
    need_extra: bool,
    want_tests: bool,
    id_columns: frozenset[str] = frozenset(),
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Row count and per-column facts of a LazyFrame, without holding a column.

    Every query is a streaming one over a single column (two for a
    correlation), so memory is about one column's read buffers plus small
    results, whatever the number of rows or columns. (On a 20M-row,
    12-column Parquet file, one query over all columns took the same time at
    five times the memory.) Per column:

    - pass 1: counts, moments, extremes, z-score fences and a rough distinct
      count;
    - pass 2, against pass 1's results: histogram bins with z-score outliers
      and the opt_dtype float facts (one group-by); string value counts, or
      the length distribution for strings with more than
      _TOP_VALUES_MAX_DISTINCT distinct values; the distinct count (see
      _streaming_distinct_count); the correlation with the target; and a
      systematic row sample for the statistical tests.

    Facts no streaming query can give (quantiles, MAD, mode counts) are
    absent.
    """
    kinds = {c: "id" if c in id_columns else _column_kind(schema[c]) for c in columns}
    n_rows = None
    target_mean = None
    target_expr = None
    if corr_target:
        target_expr = _numeric_expr(corr_target, schema[corr_target])
        if corr_target not in columns:
            target_mean = _collect(
                lf.select(target_expr.mean()), streaming=True
            ).item()

    # The test sample: every row up to _TEST_EXACT_MAX (the values a
    # DataFrame's tests use), else a systematic one of about
    # _NORMALITY_MAX_SAMPLE_SIZE rows. Its step needs the row count, so the
    # samples are taken after pass 1.
    sampled: list[str] = []

    facts: dict[str, dict[str, Any]] = {}
    # The target first, so every other column's correlation has its mean.
    order = sorted(columns, key=lambda c: c != corr_target)
    for c in order:
        dtype, kind = schema[c], kinds[c]
        exprs = _streaming_stats_exprs(c, dtype, kind, need_extra=need_extra)
        if n_rows is None:
            exprs.append(pl.len().alias("__ps_rows"))
        totals = _collect(lf.select(exprs), streaming=True).row(0, named=True)
        if n_rows is None:
            n_rows = int(totals.pop("__ps_rows"))
        f: dict[str, Any] = {
            "correlation": None,
            "mode_count": None,
            "nan_count": None,
            "quantiles": {},
            **totals,
        }
        f["count"] = int(f["count"])
        f["null_count"] = int(f["null_count"])
        approx = f.pop("approx_n_unique", None)
        if kind == "numeric":
            f["nan_count"] = int(f.get("nan_count") or 0)
            # The z-score fences come from the finite values; they differ from
            # the plain moments only when the column holds +/-inf.
            f["finite_mean"], f["finite_std"] = f["mean"], f["std"]
            extremes = (_to_float(f["min"]), _to_float(f["max"]))
            if f["count"] and not all(map(math.isfinite, extremes)):
                x = _numeric_expr(c, dtype)
                finite = pl.when(x.is_finite()).then(x)
                f["finite_mean"], f["finite_std"] = _collect(
                    lf.select(finite.mean(), finite.std().alias("std")),
                    streaming=True,
                ).row(0)
        facts[c] = f
        if c == corr_target:
            target_mean = f["mean"]
        if not f["count"]:
            if approx is not None:
                f["n_unique"] = 0
            continue

        if kind == "numeric":
            query = _distribution_query(
                lf,
                c,
                dtype,
                f,
                outlier_method="zscore",
                outlier_bounds=None,
                need_extra=need_extra,
            )
            if query is not None:
                _read_distribution(f, _collect(query, streaming=True))
            x = _numeric_expr(c, dtype)
            if approx is not None:
                f["n_unique"] = _streaming_distinct_count(lf, x, approx, n_rows)
            if corr_target and c != corr_target and target_mean is not None:
                f["correlation"] = _streaming_correlation(
                    lf, x, f["mean"], target_expr, target_mean
                )
            if want_tests:
                sampled.append(c)

        elif kind == "string":
            if approx <= _TOP_VALUES_MAX_DISTINCT:
                counts = _count_groups(
                    lf.select(pl.col(c).alias("value")),
                    "value",
                    approx,
                    counts=True,
                    as_string=True,
                )
                _string_facts_from_counts(f, counts)
            else:
                text = _string_expr(c, dtype)
                lengths = (
                    lf.select(text.str.len_chars().alias("length"))
                    .drop_nulls()
                    .group_by("length")
                    .agg(pl.len().alias("count"))
                )
                _set_length_facts(f, _collect(lengths, streaming=True).rows())
                if need_extra:
                    f["n_unique"] = _streaming_distinct_count(lf, text, approx, n_rows)

        elif kind == "temporal" and approx is not None:
            f["n_unique"] = _streaming_distinct_count(lf, pl.col(c), approx, n_rows)

        elif kind == "id":
            if approx is not None:
                f["n_unique"] = _streaming_distinct_count(lf, pl.col(c), approx, n_rows)
            if dtype == pl.String:
                lengths = (
                    lf.select(pl.col(c).str.len_chars().alias("length"))
                    .drop_nulls()
                    .group_by("length")
                    .agg(pl.len().alias("count"))
                )
                _set_length_facts(f, _collect(lengths, streaming=True).rows())

        elif kind == "other" and need_extra and f["count"] <= _DISTINCT_SAMPLE:
            # No rough count to size a hash sample from: exact or nothing.
            try:
                f["n_unique"] = _streaming_distinct_count(
                    lf, _groupable_expr(c, dtype), f["count"], n_rows
                )
            except _QUERY_ERRORS:
                pass  # a dtype Polars cannot hash

        if f.get("n_unique") is not None:
            # An estimate can exceed the number of values it counts.
            f["n_unique"] = min(f["n_unique"], f["count"])

    step = 1
    if n_rows > _TEST_EXACT_MAX:
        step = -(-n_rows // _NORMALITY_MAX_SAMPLE_SIZE)
    for c in sampled:
        values = _collect(
            lf.select(_numeric_expr(c, schema[c]).cast(pl.Float64).alias(c))
            .gather_every(step),
            streaming=True,
        ).to_series()
        values = values.drop_nulls()
        values = values.filter(values.is_finite())
        facts[c]["tests"] = {
            "n": values.len(),
            "sample": values,
            "values": values,
            "ks_d": None,
            "lo": None,
            "hi": None,
        }
    return n_rows, facts

def xray(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    include: str | list[str] | None = None,
    great_tables: bool = True,
    expanded: bool = False,
    title: str | None = None,
    percentiles: list[float] | None = None,
    outlier_method: str | None = None,
    outlier_bounds: list[float] | None = None,
    corr_target: str | None = None,
    corr_method: str | None = None,
    normality_test: str = "shapiro",
    uniformity_test: str = "ks",
    missing_threshold: float = 0.3,
    constant_threshold: float = 0.99,
    skew_threshold: float = 2.0,
    kurtosis_threshold: float = 7.0,
    outlier_threshold: float = 0.05,
    shakiness_threshold: int = 2,
    model_usability: bool = False,
    decimals: int = 2,
    sep_mark: str = ",",
    dec_mark: str = ".",
    compact: bool = False,
    pattern: str | None = None,
    exact: bool = False,
) -> Union[GT, pl.DataFrame]:
    """
    X-ray your data: comprehensive statistical analysis with quality assessment.

    This function provides deep insight into DataFrame structure and quality,
    revealing hidden issues, statistical properties, and data health indicators.
    Perfect for exploratory data analysis and data quality assessment.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        The data to summarize. A LazyFrame is never collected: by default it
        is summarized by streaming, with no column held in memory; with
        exact=True it gets the full summary, reading a few columns (or one
        column) at a time. Only the analyzed columns are read (see Notes).
    include : str, list[str], or None, default None
        Which data types to include in the analysis.
        - None (default): Only numeric columns (signed and unsigned integers,
          floats and Decimal)
        - 'all': All columns regardless of data type
        - 'numeric': Only numeric columns (same as None)
        - 'string': String-like columns (String, Categorical, Enum)
        - 'temporal': Only Date, Datetime, Time and Duration columns
        - list of strings: Specific data type names (e.g., ['Float64', 'String'])
        Boolean columns are analyzed as 0/1 numerics (Mean = share of True),
        with no outlier count; temporal columns report their earliest/latest
        values. NaN in float
        columns counts as missing: count excludes it, pct_missing includes
        it, and a nan_count column appears when any analyzed column has NaN.
        Raises ValueError when no columns match the selection.
    great_tables : bool, default True
        Whether to return a formatted Great Tables object (True) or standard
        Polars DataFrame output (False). The plain DataFrame omits the
        distribution_plot and correlation_plot columns - they are raw nanoplot
        payloads that only render in Great Tables output.
    expanded : bool, default False
        If True, shows all available statistics. If False, shows only essential
        metrics: dtype, count, null_count, mean, std, min, 25%, 50%, 75%, max,
        iqr, pct_missing, n_outliers, skew.
        For string/categorical columns the essential view also includes the top
        value, its frequency, and the min/median/avg/max value length; the
        expanded view adds the mode share %, the top-3 values, and a sample of
        distinct values. Numeric-only statistics are omitted entirely when none
        of the analyzed columns are numeric (e.g. include='string').
        Expanded columns are grouped and ordered left to right as: Basic
        Statistics (dtype, opt_dtype, count, mean, std, min, max), Quantiles,
        Counts & Ratios, Outliers, Distribution, String Statistics, Temporal,
        Correlation, Statistical Tests, and Quality Assessment.
    title : str | None, optional
        Custom title for the Great Tables output. If None, uses default titles:
        "🔬 DataFrame X-ray" (minimal) or "🔬 Expanded Statistics" (expanded).
        Only applies when great_tables=True.
    percentiles : list[float] | None, optional
        Custom percentiles to calculate, with linear interpolation (so 50% is
        the median). Default: [0.25, 0.5, 0.75].
        Example: [0.1, 0.25, 0.5, 0.75, 0.9] for additional quantiles.
    outlier_method : str | None, default None
        Method used to compute the n_outliers and pct_outliers columns.
        None means "iqr", except for a LazyFrame summarized by streaming
        (exact=False), where it means "zscore" - the only method that needs
        no quartiles.
        - "iqr": Interquartile range method; values outside
          [Q1 - 1.5*IQR, Q3 + 1.5*IQR] count as outliers
        - "percentile": Use custom percentile bounds from outlier_bounds
        - "zscore": Z-score method (values outside mean ± 3*std, both taken
          over the column's finite values)
        +/-inf counts as an outlier; NaN never does (it is missing). Boolean
        columns get no outlier count: a minority value is not an outlier.
    outlier_bounds : list[float] | None, optional
        Custom percentile bounds for outlier detection when method="percentile".
        Example: [0.05, 0.95] for 5th and 95th percentiles.
    corr_target : str | None, optional
        Target column for correlation analysis. Must be a numeric or Boolean
        column (Booleans correlate as 0/1).
        Shows correlation between each numeric column and the target.
        The accompanying bars use a fixed -1 to 1 scale.
    corr_method : str | None, optional
        Correlation method passed to pl.corr(). Requires corr_target;
        passing it without a target raises ValueError. When omitted,
        "pearson" is used.
        - "pearson" (default): Linear correlation; sensitive to outliers and
          assumes a roughly linear relationship.
        - "spearman": Rank correlation; captures monotonic (not necessarily
          linear) relationships and is robust to outliers.
    normality_test : str, default "shapiro"
        Statistical test for normality (requires scipy). Shown in expanded
        mode as the normality_test column; the cell reports only the verdict
        and p-value at alpha=0.05, e.g. "NORMAL (p=0.481)" or
        "NON-NORMAL (p<0.001)" - the method is documented here, not displayed.
        - "shapiro" (default): Shapiro-Wilk test (good for small/medium
          samples). For more than 5,000 values it automatically falls back to
          the "ks" method below, where Shapiro-Wilk p-values are unreliable.
        - "anderson": Anderson-Darling test (sensitive to tail behavior).
          SciPy interpolates its p-value in a table that spans 1%-15%, so
          the cell shows "p≤0.010" or "p≥0.150" at the ends. Older scipy
          versions report no p-value; the verdict then uses the 5% critical
          value.
        - "ks": Lilliefors test - Kolmogorov-Smirnov against a normal
          distribution fitted to the data - on a deterministic sample of at
          most 3,000 values. The p-value is analytical (Dallal-Wilkinson, as
          in R's nortest), so it resolves small p-values instead of stopping
          at the floor of a simulation.
        Note: these tests reject for tiny deviations once a column has more
        than a few hundred rows, so large real-world datasets typically show
        "NON-NORMAL (p<0.001)" everywhere. Use skew/kurtosis to judge how far
        from normal a column actually is.
    uniformity_test : str, default "ks"
        Statistical test for uniformity (requires scipy). Shown in expanded
        mode as the uniformity_test column with the same compact format, e.g.
        "UNIFORM (p=0.876)" or "NON-UNIFORM (p<0.001)".
        - "ks" (default): Kolmogorov-Smirnov test vs the uniform distribution,
          after min-max normalizing the values to [0, 1]
        - "chi2": Chi-square goodness of fit over equal-width histogram bins
        The same large-sample caveat as for normality_test applies; discrete
        columns (e.g. years) are also tested against a *continuous* uniform
        distribution and therefore tend to reject even when their value counts
        are balanced.
    missing_threshold : float, default 0.3
        Threshold for flagging high missingness (0.3 = 30%).
    constant_threshold : float, default 0.99
        Threshold for flagging quasi-constant columns (0.99 = 99% same value).
    skew_threshold : float, default 2.0
        Threshold for flagging extreme skewness.
    kurtosis_threshold : float, default 7.0
        Threshold for flagging high kurtosis (fat tails).
    outlier_threshold : float, default 0.05
        Threshold for flagging outlier-heavy columns (0.05 = 5%).
    shakiness_threshold : int, default 2
        Minimum shakiness score to flag a column as "⚠ SHAKY" in the
        quality_flag column; lower scores show "✓ OK". The score counts one
        point per quality issue - see Notes.
    model_usability : bool, default False
        Include sophisticated model usability scoring with weighted flags and recommendations.
        Adds columns: usability_flags, usability_score, recommendation
        (in both minimal and expanded mode).
    decimals : int, default 2
        Number of decimal places for numeric formatting. In Great Tables
        output this is display-only formatting; with great_tables=False the
        float statistic columns are rounded to this many decimals so the
        option behaves the same in both output modes.
    sep_mark : str, default ","
        Thousands separator mark for numeric formatting (e.g., "1,000").
    dec_mark : str, default "."
        Decimal mark for numeric formatting (e.g., "1.23").
    compact : bool, default False
        If True, large numbers are auto-scaled with suffixes (e.g., "10K", "1.5M").
    pattern : str | None, optional
        Text pattern for decorating formatted values (e.g., "[{x}]").
    exact : bool, default False
        LazyFrames only (a DataFrame gets the full summary whenever it fits).
        By default a LazyFrame is summarized from streaming statistics, which
        never hold a column in memory, so memory stays flat however many rows
        there are (see Notes for the statistics it gives). Pass True for the
        full summary - quartiles, IQR, MAD, exact distinct counts - which
        reads the frame a few columns (or one column) at a time and needs
        several times the largest column's size in memory. Like every full
        summary it is subject to the memory guard (see Notes): when it would
        not fit, the LazyFrame is summarized by streaming anyway.

    Returns
    -------
    Union[GT, pl.DataFrame]
        Either a Great Tables object (if great_tables=True) or a Polars DataFrame
        (if great_tables=False) containing the comprehensive summary statistics.
        The Great Tables object is a plain GT, so any Great Tables method can be
        chained onto it to restyle the result - see Examples. xray() does not
        relabel columns, so the names in the great_tables=False output are the
        identifiers to use in chained calls.

    Examples
    --------
    Basic data X-ray (numeric columns only):

    >>> import polars as pl
    >>> import polarscope as ps
    >>> df = pl.DataFrame({
    ...     'price': [100, 200, 150, 300, 250],
    ...     'volume': [1000, 1500, 1200, 2000, 1800],
    ...     'category': ['A', 'B', 'A', 'C', 'B'],
    ...     'rating': [4.5, 3.8, 4.2, 4.9, 4.1]
    ... })
    >>> ps.xray(df)  # Only shows price, volume, rating (numeric columns)

    Include all columns (numeric and non-numeric):

    >>> full_xray = ps.xray(df, include='all')  # Shows all columns

    Include only string columns:

    >>> string_xray = ps.xray(df, include='string')  # Only shows category

    Comprehensive analysis with all columns:

    >>> advanced_xray = ps.xray(
    ...     df,
    ...     include='all',
    ...     expanded=True,
    ...     corr_target='price',
    ...     normality_test='anderson',
    ...     percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ... )

    Custom quality thresholds:

    >>> quality_xray = ps.xray(
    ...     df,
    ...     include='all',
    ...     missing_threshold=0.2,  # Flag >20% missing
    ...     skew_threshold=1.5,     # Flag |skew| > 1.5
    ...     shakiness_threshold=1   # Flag any quality issue
    ... )

    Advanced model usability assessment:

    >>> model_xray = ps.xray(
    ...     df,
    ...     include='all',
    ...     expanded=True,
    ...     model_usability=True,   # Include sophisticated quality flags
    ...     corr_target='price'     # For correlation reliability checks
    ... )

    Advanced formatting options:

    >>> formatted_xray = ps.xray(
    ...     df,
    ...     include='all',
    ...     decimals=3,            # 3 decimal places
    ...     compact=True,          # Use "10K" instead of "10,000"
    ...     sep_mark=" ",          # Space as thousands separator
    ...     dec_mark=",",          # Comma as decimal separator
    ...     pattern="({x})"        # Wrap values in parentheses
    ... )

    Anything Great Tables can do that xray() does not expose is available by
    chaining onto the returned object. Chained calls run after xray()'s own
    styling, so they win where the two overlap:

    >>> from great_tables import loc, md
    >>> styled = (
    ...     ps.xray(df)
    ...     .tab_options(table_background_color="#fdf6e3")
    ...     .tab_footnote(
    ...         footnote=md("Excludes nulls."),
    ...         locations=loc.body(columns="mean", rows=[0]),
    ...     )
    ... )

    X-ray a LazyFrame without collecting it, e.g. a Parquet file larger than
    memory (streaming statistics), or with the full summary one column at a
    time:

    >>> lazy_xray = ps.xray(pl.scan_parquet("big.parquet"), include="all")
    >>> full_xray = ps.xray(pl.scan_parquet("big.parquet"), exact=True)

    Notes
    -----
    A LazyFrame is summarized by streaming unless exact=True: one streaming
    pass over the analyzed columns, then a small group-by per column, so no
    column is ever held in memory and memory stays flat however many rows
    there are. That summary has count, null/NaN counts, % missing, mean, std,
    min, max, skew, kurtosis, zero/positive/negative shares, a histogram,
    z-score outliers, Pearson correlation, string lengths and earliest/latest;
    in expanded mode also the distinct count, opt_dtype, quality flags and the
    tests, run on a systematic sample of about 3,000 rows (every row up to
    5,000). Distinct counts are exact up to a million distinct values and
    estimated from a hash sample above that (error about 0.1%). String columns
    with at most 100,000 distinct values also get their top values. Every
    value shown equals the DataFrame's (up to the distinct-count estimate).
    Left out are the percentiles, IQR, MAD and duplicate counts, which need
    all of a column's values at once; for the same reason
    outlier_method="iqr"/"percentile", custom percentiles and
    corr_method="spearman" raise an error unless exact=True.

    With exact=True a LazyFrame gives results identical to a DataFrame. It is
    read in batches of analyzed columns of at most 10 million cells (rows x
    columns) - a single column when one column alone is larger - and each
    batch is summarized in memory and released before the next is read, so
    peak memory is several times one batch (plus the corr_target column)
    instead of the whole frame: sorting a column copies it. On a 4-core,
    15 GB machine, a 200M-row file needed more than 12 GB this way, and
    about 1 GB by streaming.

    Either way only the analyzed columns are read, and every query
    re-evaluates the lazy plan: cheap for scan_parquet / scan_ipc, which read
    only the columns a query needs, but slow for scan_csv (every query parses
    the whole file) or a plan with expensive steps such as joins, and a plan
    that samples or shuffles rows without a fixed seed gives each query
    different rows. Collect such plans, or write them to Parquet, first. The
    subtitle shows "LazyFrame" instead of the in-memory size, which a
    LazyFrame does not know.

    Memory guard: before computing a full summary (a DataFrame, or a
    LazyFrame with exact=True), xray() estimates the memory it needs from the
    columns' sizes and compares it with the memory free right now - the
    operating system's available memory, capped by any container (cgroup)
    limit. If it needs more than half of that, xray() summarizes by streaming
    instead and says so in a warning and under the table; if the call asked
    for something only the full summary gives (outlier_method="iqr" or
    "percentile", percentiles, corr_method="spearman"), it raises MemoryError
    instead. The guard cannot be turned off.

    Identifier columns are recognized by name (whole words: "ORDER_ID",
    "Order ID", "customerId", "uuid", "kundenummer", "kunde_nr", "order_no",
    but not "HIDE" or "valid") when their dtype is an integer, a Decimal
    without decimals or a string. They get counts, missing values and
    distinct/duplicate counts (and lengths for strings) but no mean, std,
    quantiles, histogram, outliers or tests, which mean nothing for IDs; a
    note under the table names them. The correlation target is never
    treated as an ID.

    Cardinality (n_unique) is exact at every size for a DataFrame and an
    exact=True LazyFrame. Above 300,000 rows the dominant-value share used for
    quasi-constant detection is skipped for speed (a streaming summary only
    has it for strings with top values).

    The shakiness score adds one point per data quality issue. All checks are
    based on effect sizes (never on test p-values), so the score does not
    depend on sample size:
    - High missingness (pct_missing > missing_threshold)
    - Constant or quasi-constant values (dominant value share >= constant_threshold)
    - ID-like columns (ID-style name, or a near-unique integer or string
      column; near-unique floats, dates, timestamps and durations count as
      measurements, not IDs)
    - Extreme skewness (|skew| > skew_threshold)
    - High kurtosis (|kurtosis| > kurtosis_threshold)
    - High outlier percentage (pct_outliers > outlier_threshold)

    Columns with shakiness_score >= shakiness_threshold are flagged as "⚠ SHAKY"
    for potential issues with parametric statistical models. The normality test
    result is intentionally not part of the score: with more than a few hundred
    rows it rejects nearly every real-world column regardless of how mild the
    deviation is.

    P-values below 0.001 are displayed as "p<0.001".

    The opt_dtype column (expanded mode) shows the dtype ps.fix() /
    convert_datatypes gives the column with default settings (its current
    dtype when they would keep it): integers no narrower than Int32, Float64
    -> Float32 only when every value survives exactly, and Categorical for
    low-cardinality strings. The one addition is Int64 for float columns
    whose values are all whole, finite and within the Int64 range, which goes
    beyond fix(): fix() never converts floats to integers.

    Text taken from the data (column names, top values, samples) is
    HTML-escaped in the Great Tables output, so markup in the data is shown,
    not rendered; the great_tables=False output keeps the raw text.

    When model_usability=True, additional sophisticated quality assessment is performed:
    - Weighted flag system (HM, MM, ID, BN, CV, EO, ES, EK, NN, ZH, UC)
    - 0-100 usability score with actionable recommendations
    - Based on polarsight's proven model usability framework

    Statistical tests require scipy. If not available, tests are skipped
    with informative messages.
    """
    # Start timing for performance measurement
    start_time = time.perf_counter()

    # Both inputs share one code path; see the statistics engine above for
    # how each is executed.
    if isinstance(df, pl.LazyFrame):
        is_lazy = True
    elif isinstance(df, pl.DataFrame):
        is_lazy = False
    else:
        raise TypeError(
            f"xray() expects a Polars DataFrame or LazyFrame, got {type(df).__name__}"
        )
    schema = df.collect_schema()
    n_cols = len(schema)

    # What the caller asked for explicitly: the memory guard below may not
    # quietly replace it.
    explicit_exact_only = []
    if outlier_method in ("iqr", "percentile"):
        explicit_exact_only.append(f"outlier_method={outlier_method!r}")
    if percentiles is not None:
        explicit_exact_only.append("percentiles")
    if corr_method == "spearman":
        explicit_exact_only.append("corr_method='spearman'")

    # A LazyFrame is summarized from streaming statistics unless exact=True.
    streaming = is_lazy and not exact
    if streaming:
        needs_exact = explicit_exact_only
        if needs_exact:
            raise ValueError(
                f"{_options_need(needs_exact)} every value of a column in "
                "memory, which a streaming LazyFrame summary avoids. Pass "
                "exact=True for the full summary, or collect the LazyFrame first."
            )
    if outlier_method is None:
        outlier_method = "zscore" if streaming else "iqr"

    # Set default percentiles (none for a streaming LazyFrame summary)
    if streaming:
        percentiles = []
    elif percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    else:
        if len(percentiles) == 0:
            raise ValueError("percentiles must contain at least one value")
        percentiles = [float(p) for p in percentiles]
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in percentiles):
            raise ValueError("percentiles values must be finite and between 0 and 1")
        percentiles = sorted(set(percentiles))
        percentile_labels = [_percentile_to_label(p) for p in percentiles]
        if len(percentile_labels) != len(set(percentile_labels)):
            raise ValueError("percentiles must produce unique output labels")

    # Validate parameters
    if outlier_method not in ["iqr", "percentile", "zscore"]:
        raise ValueError("outlier_method must be 'iqr', 'percentile', or 'zscore'")

    if outlier_method == "percentile" and outlier_bounds is None:
        raise ValueError(
            "outlier_bounds must be provided when outlier_method='percentile'"
        )

    if outlier_bounds is not None:
        if len(outlier_bounds) != 2:
            raise ValueError("outlier_bounds must be a list of exactly 2 values")
        lower, upper = (float(bound) for bound in outlier_bounds)
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not 0 <= lower < upper <= 1
        ):
            raise ValueError(
                "outlier_bounds must be finite percentiles satisfying "
                "0 <= lower < upper <= 1"
            )
        outlier_bounds = [lower, upper]

    if normality_test not in ["shapiro", "anderson", "ks"]:
        raise ValueError("normality_test must be 'shapiro', 'anderson', or 'ks'")

    if uniformity_test not in ["ks", "chi2"]:
        raise ValueError("uniformity_test must be 'ks' or 'chi2'")

    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")

    # Validate correlation target
    if corr_target:
        if corr_target not in schema:
            raise ValueError(f"Target column '{corr_target}' not found in DataFrame")
        target_dtype = schema[corr_target]
        # Booleans correlate as 0/1, like every Boolean column.
        if not (target_dtype.is_numeric() or target_dtype == pl.Boolean):
            raise ValueError(
                f"Target column '{corr_target}' must be numeric or Boolean, got {target_dtype}"
            )
    elif corr_method is not None:
        raise ValueError(
            "Using corr_method without a valid corr_target column being set is "
            "not allowed. Pass corr_target to choose the column to correlate against."
        )

    # None only exists to distinguish "not supplied" from an explicit choice above.
    corr_method = corr_method if corr_method is not None else "pearson"

    if corr_method not in ["pearson", "spearman"]:
        raise ValueError("corr_method must be 'pearson' or 'spearman'")

    # Filter columns based on include parameter
    all_cols = _get_columns_to_analyze(schema, include)
    if not all_cols:
        if include is None or include == "numeric":
            raise ValueError(
                "No numeric columns found to analyze. "
                "Pass include='all' to analyze all column types, or include='string' for string columns."
            )
        raise ValueError(f"No columns matched include={include!r} in this DataFrame.")

    # Column names pl.col() reads as patterns ('*', '^...$') get stand-ins
    # for the computation; the summary shows the real names.
    stand_ins = literal_name_map(schema.names())
    shown_name = {v: k for k, v in stand_ins.items()}
    target_key = stand_ins.get(corr_target, corr_target) if corr_target else None
    if stand_ins:
        df = df.rename(stand_ins)
        schema = df.collect_schema()
        all_cols = [stand_ins.get(c, c) for c in all_cols]

    # Identifier columns, by name (the correlation target never is one).
    id_columns = frozenset(
        c
        for c in all_cols
        if c != target_key and _is_id_column(shown_name.get(c, c), schema[c])
    )

    # A streaming summary counts the rows in its first pass.
    if not is_lazy:
        n_rows = df.height
    elif not streaming:
        n_rows = int(_collect(df.select(pl.len()), streaming=True).item())

    # Memory guard (see _MEMORY_SHARE): the full summary runs only when its
    # estimated peak fits comfortably in the memory that is free right now.
    memory_note = None
    if not streaming:
        needed_cols = list(dict.fromkeys(all_cols + ([target_key] if target_key else [])))
        kinds = {
            c: "id" if c in id_columns else _column_kind(schema[c]) for c in needed_cols
        }
        need = _exact_memory_need(df, schema, needed_cols, kinds, n_rows)
        available = _available_memory()
        if need > _MEMORY_SHARE * available:
            gb = 1024**3
            what = (
                f"The full summary of these {n_rows:,} rows would need about "
                f"{need / gb:.1f} GB of memory, and {available / gb:.1f} GB is free"
            )
            if explicit_exact_only:
                raise MemoryError(
                    f"{what}. {_options_need(explicit_exact_only)} the full "
                    "summary; without them, xray() summarizes by streaming, which "
                    "does not hold a column in memory."
                )
            streaming = True
            percentiles = []
            outlier_method = "zscore"
            memory_note = (
                f"{what}, so it was summarized by streaming instead: no quartiles, "
                "IQR or MAD; outliers use the z-score rule."
            )
            warnings.warn(f"xray(): {memory_note}", UserWarning, stacklevel=2)

    # Calculate comprehensive statistics for all columns
    stats_data = []

    # IQR needs both quartiles. When the requested percentiles omit either one
    # the statistic is never computed, so no branch may seed it with None -
    # doing so would leave an all-null IQR column in the output.
    iqr_available = 0.25 in percentiles and 0.75 in percentiles

    # Model usability only reads the normality test, and minimal mode shows no
    # test columns, so the uniformity test runs only when it is shown.
    need_extra = expanded or model_usability
    run_normality = need_extra and bool(_check_scipy_availability())
    run_uniformity = run_normality and expanded
    want_all_values = (run_normality and normality_test == "anderson") or (
        run_uniformity and uniformity_test == "chi2"
    )

    # Gather the raw per-column facts (see the statistics engine above), run
    # the statistical tests, then turn both into summary rows below.
    if streaming:
        n_rows, facts = _gather_streaming_facts(
            df if is_lazy else df.lazy(),
            schema,
            all_cols,
            corr_target=target_key,
            need_extra=need_extra,
            want_tests=run_normality,
            id_columns=id_columns,
        )
    else:
        facts = _gather_facts(
            df,
            schema,
            all_cols,
            n_rows=n_rows,
            percentiles=percentiles,
            outlier_method=outlier_method,
            outlier_bounds=outlier_bounds,
            corr_target=target_key,
            corr_method=corr_method,
            need_extra=need_extra,
            want_tests=run_normality,
            want_ks=run_uniformity and uniformity_test == "ks",
            want_all_values=want_all_values,
            id_columns=id_columns,
        )
    test_results = _column_test_results(
        {c: f["tests"] for c, f in facts.items() if "tests" in f},
        normality_test,
        uniformity_test if run_uniformity else None,
    )

    for col in all_cols:
        dtype = schema[col]
        f = facts[col]
        is_bool = dtype == pl.Boolean
        # Identifier columns get counts only (see _is_id_column).
        is_id = col in id_columns
        # Booleans are analyzed as 0/1 numerics (Mean = share of True, etc.)
        is_numeric = (dtype.is_numeric() or is_bool) and not is_id
        n_total = n_rows
        n_valid = f["count"]
        n_missing = f["null_count"]
        n_nan = f["nan_count"]
        # NaN counts as missing: no statistic uses it (see _numeric_expr).
        pct_missing = ((n_missing + (n_nan or 0)) / n_total * 100) if n_total > 0 else 0

        # Initialize column stats
        col_stats = {
            "column": shown_name.get(col, col),
            "dtype": _dtype_label(dtype),
            "count": n_valid,
            "null_count": n_missing,
            "nan_count": n_nan,
            "pct_missing": pct_missing,
        }

        # Basic counts and ratios (nulls excluded). n_unique is None only for
        # a dtype Polars cannot count distinct values of.
        is_large_dataset = n_total > _LARGE_DATASET_ROWS
        n_unique = f.get("n_unique")
        col_stats["n_unique"] = n_unique
        if n_valid == 0:
            col_stats["uniqueness_ratio"] = 0.0
        else:
            col_stats["uniqueness_ratio"] = (
                n_unique / n_valid if n_unique is not None else None
            )

        # Duplicate analysis for historized datasets (among non-null values).
        # Not from a streaming summary: an approximate distinct count makes
        # the difference meaningless (a unique column would show thousands).
        if not streaming:
            if n_unique is None:
                n_duplicates = pct_duplicates = None
            else:
                n_duplicates = max(0, n_valid - n_unique)
                pct_duplicates = n_duplicates / n_valid * 100 if n_valid > 0 else 0.0
            col_stats["n_duplicates"] = n_duplicates
            col_stats["pct_duplicates"] = pct_duplicates

        # Dominant-value share (set exactly by the string branch; computed for
        # small/medium columns otherwise) - feeds quasi-constant detection.
        mode_share = None

        if is_numeric and n_valid > 0:
            # Numeric-specific statistics

            # Basic descriptive stats
            quantile_stats = {
                _percentile_to_label(p): _to_float(f["quantiles"][p])
                for p in percentiles
            }
            col_stats.update(quantile_stats)

            mean_val = _to_float(f["mean"])
            std_val = _to_float(f["std"]) if n_valid > 1 else None
            # Integer columns keep exact min/max (floats lose precision above
            # 2**53); the summary falls back to floats when it must mix both.
            if dtype.is_integer() or is_bool:
                min_val, max_val = int(f["min"]), int(f["max"])
            else:
                min_val, max_val = f["min_float"], f["max_float"]

            col_stats["mean"] = mean_val
            col_stats["std"] = std_val
            col_stats["min"] = min_val
            col_stats["max"] = max_val

            # IQR
            if iqr_available:
                q25 = quantile_stats.get("25%")
                q75 = quantile_stats.get("75%")
                col_stats["iqr"] = (
                    q75 - q25 if q25 is not None and q75 is not None else None
                )

            # Zero/positive/negative counts (a minimal streaming summary,
            # which does not show them, skips them)
            if f.get("n_zero") is not None:
                n_zero = int(f["n_zero"])
                n_pos = int(f["n_pos"])
                n_neg = int(f["n_neg"])
                col_stats["n_zero"] = n_zero
                col_stats["pct_zero"] = n_zero / n_valid * 100
                col_stats["pct_pos"] = n_pos / n_valid * 100
                col_stats["pct_neg"] = n_neg / n_valid * 100

            # Skewness and kurtosis - Polars' native (excess) estimators, which
            # match scipy.stats. Both are always computed so the shakiness
            # score is identical in minimal and expanded mode.
            col_stats["skew"] = _to_float(f["skew"]) if n_valid > 2 else None
            col_stats["kurtosis"] = _to_float(f["kurtosis"]) if n_valid > 2 else None

            # Distribution plot data for nanoplots
            col_stats["distribution_plot"] = f["histogram"]

            # Model usability relies on these metrics even when they are not
            # included in the minimal output table.
            if need_extra:
                # MAD (Median Absolute Deviation); needs the median
                if not streaming:
                    col_stats["mad"] = _to_float(f["mad"])

                # Optimal dtype suggestion
                col_stats["opt_dtype"] = _suggested_dtype(dtype, f, n_total)

                # Statistical tests (if scipy available)
                if run_normality and n_valid > 3:
                    if n_unique is not None and n_unique <= 1:
                        col_stats["normality_test"] = "N/A (constant data)"
                        if run_uniformity:
                            col_stats["uniformity_test"] = "N/A (constant data)"
                    else:
                        normality, uniformity = test_results[col]
                        col_stats["normality_test"] = normality
                        if run_uniformity:
                            col_stats["uniformity_test"] = uniformity

            # Outlier detection (none for Booleans, see _distribution_query)
            n_outliers = f.get("n_outliers")
            if n_outliers is None:
                pct_outliers = None
            else:
                pct_outliers = (n_outliers / n_valid * 100) if n_valid > 0 else 0
            col_stats["n_outliers"] = n_outliers
            col_stats["pct_outliers"] = pct_outliers

            # Correlation with target
            if corr_target:
                if col == target_key:
                    col_stats["correlation"] = None
                    col_stats["correlation_plot"] = None  # No plot for target column
                else:
                    corr_val = f["correlation"]
                    col_stats["correlation"] = corr_val
                    # Single value for horizontal bar nanoplot (fixed -1 to 1 scale)
                    col_stats["correlation_plot"] = corr_val

            # String/temporal-specific columns are N/A for numeric columns
            for stat in _STRING_ALL_COLS + _TEMPORAL_COLS:
                col_stats[stat] = None

        else:
            # ── Non-numeric columns (and numeric ones without values) ────
            is_string = _is_stringy(dtype)
            is_temporal = dtype.is_temporal()

            # Temporal min/max are set by the temporal branch below
            col_stats["earliest"] = None
            col_stats["latest"] = None

            if is_string and n_valid > 0:
                # String-specific statistics (Categorical/Enum analyzed as strings)
                # (A Categorical ID column has no length facts.)
                col_stats["min_length"] = f.get("min_length")
                col_stats["median_length"] = _to_float(f.get("median_length"))
                col_stats["avg_length"] = _to_float(f.get("avg_length"))
                col_stats["max_length"] = f.get("max_length")

                # rank -> (value, count); a streaming summary skips it for
                # strings with very many distinct values.
                ranked = f.get("ranked_values")
                if ranked is None:
                    for stat in ("top", "top_freq", "mode_share", "top_3", "sample_vals"):
                        col_stats[stat] = None
                    col_stats["distribution_plot"] = []
                else:
                    vc_height = n_unique
                    top_value, top_freq = ranked[0]
                    top_freq = int(top_freq)
                    col_stats["top"] = _truncate(top_value)
                    col_stats["top_freq"] = top_freq

                    # Expanded: dominant-category share, top-3, and a spread sample
                    mode_share = top_freq / n_valid  # reused for quasi-constant detection
                    col_stats["mode_share"] = mode_share * 100
                    col_stats["top_3"] = ", ".join(
                        f"{_truncate(ranked[i][0], 20)} ({int(ranked[i][1])})"
                        for i in range(min(3, vc_height))
                    )
                    # Sample of distinct values spread across the frequency range
                    # (most common / mid / rarest) - meaningful at any cardinality.
                    sample_idx = sorted({0, vc_height // 2, vc_height - 1})[:3]
                    col_stats["sample_vals"] = ", ".join(
                        _truncate(ranked[i][0], 20) for i in sample_idx
                    )

                    # Distribution plot - top category frequencies as bar chart
                    n_bars = min(12, vc_height)
                    col_stats["distribution_plot"] = [
                        int(ranked[i][1]) for i in range(n_bars)
                    ]
            elif is_temporal and n_valid > 0:
                # Temporal columns: earliest/latest values
                col_stats["earliest"] = str(f["min"])
                col_stats["latest"] = str(f["max"])
                for stat in _STRING_ALL_COLS:
                    col_stats[stat] = None
                col_stats["distribution_plot"] = []
            else:
                for stat in _STRING_ALL_COLS:
                    col_stats[stat] = None
                col_stats["distribution_plot"] = []

            # Numeric-only statistics are N/A for non-numeric columns
            numeric_only_stats = ["mean", "std", "min", "max"]
            if iqr_available:
                numeric_only_stats.append("iqr")
            for stat in numeric_only_stats:
                col_stats[stat] = None
            for p in percentiles:
                label = _percentile_to_label(p)
                col_stats[label] = None
            col_stats["skew"] = None
            col_stats["n_outliers"] = None
            col_stats["pct_outliers"] = None
            col_stats["n_zero"] = None
            col_stats["pct_zero"] = None
            col_stats["pct_pos"] = None
            col_stats["pct_neg"] = None
            col_stats["kurtosis"] = None

            if need_extra:
                if is_id:
                    not_applicable = "N/A (ID column)"
                elif dtype.is_numeric() or is_bool:
                    not_applicable = "N/A (no values)"
                else:
                    not_applicable = "N/A (non-numeric)"
                if not streaming:
                    col_stats["mad"] = None
                col_stats["opt_dtype"] = _suggested_dtype(dtype, f, n_total)
                col_stats["normality_test"] = not_applicable
                col_stats["uniformity_test"] = not_applicable

            # Correlation with target is N/A for non-numeric columns
            if corr_target:
                col_stats["correlation"] = None
                col_stats["correlation_plot"] = None

        # Dominant-value share for quasi-constant detection. The string branch
        # sets it from its value counts; otherwise it needs the column's mode
        # count, which is only computed for small/medium columns - very large
        # columns fall back to the cardinality heuristic for speed.
        if mode_share is None and n_valid > 0 and not is_large_dataset:
            mode_count = f.get("mode_count")
            mode_share = int(mode_count) / n_valid if mode_count is not None else None

        # Calculate shakiness score
        shakiness_score = _calculate_shakiness_score(
            col_stats,
            missing_threshold,
            constant_threshold,
            skew_threshold,
            kurtosis_threshold,
            outlier_threshold,
            mode_share=mode_share,
        )
        col_stats["shakiness_score"] = shakiness_score
        col_stats["quality_flag"] = (
            "⚠ SHAKY" if shakiness_score >= shakiness_threshold else "✓ OK"
        )

        # Model usability evaluation (if requested)
        if model_usability:
            usability_result = _evaluate_column_usability(
                col_stats, has_correlation=bool(corr_target)
            )
            col_stats["usability_flags"] = usability_result["flag_string"]
            col_stats["usability_score"] = usability_result["score"]
            col_stats["recommendation"] = usability_result["recommendation"]

        stats_data.append(col_stats)

    # Exact integer min/max only when every one fits Int64 and no float column
    # forces the column to Float64 anyway.
    extremes = [
        s[k] for s in stats_data for k in ("min", "max") if s.get(k) is not None
    ]
    if any(isinstance(v, float) for v in extremes) or any(
        not -(2**63) <= v < 2**63 for v in extremes
    ):
        for s in stats_data:
            for k in ("min", "max"):
                if s.get(k) is not None:
                    s[k] = float(s[k])

    # Create DataFrame. Every row is scanned for the schema: with the default
    # 100-row inference, frames with more than 100 analyzed columns of one kind
    # before another failed to build.
    summary_df = pl.DataFrame(stats_data, infer_schema_length=None)

    # Correlation is recorded for every analyzed column so the summary schema
    # stays stable, but nothing correlates with the target when no analyzed
    # column is numeric besides the target itself (e.g. include='string').
    # Drop the pair rather than render two empty columns.
    if corr_target and "correlation" in summary_df.columns:
        if summary_df["correlation"].null_count() == summary_df.height:
            summary_df = summary_df.drop(
                [
                    c
                    for c in ("correlation", "correlation_plot")
                    if c in summary_df.columns
                ]
            )

    # nan_count is shown only when some analyzed column contains NaN.
    has_nan = "nan_count" in summary_df.columns and bool(
        (summary_df["nan_count"].fill_null(0) > 0).any()
    )
    if not has_nan and "nan_count" in summary_df.columns:
        summary_df = summary_df.drop("nan_count")

    # Percentile labels (numeric-only columns)
    percentile_labels = [_percentile_to_label(p) for p in percentiles]

    # When no analyzed column is numeric (e.g. include='string', or a frame with
    # no numeric columns), the numeric-only stats are all empty - drop them so
    # the table isn't padded with columns of None. Booleans count as numeric
    # since they are analyzed as 0/1.
    has_numeric = any(
        (schema[c].is_numeric() or schema[c] == pl.Boolean) and c not in id_columns
        for c in all_cols
    )
    numeric_only_cols = [
        "mean",
        "std",
        "min",
        "max",
        "iqr",
        "skew",
        "kurtosis",
        "mad",
        "n_zero",
        "pct_zero",
        "pct_pos",
        "pct_neg",
        "n_outliers",
        "pct_outliers",
        "normality_test",
        "uniformity_test",
        "correlation",
        "correlation_plot",
    ] + percentile_labels

    # Apply column filtering based on expanded mode
    if expanded:
        final_df = summary_df
        if not has_numeric:
            final_df = final_df.drop(
                [c for c in numeric_only_cols if c in final_df.columns]
            )
        # String/temporal stats are seeded as None on every column so the summary
        # schema stays stable regardless of dtypes. Drop the ones no analyzed
        # column populated - minimal mode already does this via has_string_data.
        unpopulated = [
            c
            for c in _STRING_ALL_COLS + _TEMPORAL_COLS
            if c in final_df.columns and final_df[c].null_count() == final_df.height
        ]
        if unpopulated:
            final_df = final_df.drop(unpopulated)
        # Reorder into the documented group order: basic stats, quantiles,
        # counts & ratios, outliers, distribution, string, temporal,
        # correlation, statistical tests, quality assessment.
        groups = _expanded_column_groups(final_df.columns, percentiles, model_usability)
        ordered = ["column"] + [c for cols in groups.values() for c in cols]
        ordered += [c for c in final_df.columns if c not in ordered]
        final_df = final_df.select(ordered)
    else:
        # Minimal mode - only essential columns (percentile labels generated dynamically)
        essential_cols = ["column", "dtype", "count", "null_count"]
        if has_nan:
            essential_cols.append("nan_count")
        if has_numeric:
            essential_cols += (
                ["mean", "std", "min"] + percentile_labels + ["max", "iqr"]
            )
        essential_cols += ["pct_missing"]
        if has_numeric:
            essential_cols += ["n_outliers", "skew"]

        # Add string-specific columns only when string columns are present
        string_stat_cols = list(_STRING_DEFAULT_COLS)
        has_string_data = any(
            c in summary_df.columns and summary_df[c].null_count() < len(summary_df)
            for c in string_stat_cols
        )
        if has_string_data:
            essential_cols.extend(string_stat_cols)

        # Add temporal columns only when temporal columns are present
        has_temporal_data = any(
            c in summary_df.columns and summary_df[c].null_count() < len(summary_df)
            for c in _TEMPORAL_COLS
        )
        if has_temporal_data:
            essential_cols.extend(_TEMPORAL_COLS)

        essential_cols.append("distribution_plot")

        # Only include the essential columns that exist in the dataframe
        available_cols = [c for c in essential_cols if c in summary_df.columns]

        # Add correlation at the very end if specified
        if corr_target and "correlation" in summary_df.columns:
            available_cols.append("correlation")
        if corr_target and "correlation_plot" in summary_df.columns:
            available_cols.append("correlation_plot")

        # Model usability columns are appended in minimal mode too when requested
        if model_usability:
            available_cols.extend(c for c in _USABILITY_COLS if c in summary_df.columns)

        final_df = summary_df.select(available_cols)

    # Return standard DataFrame if great_tables=False. The *_plot columns are
    # raw nanoplot payloads that only render in Great Tables output.
    if not great_tables:
        final_df = final_df.drop(
            [
                c
                for c in ("distribution_plot", "correlation_plot")
                if c in final_df.columns
            ]
        )
        # Honor the decimals option in plain output too, by rounding the float
        # statistics. Rounding keeps the effect scoped to this frame;
        # pl.Config.set_float_precision() would leak into how every other
        # DataFrame in the session is printed.
        return final_df.with_columns(cs.float().round(decimals))

    # Subtitle detail: the in-memory size of a DataFrame. A LazyFrame has no
    # size until collected, which xray() deliberately avoids.
    source_label = "LazyFrame" if is_lazy else f"{_format_memory_usage(df)} in memory"

    # Build Great Tables object
    if expanded:
        # Full statistics mode
        # Calculate timing
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        gt_table = _build_expanded_gt_table(
            final_df,
            n_rows,
            n_cols,
            source_label,
            execution_time_ms,
            corr_target,
            percentiles,
            decimals,
            sep_mark,
            dec_mark,
            compact,
            pattern,
            title,
            model_usability,
        )
    else:
        # Calculate timing
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        gt_table = _build_minimal_gt_table(
            final_df,
            n_rows,
            n_cols,
            source_label,
            execution_time_ms,
            corr_target,
            decimals,
            sep_mark,
            dec_mark,
            compact,
            pattern,
            title,
            model_usability,
        )
    if memory_note:
        gt_table = gt_table.tab_source_note(memory_note)
    if streaming:
        skipped_tops = [
            c for c in all_cols if facts[c].get("min_length") is not None
            and facts[c].get("ranked_values") is None and c not in id_columns
        ]
        gt_table = gt_table.tab_source_note(
            _streaming_note(
                final_df.columns,
                skipped_top_values=bool(skipped_tops),
                sampled_tests=n_rows > _TEST_EXACT_MAX,
                # After the memory guard, exact=True is no way out.
                suggest_exact=memory_note is None,
            )
        )
    if id_columns:
        names = ", ".join(shown_name.get(c, c) for c in all_cols if c in id_columns)
        gt_table = gt_table.tab_source_note(
            f"Identifier columns (by name), counted but not summarized: {names}."
        )
    return gt_table


def _options_need(options: list[str]) -> str:
    """'percentiles needs' / 'percentiles and corr_method='spearman' need'."""
    return f"{' and '.join(options)} need{'s' if len(options) == 1 else ''}"


def _streaming_note(
    columns: Sequence[str],
    *,
    skipped_top_values: bool,
    sampled_tests: bool,
    suggest_exact: bool = True,
) -> str:
    """Source note of a streaming LazyFrame summary: what differs from exact."""
    parts = ["Summarized by streaming: no column was held in memory."]
    if "n_outliers" in columns:
        parts.append("Outliers use the z-score rule (beyond mean ± 3 std).")
    if "n_unique" in columns:
        parts.append(
            f"Distinct counts above {_DISTINCT_SAMPLE:,} are estimated from a "
            "hash sample (error about 0.1%)."
        )
    if "normality_test" in columns and sampled_tests:
        parts.append(
            f"The tests use a systematic sample of about "
            f"{_NORMALITY_MAX_SAMPLE_SIZE:,} rows."
        )
    if skipped_top_values:
        parts.append(
            f"Top values are left out for strings with more than "
            f"{_TOP_VALUES_MAX_DISTINCT:,} distinct values."
        )
    if suggest_exact:
        parts.append("Pass exact=True for quartiles, IQR and exact counts.")
    return " ".join(parts)


# Helper Functions


def _percentile_to_label(p: float) -> str:
    """Convert percentile float to column label in percent format."""
    return f"{p * 100:.12g}%"


def _is_percentile_label(value: str) -> bool:
    """Return whether a column name is a numeric percentage label."""
    if not value.endswith("%"):
        return False
    try:
        float(value[:-1])
    except ValueError:
        return False
    return True


# String stat column names, kept consistent across all column types so the
# summary DataFrame has a stable schema regardless of dtypes present.
_STRING_DEFAULT_COLS = [
    "top",
    "top_freq",
    "min_length",
    "median_length",
    "avg_length",
    "max_length",
]
_STRING_EXPANDED_COLS = ["mode_share", "top_3", "sample_vals"]
_STRING_ALL_COLS = _STRING_DEFAULT_COLS + _STRING_EXPANDED_COLS

# Temporal stat column names (earliest/latest timestamps)
_TEMPORAL_COLS = ["earliest", "latest"]

# Model usability column names (appended when model_usability=True)
_USABILITY_COLS = ["usability_flags", "usability_score", "recommendation"]


def _expanded_column_groups(
    columns: Sequence[str],
    percentiles: list[float],
    model_usability: bool,
) -> dict[str, list[str]]:
    """Ordered column groups for the expanded view.

    Dict insertion order defines the left-to-right display order of both the
    plain DataFrame output and the Great Tables spanners; each group lists its
    members in display order, filtered to the columns actually present.
    """
    groups = {
        "basic": ["dtype", "opt_dtype", "count", "mean", "std", "min", "max"],
        "quantiles": [_percentile_to_label(p) for p in percentiles],
        "counts": [
            "null_count",
            "nan_count",
            "pct_missing",
            "n_unique",
            "uniqueness_ratio",
            "n_duplicates",
            "pct_duplicates",
            "n_zero",
            "pct_zero",
            "pct_pos",
            "pct_neg",
        ],
        "outliers": ["n_outliers", "pct_outliers"],
        "distribution": ["iqr", "skew", "kurtosis", "mad", "distribution_plot"],
        "string": list(_STRING_ALL_COLS),
        "temporal": list(_TEMPORAL_COLS),
        "correlation": ["correlation", "correlation_plot"],
        "tests": ["normality_test", "uniformity_test"],
        "quality": ["shakiness_score", "quality_flag"]
        + (list(_USABILITY_COLS) if model_usability else []),
    }
    return {name: [c for c in cols if c in columns] for name, cols in groups.items()}


def _to_float(value: Any) -> float | None:
    """float() that passes None through (Polars aggregates of no data)."""
    return None if value is None else float(value)


def _truncate(value: object, limit: int = 30) -> str:
    """Stringify and truncate a value with an ellipsis for table display."""
    s = str(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _finite_values(data: pl.Series | Iterable[float]):
    """Finite values as a float64 NumPy array (NumPy ships with SciPy)."""
    if isinstance(data, pl.Series):
        series = data.cast(pl.Float64)
    else:
        series = pl.Series("values", list(data), dtype=pl.Float64, strict=False)
    return series.filter(series.is_finite()).to_numpy()


def _lilliefors_p_value(values) -> float | None:
    """Lilliefors test: KS distance to the fitted normal, analytical p-value.

    The test runs on a deterministic stride sample of at most
    _NORMALITY_MAX_SAMPLE_SIZE values (xray() passes that sample already).
    The p-value is Dallal and Wilkinson's (1986) approximation below 0.1 and
    Stephens' (1974) polynomial above it, as in R's nortest::lillie.test, so
    small p-values are resolved rather than floored: a Monte Carlo p-value from
    B simulations cannot go below 1 / (B + 1). None for constant values.
    """
    if len(values) > _NORMALITY_MAX_SAMPLE_SIZE:
        step = len(values) / _NORMALITY_MAX_SAMPLE_SIZE
        values = values[[int(i * step) for i in range(_NORMALITY_MAX_SAMPLE_SIZE)]]
    import numpy as np  # ships with SciPy

    x = np.sort(values)
    n = len(x)
    sd = float(x.std(ddof=1))
    if not sd > 0:
        return None
    cdf = stats.norm.cdf((x - x.mean()) / sd)
    i = np.arange(1, n + 1)
    d = max(float((i / n - cdf).max()), float((cdf - (i - 1) / n).max()))

    kd, nd = (d, n) if n <= 100 else (d * (n / 100) ** 0.49, 100)
    p = math.exp(
        -7.01256 * kd**2 * (nd + 2.78019)
        + 2.99587 * kd * math.sqrt(nd + 2.78019)
        - 0.122119
        + 0.974598 / math.sqrt(nd)
        + 1.67997 / nd
    )
    if p > 0.1:
        kk = (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n)) * d
        # Polynomial in kk, coefficients from the constant term up.
        if kk <= 0.302:
            coefficients = (1.0,)
        elif kk <= 0.5:
            coefficients = (2.76773, -19.828315, 80.709644, -138.55152, 81.218052)
        elif kk <= 0.9:
            coefficients = (-4.901232, 40.662806, -97.490286, 94.029866, -32.355711)
        elif kk <= 1.31:
            coefficients = (6.198765, -19.558097, 23.186922, -12.234627, 2.423045)
        else:
            coefficients = (0.0,)
        p = sum(c * kk**k for k, c in enumerate(coefficients))
    return min(max(p, 0.0), 1.0)


def _format_p_value(p_value: float) -> str:
    """Format a p-value for table display; tiny values render as 'p<0.001'."""
    return "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"


def _test_normality(
    data: pl.Series | Iterable[float], test_type: str, *, n: int | None = None
) -> str:
    """Perform a normality test and return a compact formatted result.

    Returns ``"NORMAL (p=…)"`` / ``"NON-NORMAL (p<0.001)"`` at alpha=0.05.
    The method itself is intentionally omitted from the cell text; it is
    documented on :func:`xray` (``normality_test`` parameter).

    - ``"shapiro"``: Shapiro-Wilk. Above 5,000 values it silently falls back
      to the fitted-normal KS goodness-of-fit test below, where Shapiro-Wilk
      p-values become unreliable.
    - ``"anderson"``: Anderson-Darling. Uses SciPy's interpolated p-value when
      available; older SciPy falls back to the 5% critical value and returns
      the verdict without a p-value.
    - ``"ks"``: Lilliefors test (Kolmogorov-Smirnov against a fitted normal)
      with an analytical p-value, on a deterministic sample of at most 3,000
      values (see _lilliefors_p_value).

    ``n`` is the number of finite values ``data`` was sampled from, when
    ``data`` is xray()'s stride sample of a larger column.
    """
    if not _check_scipy_availability():
        return "N/A (scipy not available)"

    values = _finite_values(data)
    n = len(values) if n is None else n
    if n < 3:
        return "N/A (insufficient data)"

    try:
        p_value: float | None
        if test_type == "shapiro":
            if n > _TEST_EXACT_MAX:
                # Shapiro-Wilk is unreliable for large samples, so fall back to
                # the Lilliefors (fitted-normal KS) test.
                p_value = _lilliefors_p_value(values)
            else:
                _stat, p_value = stats.shapiro(values)

        elif test_type == "anderson":
            if "method" in inspect.signature(stats.anderson).parameters:
                result = stats.anderson(
                    values,
                    dist="norm",
                    method="interpolate",
                )
                p_value = result.pvalue
                # SciPy interpolates within its table of critical values and
                # clips outside it: 0.01 means "0.01 or less", 0.15 "0.15 or
                # more". Show the bound as a bound.
                result_str = "NORMAL" if p_value > 0.05 else "NON-NORMAL"
                if p_value <= _ANDERSON_P_RANGE[0]:
                    return f"{result_str} (p≤{_ANDERSON_P_RANGE[0]:.3f})"
                if p_value >= _ANDERSON_P_RANGE[1]:
                    return f"{result_str} (p≥{_ANDERSON_P_RANGE[1]:.3f})"
            else:
                # Older SciPy reports no p-value; compare against the 5%
                # critical value instead.
                result = stats.anderson(values, dist="norm")
                if result.statistic < result.critical_values[2]:
                    return "NORMAL"
                return "NON-NORMAL"

        elif test_type == "ks":
            if len(values) < 5:
                return "N/A (insufficient data)"
            p_value = _lilliefors_p_value(values)

        if p_value is None:
            return "N/A (constant data)"
        result_str = "NORMAL" if p_value > 0.05 else "NON-NORMAL"
        return f"{result_str} ({_format_p_value(p_value)})"

    except Exception as e:
        return f"Error ({test_type}): {str(e)[:20]}"


def _uniformity_result(p_value: float) -> str:
    result_str = "UNIFORM" if p_value > 0.05 else "NON-UNIFORM"
    return f"{result_str} ({_format_p_value(p_value)})"


def _test_uniformity(data: pl.Series | Iterable[float], test_type: str) -> str:
    """Perform a uniformity test and return a compact formatted result.

    Returns ``"UNIFORM (p=…)"`` / ``"NON-UNIFORM (p<0.001)"`` at alpha=0.05.
    The method itself is intentionally omitted from the cell text; it is
    documented on :func:`xray` (``uniformity_test`` parameter).

    - ``"ks"``: Kolmogorov-Smirnov against the uniform distribution, after
      min-max normalizing the values to [0, 1] (asymptotic p-value above
      5,000 values).
    - ``"chi2"``: Chi-square goodness of fit over equal-width histogram bins.
    """
    if not _check_scipy_availability():
        return "N/A (scipy not available)"

    values = _finite_values(data)
    if len(values) < 5:
        return "N/A (insufficient data)"

    try:
        if test_type == "ks":
            min_val, max_val = values.min(), values.max()
            if min_val == max_val:
                return "N/A (constant data)"
            normalized = (values - min_val) / (max_val - min_val)
            ks_method = "asymp" if len(values) > _TEST_EXACT_MAX else "auto"
            _stat, p_value = stats.kstest(normalized, "uniform", method=ks_method)

        elif test_type == "chi2":
            # Chi-square goodness of fit over equal-width bins
            n_bins = min(10, math.isqrt(len(values)))
            observed = (
                pl.Series("values", values)
                .hist(bin_count=n_bins)
                .get_column("count")
                .to_list()
            )
            expected_count = len(values) / len(observed)

            # Remove bins with very low expected frequency
            if expected_count < 5 or len(observed) < 2:
                return "N/A (insufficient bins)"

            expected = [expected_count] * len(observed)
            _stat, p_value = stats.chisquare(observed, expected)

        return _uniformity_result(p_value)

    except Exception as e:
        return f"Error ({test_type}): {str(e)[:20]}"


def _ks_uniform_asymptotic(d_stat: float, n: int, lo: float, hi: float) -> str:
    """KS uniformity verdict from a statistic computed in Polars.

    For columns above 5,000 values: the same asymptotic p-value as
    ``scipy.stats.kstest(..., method="asymp")``, without moving the column
    into Python.
    """
    if not _check_scipy_availability():
        return "N/A (scipy not available)"
    if lo == hi:
        return "N/A (constant data)"
    try:
        return _uniformity_result(float(stats.kstwobign.sf(d_stat * n**0.5)))
    except Exception as e:
        return f"Error (ks): {str(e)[:20]}"


def _column_test_results(
    tests: dict[str, dict[str, Any]],
    normality_test: str,
    uniformity_test: str | None,
) -> dict[str, tuple[str, str | None]]:
    """Normality and (unless None) uniformity verdicts for each column.

    ``tests`` holds each column's test inputs from the numeric query: its
    finite-value count ``n``, a ``sample`` (every value up to 5,000, else the
    stride sample), the KS uniformity statistic with its range, and all
    ``values`` when a test needs them. Columns run concurrently: the SciPy
    calls spend most of their time in NumPy.
    """

    def run(t: dict[str, Any]) -> tuple[str, str | None]:
        n, sample = t["n"], t["sample"]
        large = n > _TEST_EXACT_MAX
        if normality_test == "anderson" and large:
            normality = _test_normality(t["values"], "anderson")
        else:
            normality = _test_normality(sample, normality_test, n=n)
        uniformity = None
        if uniformity_test == "ks":
            if large:
                uniformity = _ks_uniform_asymptotic(t["ks_d"], n, t["lo"], t["hi"])
            else:
                uniformity = _test_uniformity(sample, "ks")
        elif uniformity_test == "chi2":
            uniformity = _test_uniformity(t["values"] if large else sample, "chi2")
        return normality, uniformity

    if not tests:
        return {}
    with ThreadPoolExecutor() as pool:
        return dict(zip(tests, pool.map(run, tests.values())))


def _calculate_shakiness_score(
    col_stats: dict,
    missing_threshold: float,
    constant_threshold: float,
    skew_threshold: float,
    kurtosis_threshold: float,
    outlier_threshold: float,
    mode_share: float | None = None,
) -> int:
    """Calculate shakiness score based on data quality indicators.

    One point per issue, all evaluated on effect sizes rather than p-values so
    the score is sample-size independent and identical in minimal and expanded
    mode:

    - missing share > ``missing_threshold``
    - constant or quasi-constant (dominant value share >= ``constant_threshold``)
    - ID-like (name/cardinality heuristic; high-uniqueness floats are treated
      as measurements, not identifiers)
    - ``|skew|`` > ``skew_threshold``
    - ``|kurtosis|`` > ``kurtosis_threshold``
    - outlier share > ``outlier_threshold``

    The normality test result is deliberately NOT part of the score: with more
    than a few hundred rows the test rejects nearly every real-world column,
    and skew/kurtosis already capture departure from normality by magnitude.
    """
    score = 0

    # High missingness
    if col_stats.get("pct_missing", 0) > missing_threshold * 100:
        score += 1

    # Constant/quasi-constant: flag when a single value dominates the column.
    n_unique_val = col_stats.get("n_unique")
    if n_unique_val == 1:
        score += 1
    elif mode_share is not None:
        # Exact dominant-value share: flag if the most common value covers at
        # least `constant_threshold` (e.g. 0.99 = 99%) of the non-null values.
        if mode_share >= constant_threshold:
            score += 1

    # ID-like columns. Uses the shared name/cardinality heuristic so that
    # continuous float measurements (which are ~100% unique by nature) do not
    # collect a free penalty point.
    if _is_likely_id_column(col_stats.get("column", ""), col_stats):
        score += 1

    # Extreme skewness
    skewness = col_stats.get("skew")
    if skewness is not None and abs(skewness) > skew_threshold:
        score += 1

    # High kurtosis
    kurtosis = col_stats.get("kurtosis")
    if kurtosis is not None and abs(kurtosis) > kurtosis_threshold:
        score += 1

    # Outlier-heavy
    pct_outliers = col_stats.get("pct_outliers", 0)
    if pct_outliers is not None and pct_outliers > outlier_threshold * 100:
        score += 1

    return score


# Model Usability Functions (inspired by polarsight)


def _check_missing_values_usability(stats: dict) -> set[str]:
    """Check for missing value flags using polarsight thresholds."""
    flags = set()
    null_pct = stats.get("pct_missing", 0)

    if null_pct > 90.0:  # High missing (>90%)
        flags.add("HM")
    elif null_pct > 50.0:  # Moderate missing (>50%)
        flags.add("MM")

    return flags


# Dtype names (by prefix) whose near-unique columns count as ID-like.
_ID_LIKE_DTYPES = ("Int", "UInt", "String", "Categorical", "Enum")


def _is_likely_id_column(col_name: str, stats: dict) -> bool:
    """
    Detect likely identifier columns from tokenized names and cardinality.

    Floating-point measures are not classified as IDs based on uniqueness alone.
    """
    tokenized_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", col_name)
    name_tokens = set(re.findall(r"[a-z0-9]+", tokenized_name.lower()))
    has_id_name = bool(name_tokens & {"id", "key", "code", "num", "number", "no"})

    # Get statistics
    n_unique = stats.get("n_unique") or 0
    n_total = stats.get("count", 0) + stats.get(
        "null_count", 0
    )  # Total including nulls
    uniqueness_ratio = stats.get("uniqueness_ratio") or 0

    if n_total == 0:
        return False

    # High cardinality check (even with duplicates, many unique values suggests ID)
    high_cardinality = n_unique > 1000  # More than 1000 unique values

    # Without an ID-style name, only near-unique integer and string columns
    # look like IDs. Floats are measurements, and dates, timestamps and
    # durations are naturally near-unique without identifying anything.
    if not has_id_name and uniqueness_ratio > 0.95:
        return str(stats.get("dtype", "")).startswith(_ID_LIKE_DTYPES)

    # For columns with ID-like names, be more lenient with uniqueness
    # This handles historized datasets where IDs are repeated across time periods
    if has_id_name:
        # If it has an ID-like name and high cardinality, it's likely an ID
        if high_cardinality:
            return True
        # Even with lower cardinality, if uniqueness is reasonable, flag it
        if uniqueness_ratio > 0.1 and n_unique > 100:
            return True

    return False


def _check_unique_values_usability(stats: dict) -> set[str]:
    """Check for unique value related flags using enhanced ID detection."""
    flags = set()
    n_unique_val = stats.get("n_unique")
    count = stats.get("count", 1)
    col_name = stats.get("column", "")

    if count > 0:
        if n_unique_val == 1:  # Constant value
            flags.add("CV")
        elif n_unique_val == 2:  # Binary column
            flags.add("BN")
        elif _is_likely_id_column(col_name, stats):  # Enhanced ID detection
            flags.add("ID")

    return flags


def _check_distribution_usability(stats: dict) -> set[str]:
    """Check for distribution-related flags using polarsight thresholds."""
    flags = set()

    # Check outliers (None for non-numeric columns)
    outlier_pct = stats.get("pct_outliers") or 0
    if outlier_pct > 10.0:  # Extreme outliers (>10%)
        flags.add("EO")

    # Check skew
    skew = stats.get("skew")
    if skew is not None and abs(skew) > 3.0:  # Extreme skew (|skew| > 3)
        flags.add("ES")

    # Check kurtosis
    kurtosis = stats.get("kurtosis")
    if (
        kurtosis is not None and abs(kurtosis) > 7.0
    ):  # Extreme kurtosis (|kurtosis| > 7)
        flags.add("EK")

    # Check normality
    normality_test = stats.get("normality_test", "")
    if "NON-NORMAL" in normality_test:  # Non-normal distribution
        flags.add("NN")

    # Check zero values (None for non-numeric columns)
    zero_pct = stats.get("pct_zero") or 0
    if zero_pct > 80.0:  # High zero values (>80%)
        flags.add("ZH")

    return flags


def _check_correlation_reliability_usability(
    stats: dict, has_correlation: bool
) -> set[str]:
    """Check if correlation values are reliable using polarsight logic."""
    flags = set()

    # If column is non-normal and we're showing correlations, flag it
    if has_correlation and "NON-NORMAL" in stats.get("normality_test", ""):
        flags.add("UC")

    return flags


def _calculate_usability_score(flags: set[str]) -> tuple[float, str]:
    """
    Calculate model usability score based on flags using polarsight weights.
    Returns score (0-100, higher is better) and recommendation.
    """
    # Define flag weights (from polarsight)
    flag_weights = {
        "HM": 5.0,  # High missing values (>90%)
        "MM": 3.0,  # Moderate missing values (>50%)
        "ID": 4.0,  # ID-like column (>95% unique values)
        "BN": 1.0,  # Binary column (exactly 2 unique values)
        "CV": 5.0,  # Constant value (only 1 unique value)
        "EO": 2.5,  # Extreme outliers (>10% outliers)
        "ES": 2.0,  # Extreme skew (|skew| > 3)
        "EK": 2.0,  # Extreme kurtosis (|kurtosis| > 7)
        "NN": 1.5,  # Non-normal distribution (p-value < 0.05)
        "ZH": 2.5,  # High zero values (>80%)
        "UC": 1.0,  # Unreliable correlation (non-normal with correlation)
    }

    if not flags:
        return 100.0, "Good for modeling"

    # Calculate weighted penalty
    total_weight = sum(flag_weights.get(flag, 0) for flag in flags)

    # Maximum possible weight (if all flags were present)
    max_weight = sum(flag_weights.values())

    # Score from 0-100 (higher is better)
    score = max(0, 100 - (total_weight / max_weight * 100))

    # Hard-drop flags must not be paired with a deceptively high score.
    if flags & {"CV", "HM", "ID"}:
        score = min(score, 20.0)

    # Generate recommendation based on flags
    if "CV" in flags:
        recommendation = "Drop - constant value"
    elif "HM" in flags:
        recommendation = "Drop - too many missing values"
    elif "ID" in flags:
        recommendation = "Drop - likely an ID column"
    elif score < 30:
        recommendation = "Use with extreme caution"
    elif score < 50:
        recommendation = "Use with caution"
    elif score < 70:
        recommendation = "Review before using"
    elif score < 90:
        recommendation = "Minor issues - review"
    else:
        recommendation = "Good for modeling"

    return score, recommendation


def _evaluate_column_usability(
    stats: dict, has_correlation: bool = False
) -> dict[str, Any]:
    """
    Evaluate column usability for modeling using polarsight approach.

    Returns dictionary with:
    - flags: Set of flag codes
    - flag_string: Comma-separated flag codes
    - score: Usability score (0-100)
    - recommendation: Text recommendation
    """
    flags = set()

    # Check all flag categories
    flags.update(_check_missing_values_usability(stats))
    flags.update(_check_unique_values_usability(stats))
    flags.update(_check_distribution_usability(stats))
    flags.update(_check_correlation_reliability_usability(stats, has_correlation))

    # Calculate score and recommendation
    score, recommendation = _calculate_usability_score(flags)

    return {
        "flags": flags,
        "flag_string": ",".join(sorted(flags)) if flags else "-",
        "score": score,
        "recommendation": recommendation,
    }


def _sanitize_nanoplot_column(
    summary_df: pl.DataFrame, column: str, *, list_payload: bool
) -> tuple[pl.DataFrame, bool]:
    """Normalize nanoplot columns to numeric payloads to avoid GT rendering errors."""
    if column not in summary_df.columns:
        return summary_df, False

    values = summary_df.get_column(column).to_list()
    cleaned: list[object] = []
    has_valid_payload = False

    if list_payload:
        for value in values:
            if isinstance(value, (list, tuple)):
                numeric_values: list[float] = []
                for item in value:
                    if isinstance(item, Real) and math.isfinite(float(item)):
                        numeric_values.append(float(item))
                if numeric_values:
                    has_valid_payload = True
                    cleaned.append(numeric_values)
                else:
                    cleaned.append(None)
            elif isinstance(value, Real) and math.isfinite(float(value)):
                has_valid_payload = True
                cleaned.append([float(value)])
            else:
                cleaned.append(None)
    else:
        for value in values:
            if isinstance(value, Real) and math.isfinite(float(value)):
                has_valid_payload = True
                cleaned.append(float(value))
            else:
                cleaned.append(None)

    return summary_df.with_columns(pl.Series(column, cleaned)), has_valid_payload


# Correlation bar geometry. The track is a fixed pixel width representing the
# whole -1..1 range, so bar length reads as correlation strength directly -
# both between rows and between separate tables.
_CORR_TRACK_WIDTH_PX = 120
_CORR_TRACK_HEIGHT_PX = 12
_CORR_BAR_HEIGHT_PX = 6
_CORR_POSITIVE_COLOR = "#4A90E2"
_CORR_NEGATIVE_COLOR = "#E24A4A"
_CORR_TRACK_COLOR = "#eceef1"
_CORR_ZERO_TICK_COLOR = "#adb5bd"
# Text shown for missing values in the Great Tables output (statistics that
# do not apply to a column, e.g. a mean for a String column).
_MISSING_TEXT = "-"
_CORR_MISSING_TEXT = _MISSING_TEXT


def _correlation_bar_html(value: object) -> str:
    """Render one correlation as a bar on a fixed-width -1..1 track.

    Great Tables' ``fmt_nanoplot`` cannot do this: a scalar payload takes its
    single-horizontal-bar path, which rescales to the observed value range and
    ignores ``expand_x``/``expand_y`` entirely, so every correlation came out
    the same length regardless of magnitude.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return _CORR_MISSING_TEXT
    number = float(value)
    if not math.isfinite(number):
        return _CORR_MISSING_TEXT

    clamped = max(-1.0, min(1.0, number))
    half = _CORR_TRACK_WIDTH_PX / 2
    length = abs(clamped) * half
    if 0 < length < 1:
        length = 1.0  # keep very weak correlations from vanishing entirely
    offset = half if clamped >= 0 else half - length
    color = _CORR_POSITIVE_COLOR if clamped >= 0 else _CORR_NEGATIVE_COLOR
    bar_top = (_CORR_TRACK_HEIGHT_PX - _CORR_BAR_HEIGHT_PX) / 2

    return (
        f'<div class="ps-corr-track" style="position:relative;'
        f"width:{_CORR_TRACK_WIDTH_PX}px;height:{_CORR_TRACK_HEIGHT_PX}px;"
        f'margin:0 auto;background:{_CORR_TRACK_COLOR};border-radius:2px;">'
        f'<div style="position:absolute;left:{half - 0.5}px;top:0;width:1px;'
        f'height:{_CORR_TRACK_HEIGHT_PX}px;background:{_CORR_ZERO_TICK_COLOR};"></div>'
        f'<div class="ps-corr-bar" style="position:absolute;left:{offset}px;'
        f"top:{bar_top}px;width:{length}px;height:{_CORR_BAR_HEIGHT_PX}px;"
        f'background:{color};border-radius:1px;"></div>'
        f"</div>"
    )


def _apply_correlation_columns(
    gt_table: GT,
    corr_target: str | None,
    summary_df: pl.DataFrame,
    has_corr_data: bool,
) -> GT:
    """Format the correlation pair shared by both table modes."""
    if not (corr_target and "correlation" in summary_df.columns):
        return gt_table

    corr_cols = [
        c for c in ("correlation", "correlation_plot") if c in summary_df.columns
    ]

    if has_corr_data and "correlation_plot" in summary_df.columns:
        gt_table = gt_table.fmt(_correlation_bar_html, columns="correlation_plot")

    return (
        gt_table.sub_missing(columns=["correlation"], missing_text=_CORR_MISSING_TEXT)
        .tab_spanner(label=f"Correlation with '{corr_target}'", columns=corr_cols)
        .cols_align(align="center", columns=corr_cols)
    )


def _numeric_format_columns(summary_df: pl.DataFrame) -> list[str]:
    """Scalar numeric columns, formatted with the decimals/separator options.

    All of them route through fmt_number: with drop_trailing_zeros, integers
    render as before ("4,090,836") while compact output honors the decimals
    option ("4.09M") - fmt_integer has no decimals parameter.
    """
    return [
        column
        for column, dtype in summary_df.schema.items()
        if dtype.is_numeric() and column != "correlation_plot"
    ]


# Summary columns holding text that comes from the user's data.
_USER_TEXT_COLUMNS = ("column", "top", "top_3", "sample_vals", "earliest", "latest")


@functools.lru_cache(maxsize=None)
def _great_tables_escapes_cells() -> bool:
    """Whether the installed Great Tables escapes plain string cells.

    Great Tables 1.x does; 0.x writes them into the HTML as they are. Titles,
    column labels and spanners are escaped by every version.
    """
    from great_tables import GT

    return "&lt;" in GT(pl.DataFrame({"probe": ["<"]})).as_raw_html()


def _escape_user_text(summary_df: pl.DataFrame) -> pl.DataFrame:
    """HTML-escape text from the data before it reaches Great Tables.

    Great Tables 0.x writes plain string cells into the HTML as they are, so a
    value such as "<NA>" vanished as an unknown tag and markup in the data
    rendered (or ran) in the notebook. Versions that escape cells themselves
    are left to it, so nothing is escaped twice.
    """
    if _great_tables_escapes_cells():
        return summary_df
    exprs = [
        pl.col(c)
        .str.replace_all("&", "&amp;", literal=True)
        .str.replace_all("<", "&lt;", literal=True)
        .str.replace_all(">", "&gt;", literal=True)
        for c in _USER_TEXT_COLUMNS
        if summary_df.schema.get(c) == pl.String
    ]
    return summary_df.with_columns(exprs) if exprs else summary_df


def _format_nanoplot_value(value: float) -> str:
    """Format a numeric nanoplot label with three significant digits."""
    return format(value, ".3g")


def _apply_nanoplots(gt_table: GT, has_hist_data: bool) -> GT:
    """Apply histogram nanoplot formatting shared by both table modes."""
    # Histogram nanoplots for the distribution_plot column
    if has_hist_data:
        try:
            from great_tables import nanoplot_options

            gt_table = gt_table.fmt_nanoplot(
                columns="distribution_plot",
                plot_type="bar",
                options=nanoplot_options(
                    data_bar_stroke_width=0,  # No gaps between bars (like histogram)
                    data_bar_fill_color="#4A90E2",
                    show_data_line=False,
                    show_data_area=False,
                    y_val_fmt_fn=_format_nanoplot_value,
                    y_axis_fmt_fn=_format_nanoplot_value,
                ),
            )
        except ImportError:
            # If nanoplot_options not available, use basic nanoplot
            gt_table = gt_table.fmt_nanoplot(
                columns="distribution_plot", plot_type="bar"
            )
        except Exception:
            # If nanoplot formatting fails, continue without it
            pass

    return gt_table


def _build_minimal_gt_table(
    summary_df: pl.DataFrame,
    n_rows: int,
    n_cols: int,
    source_label: str,
    execution_ms: float,
    corr_target: str | None,
    decimals: int,
    sep_mark: str,
    dec_mark: str,
    compact: bool,
    pattern: str | None,
    title: str | None,
    model_usability: bool = False,
) -> GT:
    """Build minimal Great Tables object."""
    summary_df, has_hist_data = _sanitize_nanoplot_column(
        summary_df, "distribution_plot", list_payload=True
    )
    summary_df, has_corr_data = _sanitize_nanoplot_column(
        summary_df, "correlation_plot", list_payload=False
    )

    # Determine column organization (detect percentile columns dynamically)
    pct_cols = sorted(
        [c for c in summary_df.columns if _is_percentile_label(c)],
        key=lambda x: float(x[:-1]),
    )
    basic_cols = (
        ["dtype", "count", "null_count", "nan_count", "mean", "std", "min"]
        + pct_cols
        + ["max"]
    )
    essential_cols = ["iqr", "pct_missing", "n_outliers", "skew"]
    string_cols = list(_STRING_DEFAULT_COLS)
    temporal_cols = list(_TEMPORAL_COLS)
    quality_cols = list(_USABILITY_COLS) if model_usability else []

    # Filter to existing columns and ensure all are strings
    basic_cols = [str(c) for c in basic_cols if c in summary_df.columns]
    essential_cols = [str(c) for c in essential_cols if c in summary_df.columns]
    # Only include string/temporal stats columns if there's at least one non-null value
    string_cols = [
        str(c)
        for c in string_cols
        if c in summary_df.columns and summary_df[c].null_count() < len(summary_df)
    ]
    temporal_cols = [
        str(c)
        for c in temporal_cols
        if c in summary_df.columns and summary_df[c].null_count() < len(summary_df)
    ]
    quality_cols = [str(c) for c in quality_cols if c in summary_df.columns]

    from great_tables import GT

    try:
        # Use custom title or default
        table_title = title if title is not None else "🔬 DataFrame X-ray"

        gt_table = GT(_escape_user_text(summary_df)).tab_header(
            title=table_title,
            subtitle=f"Dataset: {n_rows:,} rows × {n_cols} columns ({source_label}) - X-rayed in {_format_execution_time(execution_ms)}",
        )
    except Exception as e:
        # If GT creation fails, return a basic table without advanced formatting
        raise ValueError(
            f"Great Tables formatting failed. Try using great_tables=False. Error: {e}"
        )

    # Add spanners only for non-empty column groups
    if basic_cols:
        gt_table = gt_table.tab_spanner(label="Basic Statistics", columns=basic_cols)
    if essential_cols:
        gt_table = gt_table.tab_spanner(label="Key Metrics", columns=essential_cols)
    if string_cols:
        gt_table = gt_table.tab_spanner(label="String Statistics", columns=string_cols)
    if temporal_cols:
        gt_table = gt_table.tab_spanner(label="Temporal", columns=temporal_cols)
    if quality_cols:
        gt_table = gt_table.tab_spanner(
            label="Quality Assessment", columns=quality_cols
        )

    gt_table = gt_table.fmt_number(
        columns=_numeric_format_columns(summary_df),
        decimals=decimals,
        drop_trailing_zeros=True,  # 1.00 -> 1, but 1.58 keeps its decimals
        sep_mark=sep_mark,
        dec_mark=dec_mark,
        compact=compact,
        **({"pattern": pattern} if pattern is not None else {}),
    )

    # Alignment - free-text usability columns are left-aligned
    center_cols = [
        str(c)
        for c in (
            basic_cols
            + essential_cols
            + [c for c in string_cols if c != "top"]
            + temporal_cols
            + [c for c in quality_cols if c == "usability_score"]
        )
    ]
    left_cols = (
        ["column"]
        + [c for c in ["top"] if c in string_cols]
        + [c for c in quality_cols if c != "usability_score"]
    )

    gt_table = (
        gt_table.cols_align(align="center", columns=center_cols)
        .cols_align(align="left", columns=left_cols)
        .tab_options(
            table_font_size="13px",
            heading_background_color="#f8f9fa",
            column_labels_background_color="#e9ecef",
        )
    )

    gt_table = _apply_correlation_columns(
        gt_table, corr_target, summary_df, has_corr_data
    )
    gt_table = _apply_nanoplots(gt_table, has_hist_data)

    return gt_table.sub_missing(missing_text=_MISSING_TEXT)


def _build_expanded_gt_table(
    summary_df: pl.DataFrame,
    n_rows: int,
    n_cols: int,
    source_label: str,
    execution_ms: float,
    corr_target: str | None,
    percentiles: list[float],
    decimals: int,
    sep_mark: str,
    dec_mark: str,
    compact: bool,
    pattern: str | None,
    title: str | None,
    model_usability: bool = False,
) -> GT:
    """Build expanded Great Tables object with all statistics."""
    summary_df, has_hist_data = _sanitize_nanoplot_column(
        summary_df, "distribution_plot", list_payload=True
    )
    summary_df, has_corr_data = _sanitize_nanoplot_column(
        summary_df, "correlation_plot", list_payload=False
    )

    # Organize columns by category (shared with the DataFrame ordering in xray()).
    # summary_df arrives already ordered; the spanners just label the groups.
    groups = _expanded_column_groups(summary_df.columns, percentiles, model_usability)
    basic_cols = groups["basic"]
    quantile_cols = groups["quantiles"]
    count_cols = groups["counts"]
    outlier_cols = groups["outliers"]
    distribution_cols = groups["distribution"]
    string_stat_cols = groups["string"]
    temporal_stat_cols = groups["temporal"]
    test_cols = groups["tests"]
    quality_cols = groups["quality"]

    from great_tables import GT

    try:
        # Use custom title or default
        table_title = title if title is not None else "🔬 Expanded Statistics"

        gt_table = GT(_escape_user_text(summary_df)).tab_header(
            title=table_title,
            subtitle=f"Dataset: {n_rows:,} rows × {n_cols} columns ({source_label}) - X-rayed in {_format_execution_time(execution_ms)}",
        )
    except Exception as e:
        # If GT creation fails, return a basic table without advanced formatting
        raise ValueError(
            f"Great Tables formatting failed. Try using great_tables=False. Error: {e}"
        )

    # Add spanners only for non-empty column groups, in display order
    if basic_cols:
        gt_table = gt_table.tab_spanner(label="Basic Statistics", columns=basic_cols)
    if quantile_cols:
        gt_table = gt_table.tab_spanner(label="Quantiles", columns=quantile_cols)
    if count_cols:
        gt_table = gt_table.tab_spanner(label="Counts & Ratios", columns=count_cols)
    if outlier_cols:
        gt_table = gt_table.tab_spanner(label="Outliers", columns=outlier_cols)
    if distribution_cols:
        gt_table = gt_table.tab_spanner(label="Distribution", columns=distribution_cols)
    if string_stat_cols:
        gt_table = gt_table.tab_spanner(
            label="String Statistics", columns=string_stat_cols
        )
    if temporal_stat_cols:
        gt_table = gt_table.tab_spanner(label="Temporal", columns=temporal_stat_cols)
    if test_cols:
        gt_table = gt_table.tab_spanner(label="Statistical Tests", columns=test_cols)
    if quality_cols:
        gt_table = gt_table.tab_spanner(
            label="Quality Assessment", columns=quality_cols
        )

    gt_table = gt_table.fmt_number(
        columns=_numeric_format_columns(summary_df),
        decimals=decimals,
        drop_trailing_zeros=True,  # 1.00 -> 1, but 1.58 keeps its decimals
        sep_mark=sep_mark,
        dec_mark=dec_mark,
        compact=compact,
        **({"pattern": pattern} if pattern is not None else {}),
    )

    # Alignment - free-text string columns are left-aligned, the rest centered
    text_string_cols = {"top", "top_3", "sample_vals"}
    center_cols = [
        str(c)
        for c in (
            [c for c in basic_cols if c != "opt_dtype"]
            + quantile_cols
            + distribution_cols
            + count_cols
            + outlier_cols
            + [c for c in string_stat_cols if c not in text_string_cols]
            + temporal_stat_cols
            + ["shakiness_score"]
            + (
                ["usability_score"]
                if model_usability and "usability_score" in summary_df.columns
                else []
            )
        )
    ]
    left_cols = [
        str(c)
        for c in (
            ["column", "opt_dtype", "quality_flag"]
            + [c for c in string_stat_cols if c in text_string_cols]
            + (["usability_flags", "recommendation"] if model_usability else [])
            + test_cols
        )
    ]

    gt_table = (
        gt_table.cols_align(align="center", columns=center_cols)
        .cols_align(align="left", columns=left_cols)
        .tab_options(
            table_font_size="12px",
            heading_background_color="#f8f9fa",
            column_labels_background_color="#e9ecef",
        )
    )

    gt_table = _apply_correlation_columns(
        gt_table, corr_target, summary_df, has_corr_data
    )
    gt_table = _apply_nanoplots(gt_table, has_hist_data)

    return gt_table.sub_missing(missing_text=_MISSING_TEXT)
