"""
Cleaning and optimization utilities for Polars DataFrames.

Public functions:
    fix()                - one-call clean & optimize pipeline
    clean_column_names() - column name normalization (case styles)
    convert_datatypes()  - dtype shrinking for memory savings
    drop_missing()       - drop sparse rows/columns
"""

from __future__ import annotations

import math
import re
import unicodedata

import polars as pl

from ._helpers import literal_name_map, require_dataframe

# Case styles accepted by clean_column_names / fix
_CASE_STYLES = ("snake", "camel", "pascal", "kebab", "upper", "lower")


def _split_words(s: str) -> list[str]:
    """Tokenize a name into words: separators and camelCase boundaries split."""
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"[_\s\-]+", " ", s)
    # fooBar -> foo Bar ; HTTPServer -> HTTP Server
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return [w for w in s.split() if w]


def _normalize(s: str, case: str = "lower", ascii_only: bool = True) -> str:
    s = s.strip()
    if ascii_only:
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")

    if case in ("lower", "upper"):
        # Legacy simple behavior: separators -> "_", no camelCase splitting
        s = re.sub(r"[^\w\s-]+", "", s)
        s = re.sub(r"[\s\-]+", "_", s)
        return s.lower() if case == "lower" else s.upper()

    words = _split_words(s)
    if not words:
        return ""
    if case == "snake":
        return "_".join(w.lower() for w in words)
    if case == "kebab":
        return "-".join(w.lower() for w in words)
    if case == "camel":
        return words[0].lower() + "".join(w.capitalize() for w in words[1:])
    if case == "pascal":
        return "".join(w.capitalize() for w in words)
    raise ValueError(f"case must be one of {_CASE_STYLES}, got {case!r}")


def clean_column_names(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    case: str = "lower",
    ascii_only: bool = True,
    dedupe: bool = True,
) -> pl.DataFrame:
    """
    Normalize column names.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        The input frame; a LazyFrame stays lazy.
    case : str, default "lower"
        Naming style for the cleaned columns:
        - "snake":  my_column_name (splits camelCase words)
        - "camel":  myColumnName
        - "pascal": MyColumnName
        - "kebab":  my-column-name
        - "lower":  lowercase, separators -> "_" (no camelCase splitting)
        - "upper":  UPPERCASE, separators -> "_" (no camelCase splitting)
    ascii_only : bool, default True
        Fold accented characters to ASCII (e.g. "Årlig" -> "Arlig").
    dedupe : bool, default True
        Suffix colliding names with _1, _2, ... so the result is unique.
    """
    if case not in _CASE_STYLES:
        raise ValueError(f"case must be one of {_CASE_STYLES}, got {case!r}")
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            f"clean_column_names() expects a Polars DataFrame or LazyFrame, got {type(df).__name__}"
        )

    old = df.collect_schema().names()
    seen: dict[str, int] = {}
    used: set[str] = set()
    new: list[str] = []
    for name in old:
        base = _normalize(str(name), case=case, ascii_only=ascii_only)
        if not base:
            base = "col"
        if dedupe:
            # Find the next free name, skipping any candidate that would collide
            # with an already-assigned name (including pre-existing real columns
            # such as a literal "base_1" elsewhere in the frame).
            candidate = base
            counter = seen.get(base, 0)
            while candidate in used:
                counter += 1
                candidate = f"{base}_{counter}"
            seen[base] = counter
            used.add(candidate)
            new_name = candidate
        else:
            new_name = base
        new.append(new_name)
    mapping = dict(zip(old, new))
    return df.rename(mapping)


# Signed integer targets in size order: (bits, dtype, min, max).
_SIGNED_INTS = (
    (8, pl.Int8, -(2**7), 2**7 - 1),
    (16, pl.Int16, -(2**15), 2**15 - 1),
    (32, pl.Int32, -(2**31), 2**31 - 1),
    (64, pl.Int64, -(2**63), 2**63 - 1),
)
_MIN_INT_BITS = (8, 16, 32, 64)


def _int_bits(dtype: pl.DataType) -> int:
    """Width in bits of an integer dtype."""
    return int(re.sub(r"\D", "", str(dtype.base_type())))


def _shrink_fact_exprs(name: str, dtype: pl.DataType) -> dict[str, pl.Expr]:
    """Aggregates _shrink_target() needs for one column (see there)."""
    column = pl.col(name)
    if dtype == pl.String:
        return {"n_unique": column.drop_nulls().n_unique()}
    if dtype.is_integer():
        return {"vmin": column.min(), "vmax": column.max()}
    if dtype == pl.Float64:
        # NaN and inf survive the round trip (Polars compares NaN equal to NaN).
        roundtrip = column.cast(pl.Float32).cast(pl.Float64)
        return {
            "n_finite": column.is_finite().sum(),
            "f32_exact": (roundtrip == column).all(),
        }
    return {}


def _shrink_target(
    dtype: pl.DataType,
    *,
    n_rows: int,
    n_unique: int | None = None,
    vmin: int | None = None,
    vmax: int | None = None,
    n_finite: int | None = None,
    f32_exact: bool | None = None,
    max_cardinality: int = 20,
    categorical_threshold: float = 0.5,
    str_to_cat: bool = True,
    downcast_ints: bool = True,
    downcast_floats: bool = True,
    min_int_bits: int = 32,
) -> pl.DataType | None:
    """Smaller dtype convert_datatypes() casts a column to, or None to keep it.

    The single source of the shrink policy: xray()'s opt_dtype calls it too,
    so the suggestion always matches what fix() does.
    """
    if dtype == pl.String:
        if (
            str_to_cat
            and n_rows > 0
            and n_unique is not None
            and n_unique <= max_cardinality
            and n_unique / n_rows <= categorical_threshold
        ):
            return pl.Categorical
        return None

    if dtype.is_integer():
        if not downcast_ints or vmin is None or vmax is None:
            return None  # disabled, or an all-null column
        for bits, target, lo, hi in _SIGNED_INTS:
            if bits >= min_int_bits and lo <= vmin and vmax <= hi:
                break
        else:
            return None  # e.g. UInt64 values above the Int64 maximum
        current = _int_bits(dtype)
        if dtype.is_signed_integer():
            return target if bits < current else None
        # Unsigned columns become signed (unsigned subtraction wraps:
        # 0 - 1 -> 255), but only when that does not widen them.
        return target if bits <= current else None

    if dtype == pl.Float64:
        if downcast_floats and n_finite and f32_exact:
            return pl.Float32
        return None

    return None


def convert_datatypes(
    df: pl.DataFrame,
    *,
    max_cardinality: int = 20,
    categorical_threshold: float = 0.5,
    str_to_cat: bool = True,
    downcast_ints: bool = True,
    downcast_floats: bool = True,
    min_int_bits: int = 32,
) -> pl.DataFrame:
    """
    Optimize DataFrame data types to reduce memory usage, without changing values.

    Columns containing nulls are optimized too - Polars casts preserve nulls.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to optimize.
    max_cardinality : int, default 20
        Maximum unique values for converting strings to categorical.
    categorical_threshold : float, default 0.5
        Threshold for categorical conversion (unique/total ratio, nulls
        included in the total).
    str_to_cat : bool, default True
        Convert eligible string columns to categorical. On Polars < 1.32,
        Categorical columns created by separate calls do not share a string
        cache, so comparing or joining them raises StringCacheMismatchError
        unless both were created inside ``with pl.StringCache():``.
    downcast_ints : bool, default True
        Downcast integer columns to the smallest signed type that holds every
        value, but no smaller than ``min_int_bits``.
    downcast_floats : bool, default True
        Downcast Float64 columns to Float32 when every value survives the
        round trip exactly, so no value changes. Most measured data does not
        qualify (0.1 has no exact Float32 form); values already stored at
        Float32 precision do.
    min_int_bits : {8, 16, 32, 64}, default 32
        Smallest integer width to downcast to. Polars integer arithmetic
        wraps on overflow for every integer type, signed or not (Int8:
        100 + 30 -> -126), so narrower types overflow sooner in later
        calculations. Lower it only when you know the column's later uses.
        Unsigned columns are converted to signed (unsigned subtraction wraps:
        0 - 1 -> 255) only when that does not make them wider; UInt64 values
        above the Int64 maximum are left untouched.

    Returns
    -------
    pl.DataFrame
        DataFrame with optimized data types.
    """
    require_dataframe(df, "convert_datatypes")
    if min_int_bits not in _MIN_INT_BITS:
        raise ValueError(f"min_int_bits must be one of {_MIN_INT_BITS}, got {min_int_bits!r}")

    # Column names pl.col() would read as patterns get stand-ins for the
    # duration of the work.
    names = literal_name_map(df.columns)
    work = df.rename(names) if names else df
    schema = work.schema

    # One pass over the frame for every column's facts.
    exprs = []
    for i, (name, dtype) in enumerate(schema.items()):
        exprs += [
            expr.alias(f"{i}\x1f{fact}")
            for fact, expr in _shrink_fact_exprs(name, dtype).items()
        ]
    row = work.select(exprs).row(0, named=True) if exprs else {}
    facts: dict[int, dict] = {}
    for key, value in row.items():
        index, fact = key.split("\x1f", 1)
        facts.setdefault(int(index), {})[fact] = value

    casts = []
    for i, (name, dtype) in enumerate(schema.items()):
        target = _shrink_target(
            dtype,
            n_rows=work.height,
            **facts.get(i, {}),
            max_cardinality=max_cardinality,
            categorical_threshold=categorical_threshold,
            str_to_cat=str_to_cat,
            downcast_ints=downcast_ints,
            downcast_floats=downcast_floats,
            min_int_bits=min_int_bits,
        )
        if target is not None and target != dtype:
            casts.append(pl.col(name).cast(target))

    result = work.with_columns(casts) if casts else work.clone()
    return result.rename({v: k for k, v in names.items()}) if names else result


def drop_missing(
    df: pl.DataFrame,
    *,
    axis: str = "rows",
    thresh: float | None = None,
    subset: list[str] | None = None,
) -> pl.DataFrame:
    """
    Drop rows or columns with missing (null) values.

    Only nulls count as missing. NaN is an ordinary float value in Polars;
    call ``df.fill_nan(None)`` first to treat NaN as missing too.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame.
    axis : str, default "rows"
        Whether to drop "rows" or "columns" with missing values.
    thresh : float | None, optional
        Threshold for dropping. If float, interpreted as percentage of
        non-null values required (0.0 to 1.0). If None, drop any with nulls.
    subset : list[str] | None, optional
        Specific columns to consider for row dropping. Only supported when
        ``axis="rows"`` because Polars DataFrames do not have row labels.

    Returns
    -------
    pl.DataFrame
        DataFrame with missing values dropped.
    """
    require_dataframe(df, "drop_missing")
    if thresh is not None and not 0.0 <= thresh <= 1.0:
        raise ValueError("thresh must be between 0 and 1")

    if axis not in {"rows", "columns"}:
        raise ValueError("axis must be 'rows' or 'columns'")

    if axis == "columns" and subset is not None:
        raise ValueError("subset is only supported when axis='rows'")

    if subset:
        missing_subset = [c for c in subset if c not in df.columns]
        if missing_subset:
            raise ValueError(f"subset contains unknown columns: {missing_subset}")

    if axis == "rows":
        if not df.columns:
            return df.clone()
        names = literal_name_map(df.columns)
        work = df.rename(names) if names else df
        cols = [names.get(c, c) for c in (subset or df.columns)]
        if thresh is None:
            kept = work.drop_nulls(cols)
        else:
            required_count = math.ceil(thresh * len(cols))
            kept = work.filter(
                pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in cols])
                >= required_count
            )
        return kept.rename({v: k for k, v in names.items()}) if names else kept

    total_rows = df.height
    if total_rows == 0:
        return df.clone()
    null_counts = df.null_count().row(0)
    dropped = [
        col
        for col, n_null in zip(df.columns, null_counts)
        if not (n_null == 0 if thresh is None else (total_rows - n_null) / total_rows >= thresh)
    ]
    return _drop_columns(df, dropped)


def _drop_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Drop columns by literal name (also '*' and '^...$' names)."""
    if not columns:
        return df
    drop = set(columns)
    return pl.DataFrame([s for s in df.get_columns() if s.name not in drop])


def _fmt_size(n_bytes: float) -> str:
    """Format a byte count with a sensible unit."""
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / 1024 ** 3:.1f} GB"
    if n_bytes >= 1024 ** 2:
        return f"{n_bytes / 1024 ** 2:.1f} MB"
    return f"{n_bytes / 1024:.1f} KB"


def _n_unique_expr(name: str, dtype: pl.DataType) -> pl.Expr:
    """Distinct non-null values; Array is counted as List (Polars 1.25 cannot group Array)."""
    column = pl.col(name)
    if isinstance(dtype, pl.Array):
        column = column.cast(pl.List(dtype.inner))
    return column.drop_nulls().n_unique()


def _hashable(df: pl.DataFrame) -> pl.DataFrame:
    """Cast Array columns to List so the frame can be grouped on Polars 1.25."""
    arrays = [
        pl.col(c).cast(pl.List(dt.inner)) for c, dt in df.schema.items() if isinstance(dt, pl.Array)
    ]
    return df.with_columns(arrays) if arrays else df


def fix(
    df: pl.DataFrame,
    *,
    case: str | None = "snake",
    strip_strings: bool = True,
    shrink_dtypes: bool = True,
    drop_empty_columns: bool = True,
    drop_duplicate_rows: bool = False,
    missing_threshold: float | None = None,
    drop_constant_columns: bool = False,
    outliers: str | None = None,
    outlier_threshold: float | None = None,
    min_int_bits: int = 32,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Clean and optimize a DataFrame in one call.

    The default steps change no values: names, whitespace, empty columns and
    dtypes that hold every value exactly. Anything that removes or alters data
    (duplicate rows, sparse/constant columns, outliers) is opt-in.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame. Never mutated; a cleaned copy is returned.
    case : str | None, default "snake"
        Column naming style: "snake", "camel", "pascal", "kebab", "upper",
        "lower", or None to leave names untouched.
        Example: "Kunde Navn" -> kunde_navn / kundeNavn / KundeNavn /
        kunde-navn / KUNDE_NAVN / kunde_navn.
    strip_strings : bool, default True
        Trim whitespace in string columns; empty/whitespace-only -> null.
    shrink_dtypes : bool, default True
        Shrink dtypes via convert_datatypes: integers to the smallest signed
        type of at least ``min_int_bits``, Float64 to Float32 only when every
        value survives exactly, and low-cardinality strings to Categorical.
        See convert_datatypes for the rules, including the string-cache
        caveat for Categorical on Polars < 1.32.
    drop_empty_columns : bool, default True
        Drop columns where every value is null.
    drop_duplicate_rows : bool, default False
        Drop rows that are exact duplicates across all columns.
    missing_threshold : float | None, optional
        Drop columns whose null share exceeds this (e.g. 0.9 = drop >90%
        missing). None disables. NaN is not counted as missing.
    drop_constant_columns : bool, default False
        Drop columns with a single distinct non-null value.
    outliers : str | None, optional
        "iqr" or "zscore": replace extreme finite numeric values with null.
        The fences come from the column's finite values (linear-interpolated
        quartiles for "iqr"); NaN and +/-inf are never replaced. None
        (default) leaves data untouched.
    outlier_threshold : float | None, optional
        Fence multiplier: defaults to 1.5 for "iqr", 3.0 for "zscore".
    min_int_bits : {8, 16, 32, 64}, default 32
        Smallest integer width shrink_dtypes may produce. Polars integer
        arithmetic wraps on overflow for every integer type, so narrower types
        overflow sooner in later calculations (Int8: 100 + 30 -> -126).
    verbose : bool, default True
        Print a compact report of what changed.

    Returns
    -------
    pl.DataFrame
        The cleaned, optimized DataFrame.

    Examples
    --------
    >>> import polarscope as ps
    >>> df = ps.fix(df)                          # safe defaults
    >>> df = ps.fix(df, case="camel")            # camelCase headers
    >>> df = ps.fix(df, drop_duplicate_rows=True, missing_threshold=0.9)
    """
    require_dataframe(df, "fix")
    if outliers is not None and outliers not in ("iqr", "zscore"):
        raise ValueError("outliers must be None, 'iqr', or 'zscore'")
    if missing_threshold is not None and not 0.0 <= missing_threshold <= 1.0:
        raise ValueError("missing_threshold must be between 0 and 1")
    if (
        outlier_threshold is not None
        and (
            not math.isfinite(outlier_threshold)
            or outlier_threshold <= 0
        )
    ):
        raise ValueError("outlier_threshold must be a positive finite number")
    if min_int_bits not in _MIN_INT_BITS:
        raise ValueError(f"min_int_bits must be one of {_MIN_INT_BITS}, got {min_int_bits!r}")

    result = df.clone()
    report: list[str] = []
    mem_before = df.estimated_size()
    rows_before, cols_before = df.shape

    # 1. Column names
    if case is not None:
        before_names = list(result.columns)
        result = clean_column_names(result, case=case)
        n_renamed = sum(1 for a, b in zip(before_names, result.columns) if a != b)
        if n_renamed:
            report.append(f"Renamed {n_renamed} column(s) to {case} case")

    # Column names pl.col() would read as patterns ('*', '^...$') get
    # stand-ins until the end; reports show the real names.
    names = literal_name_map(result.columns)
    shown = {v: k for k, v in names.items()}
    if names:
        result = result.rename(names)

    def label(columns: list[str]) -> str:
        return ", ".join(shown.get(c, c) for c in columns)

    # 2. String values: trim whitespace, empty -> null
    if strip_strings:
        str_cols = [c for c, dt in result.schema.items() if dt == pl.String]
        if str_cols and result.height > 0:
            stripped = {c: pl.col(c).str.strip_chars() for c in str_cols}
            counts = result.select(
                pl.sum_horizontal([(pl.col(c) != stripped[c]).sum() for c in str_cols]).alias("trimmed"),
                pl.sum_horizontal([(stripped[c].str.len_chars() == 0).sum() for c in str_cols]).alias("emptied"),
            ).row(0)
            n_trimmed, n_emptied = (int(n or 0) for n in counts)
            if n_trimmed or n_emptied:
                result = result.with_columns([
                    pl.when(stripped[c].str.len_chars() == 0)
                    .then(None)
                    .otherwise(stripped[c])
                    .alias(c)
                    for c in str_cols
                ])
                report.append(
                    f"Stripped whitespace in {len(str_cols)} string column(s); "
                    f"{n_trimmed} value(s) trimmed, {n_emptied} empty string(s) -> null"
                )

    # 3. Empty columns (100% null)
    if drop_empty_columns and result.height > 0:
        null_counts = dict(zip(result.columns, result.null_count().row(0)))
        empty = [c for c, n in null_counts.items() if n == result.height]
        if empty:
            result = _drop_columns(result, empty)
            report.append(f"Dropped {len(empty)} empty column(s): {label(empty)}")

    # 4. Sparse columns above the missing threshold
    if missing_threshold is not None and result.height > 0:
        null_counts = dict(zip(result.columns, result.null_count().row(0)))
        sparse = [c for c, n in null_counts.items() if n / result.height > missing_threshold]
        if sparse:
            result = _drop_columns(result, sparse)
            report.append(
                f"Dropped {len(sparse)} column(s) >{missing_threshold:.0%} missing: {label(sparse)}"
            )

    # 5. Constant columns (single distinct non-null value)
    if drop_constant_columns and result.height > 0 and result.width > 0:
        try:
            counts = result.select(
                [_n_unique_expr(c, dt).alias(c) for c, dt in result.schema.items()]
            ).row(0, named=True)
        except (pl.exceptions.PolarsError, pl.exceptions.PanicException):
            # Some dtype (e.g. Object) cannot be counted: count column by
            # column and keep the ones that fail.
            counts = {}
            for c, dt in result.schema.items():
                try:
                    counts[c] = result.select(_n_unique_expr(c, dt)).item()
                except (pl.exceptions.PolarsError, pl.exceptions.PanicException):
                    counts[c] = None
        constant = [c for c, n in counts.items() if n == 1]
        if constant:
            result = _drop_columns(result, constant)
            report.append(f"Dropped {len(constant)} constant column(s): {label(constant)}")

    # 6. Duplicate rows
    if drop_duplicate_rows and result.width > 0:
        before = result.height
        first = _hashable(result).select(pl.struct(pl.all()).is_first_distinct()).to_series()
        result = result.filter(first)
        removed = before - result.height
        if removed:
            report.append(f"Removed {removed} duplicate row(s)")

    # 7. Outliers -> null (opt-in)
    if outliers is not None and result.height > 0:
        fence = outlier_threshold if outlier_threshold is not None else (1.5 if outliers == "iqr" else 3.0)
        numeric_cols = [c for c, dt in result.schema.items() if dt.is_numeric()]

        def finite(c: str) -> pl.Expr:
            values = pl.col(c).cast(pl.Float64)
            return pl.when(values.is_finite()).then(values)

        # Fences from finite values only: NaN or inf in a column must not
        # poison its mean, std or quartiles (Polars orders NaN above every
        # number, so a NaN fence flags everything).
        if outliers == "iqr":
            stats = [
                e
                for c in numeric_cols
                for e in (
                    finite(c).quantile(0.25, "linear").alias(f"{c}\x1flo"),
                    finite(c).quantile(0.75, "linear").alias(f"{c}\x1fhi"),
                )
            ]
        else:
            stats = [
                e
                for c in numeric_cols
                for e in (finite(c).mean().alias(f"{c}\x1flo"), finite(c).std().alias(f"{c}\x1fhi"))
            ]
        row = result.select(stats).row(0, named=True) if stats else {}
        masks = {}
        for c in numeric_cols:
            a, b = row[f"{c}\x1flo"], row[f"{c}\x1fhi"]
            if a is None or b is None:
                continue  # no finite values
            if outliers == "iqr":
                lo, hi = a - fence * (b - a), b + fence * (b - a)
            else:
                if b == 0:
                    continue
                lo, hi = a - fence * b, a + fence * b
            masks[c] = finite(c).is_not_null() & ((finite(c) < lo) | (finite(c) > hi))
        if masks:
            n_out = result.select([m.sum().alias(c) for c, m in masks.items()]).row(0, named=True)
            exprs = [
                pl.when(masks[c]).then(None).otherwise(pl.col(c)).alias(c)
                for c, n in n_out.items()
                if n
            ]
            n_nulled = sum(int(n or 0) for n in n_out.values())
            if exprs:
                result = result.with_columns(exprs)
                report.append(f"Nulled {n_nulled} outlier value(s) ({outliers}, fence={fence})")

    # 8. Dtype shrinking
    if shrink_dtypes:
        schema_before = dict(result.schema)
        result = convert_datatypes(result, min_int_bits=min_int_bits)
        changed = [
            f"{shown.get(c, c)}: {schema_before[c]} -> {dt}"
            for c, dt in result.schema.items()
            if schema_before.get(c) != dt
        ]
        if changed:
            report.append(f"Shrunk dtypes in {len(changed)} column(s): " + "; ".join(changed))

    if names:
        result = result.rename({v: k for k, v in names.items() if v in result.columns})

    # Report
    if verbose:
        mem_after = result.estimated_size()
        print("ps.fix report")
        print("─" * 13)
        if report:
            for line in report:
                print(line)
        else:
            print("Nothing to fix - data already clean")
        pct = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
        print(f"Memory: {_fmt_size(mem_before)} -> {_fmt_size(mem_after)} ({pct:.0f}% saved)" if pct > 0.5
              else f"Memory: {_fmt_size(mem_before)} -> {_fmt_size(mem_after)}")
        rows_after, cols_after = result.shape
        row_note = f"{rows_before:,} -> {rows_after:,}" if rows_after != rows_before else f"{rows_before:,} (unchanged)"
        col_note = f"{cols_before} -> {cols_after}" if cols_after != cols_before else f"{cols_before} (unchanged)"
        print(f"Rows: {row_note} | Columns: {col_note}")

    return result
