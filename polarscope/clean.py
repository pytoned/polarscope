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
    df: pl.DataFrame,
    *,
    case: str = "lower",
    ascii_only: bool = True,
    dedupe: bool = True,
) -> pl.DataFrame:
    """
    Normalize column names.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame.
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

    old = list(df.columns)
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


def convert_datatypes(
    df: pl.DataFrame,
    *,
    max_cardinality: int = 20,
    categorical_threshold: float = 0.5,
    str_to_cat: bool = True,
    downcast_ints: bool = True,
    downcast_floats: bool = True,
) -> pl.DataFrame:
    """
    Optimize DataFrame data types to reduce memory usage.

    Columns containing nulls are optimized too - Polars casts preserve nulls.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to optimize.
    max_cardinality : int, default 20
        Maximum unique values for converting strings to categorical.
    categorical_threshold : float, default 0.5
        Threshold for categorical conversion (unique/total ratio).
    str_to_cat : bool, default True
        Convert eligible string columns to categorical.
    downcast_ints : bool, default True
        Downcast integer columns to smallest possible type.
    downcast_floats : bool, default True
        Downcast float columns to smallest possible type.

    Returns
    -------
    pl.DataFrame
        DataFrame with optimized data types.
    """
    result = df.clone()

    transformations: list[pl.Expr] = []
    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        if str_to_cat and dtype in (pl.String, pl.Utf8):
            unique_count = series.drop_nulls().n_unique()
            total_count = len(series)
            if (
                total_count > 0
                and unique_count <= max_cardinality
                and unique_count / total_count <= categorical_threshold
            ):
                transformations.append(pl.col(col).cast(pl.Categorical).alias(col))
            continue

        if downcast_ints and dtype.is_integer():
            min_val = series.min()
            max_val = series.max()
            if min_val is None or max_val is None:
                continue  # all-null column

            if min_val >= 0:
                if max_val <= 255:
                    transformations.append(pl.col(col).cast(pl.UInt8).alias(col))
                elif max_val <= 65535:
                    transformations.append(pl.col(col).cast(pl.UInt16).alias(col))
                elif max_val <= 4294967295:
                    transformations.append(pl.col(col).cast(pl.UInt32).alias(col))
            else:
                if min_val >= -128 and max_val <= 127:
                    transformations.append(pl.col(col).cast(pl.Int8).alias(col))
                elif min_val >= -32768 and max_val <= 32767:
                    transformations.append(pl.col(col).cast(pl.Int16).alias(col))
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    transformations.append(pl.col(col).cast(pl.Int32).alias(col))
            continue

        if downcast_floats and dtype.is_float():
            try:
                finite_mask = (~series.is_nan()) & (~series.is_infinite())
                finite_values = series.filter(finite_mask)
                if finite_values.len() == 0:
                    continue

                series_f32 = series.cast(pl.Float32)
                roundtrip = series_f32.cast(pl.Float64)
                max_abs_diff = (series.filter(finite_mask) - roundtrip.filter(finite_mask)).abs().max()
                if max_abs_diff is not None and float(max_abs_diff) <= 1e-6:
                    transformations.append(pl.col(col).cast(pl.Float32).alias(col))
            except Exception:
                pass  # Keep original type if conversion fails.

    if transformations:
        result = result.with_columns(transformations)

    return result


def drop_missing(
    df: pl.DataFrame,
    *,
    axis: str = "rows",
    thresh: float | None = None,
    subset: list[str] | None = None,
) -> pl.DataFrame:
    """
    Drop rows or columns with missing values.

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
        if subset:
            if thresh is None:
                return df.filter(~pl.any_horizontal([pl.col(c).is_null() for c in subset]))
            else:
                required_count = math.ceil(thresh * len(subset))
                return df.filter(
                    pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in subset]) >= required_count
                )
        else:
            if thresh is None:
                return df.filter(~pl.any_horizontal([pl.col(c).is_null() for c in df.columns]))
            else:
                required_count = math.ceil(thresh * len(df.columns))
                return df.filter(
                    pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in df.columns]) >= required_count
                )

    else:
        cols_to_keep = []
        total_rows = len(df)

        if total_rows == 0:
            return df.clone()

        for col in df.columns:
            non_null_count = df[col].drop_nulls().len()

            if thresh is None:
                if non_null_count == total_rows:
                    cols_to_keep.append(col)
            else:
                if non_null_count / total_rows >= thresh:
                    cols_to_keep.append(col)

        return df.select(cols_to_keep) if cols_to_keep else df.select([])

def _fmt_size(n_bytes: float) -> str:
    """Format a byte count with a sensible unit."""
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / 1024 ** 3:.1f} GB"
    if n_bytes >= 1024 ** 2:
        return f"{n_bytes / 1024 ** 2:.1f} MB"
    return f"{n_bytes / 1024:.1f} KB"


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
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Clean and optimize a DataFrame in one call.

    Lossless steps are on by default; anything that removes or alters data
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
        Downcast ints/floats and convert low-cardinality strings to
        Categorical (via convert_datatypes; handles nulls).
    drop_empty_columns : bool, default True
        Drop columns where every value is null.
    drop_duplicate_rows : bool, default False
        Drop rows that are exact duplicates across all columns.
    missing_threshold : float | None, optional
        Drop columns whose null share exceeds this (e.g. 0.9 = drop >90%
        missing). None disables.
    drop_constant_columns : bool, default False
        Drop columns with a single distinct non-null value.
    outliers : str | None, optional
        "iqr" or "zscore": replace extreme numeric values with null.
        None (default) leaves data untouched.
    outlier_threshold : float | None, optional
        Fence multiplier: defaults to 1.5 for "iqr", 3.0 for "zscore".
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

    # 2. String values: trim whitespace, empty -> null
    if strip_strings:
        str_cols = [c for c, dt in result.schema.items() if dt in (pl.String, pl.Utf8)]
        if str_cols and result.height > 0:
            n_trimmed = int(result.select(
                pl.sum_horizontal([
                    (pl.col(c) != pl.col(c).str.strip_chars()).sum() for c in str_cols
                ])
            ).item() or 0)
            n_emptied = int(result.select(
                pl.sum_horizontal([
                    (pl.col(c).str.strip_chars().str.len_chars() == 0).sum() for c in str_cols
                ])
            ).item() or 0)
            result = result.with_columns([
                pl.when(pl.col(c).str.strip_chars().str.len_chars() == 0)
                .then(None)
                .otherwise(pl.col(c).str.strip_chars())
                .alias(c)
                for c in str_cols
            ])
            if n_trimmed or n_emptied:
                report.append(
                    f"Stripped whitespace in {len(str_cols)} string column(s); "
                    f"{n_trimmed} value(s) trimmed, {n_emptied} empty string(s) -> null"
                )

    # 3. Empty columns (100% null)
    if drop_empty_columns and result.height > 0:
        empty = [c for c in result.columns if result[c].null_count() == result.height]
        if empty:
            result = result.drop(empty)
            report.append(f"Dropped {len(empty)} empty column(s): {', '.join(empty)}")

    # 4. Sparse columns above the missing threshold
    if missing_threshold is not None and result.height > 0:
        sparse = [
            c for c in result.columns
            if result[c].null_count() / result.height > missing_threshold
        ]
        if sparse:
            result = result.drop(sparse)
            report.append(
                f"Dropped {len(sparse)} column(s) >{missing_threshold:.0%} missing: {', '.join(sparse)}"
            )

    # 5. Constant columns (single distinct non-null value)
    if drop_constant_columns and result.height > 0:
        constant = [
            c for c in result.columns
            if result[c].drop_nulls().n_unique() == 1
        ]
        if constant:
            result = result.drop(constant)
            report.append(f"Dropped {len(constant)} constant column(s): {', '.join(constant)}")

    # 6. Duplicate rows
    if drop_duplicate_rows:
        before = result.height
        result = result.unique(maintain_order=True)
        removed = before - result.height
        if removed:
            report.append(f"Removed {removed} duplicate row(s)")

    # 7. Outliers -> null (opt-in)
    if outliers is not None and result.height > 0:
        fence = outlier_threshold if outlier_threshold is not None else (1.5 if outliers == "iqr" else 3.0)
        numeric_cols = [c for c, dt in result.schema.items() if dt.is_numeric()]
        exprs = []
        n_nulled = 0
        for c in numeric_cols:
            series = result[c]
            if outliers == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                if q1 is None or q3 is None:
                    continue
                iqr = q3 - q1
                lo, hi = q1 - fence * iqr, q3 + fence * iqr
                mask = (pl.col(c) < lo) | (pl.col(c) > hi)
            else:  # zscore
                mean, std = series.mean(), series.std()
                if mean is None or std is None or std == 0:
                    continue
                mask = ((pl.col(c) - mean) / std).abs() > fence
            n_out = int(result.select(mask.sum()).item() or 0)
            if n_out:
                n_nulled += n_out
                exprs.append(pl.when(mask).then(None).otherwise(pl.col(c)).alias(c))
        if exprs:
            result = result.with_columns(exprs)
            report.append(f"Nulled {n_nulled} outlier value(s) ({outliers}, fence={fence})")

    # 8. Dtype shrinking
    if shrink_dtypes:
        schema_before = dict(result.schema)
        result = convert_datatypes(result)
        changed = [
            f"{c}: {schema_before[c]} -> {dt}"
            for c, dt in result.schema.items()
            if schema_before.get(c) != dt
        ]
        if changed:
            report.append(f"Shrunk dtypes in {len(changed)} column(s): " + "; ".join(changed))

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
