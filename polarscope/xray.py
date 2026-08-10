from __future__ import annotations

import inspect
import math
import re
import time
from collections.abc import Iterable
from numbers import Real
from typing import Any, Union

import polars as pl
from great_tables import GT

# Optional scipy imports - lazy loaded to avoid import warnings
SCIPY_AVAILABLE = None  # Will be checked when needed
stats = None
_NORMALITY_MONTE_CARLO_SAMPLES = 199
_NORMALITY_MAX_SAMPLE_SIZE = 3000


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
    """Format memory usage with appropriate units using Polars unit parameter."""
    try:
        size_mb = df.estimated_size(unit='mb')
        if size_mb >= 1000:  # >= 1 GB
            return f"{size_mb / 1024:.1f} GB"
        elif size_mb >= 1.0:  # >= 1 MB
            return f"{size_mb:.1f} MB"
        else:  # < 1 MB, use KB
            return f"{size_mb * 1024:.1f} KB"
    except Exception:
        # Fallback for older Polars versions
        try:
            memory_bytes = df.estimated_size()
            memory_mb = memory_bytes / 1024 / 1024
            if memory_mb < 1.0:
                memory_kb = memory_mb * 1024
                return f"{memory_kb:.1f} KB"
            else:
                return f"{memory_mb:.1f} MB"
        except Exception:
            return "Unknown"


def _is_stringy(dt) -> bool:
    """True for dtypes analyzed with string statistics (String/Categorical/Enum)."""
    return dt == pl.String or isinstance(dt, (pl.Categorical, pl.Enum))


def _get_columns_to_analyze(df: pl.DataFrame, include: str | list[str] | None) -> list[str]:
    """
    Filter columns based on include parameter.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to analyze
    include : str, list[str], or None
        Which data types to include

    Returns
    -------
    list[str]
        List of column names to analyze
    """
    if include is None or include == 'numeric':
        # Default behavior: only numeric columns
        return [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]

    elif include == 'all':
        # Include all columns
        return df.columns

    elif include == 'string':
        # String-like columns (String, Categorical, Enum)
        return [c for c, dt in zip(df.columns, df.dtypes) if _is_stringy(dt)]
    
    elif include == 'temporal':
        # Only date/datetime columns
        return [c for c, dt in zip(df.columns, df.dtypes) if dt.is_temporal()]
    
    elif isinstance(include, list):
        # Match exact dtypes and parameterized dtype families such as Datetime.
        include_types = {str(dtype) for dtype in include}
        return [
            c
            for c, dt in zip(df.columns, df.dtypes)
            if str(dt) in include_types or str(dt.base_type()) in include_types
        ]
    
    else:
        raise ValueError(f"Invalid include parameter: {include}. "
                        f"Must be None, 'all', 'numeric', 'string', 'temporal', or list of dtype names.")


def xray(
    df: pl.DataFrame,
    *,
    include: str | list[str] | None = None,
    great_tables: bool = True,
    expanded: bool = False,
    title: str | None = None,
    percentiles: list[float] | None = None,
    outlier_method: str = "iqr",
    outlier_bounds: list[float] | None = None,
    corr_target: str | None = None,
    normality_test: str = "shapiro",
    uniformity_test: str = "ks",
    missing_threshold: float = 0.3,
    constant_threshold: float = 0.99,
    skew_threshold: float = 2.0,
    kurtosis_threshold: float = 7.0,
    outlier_threshold: float = 0.05,
    shakiness_threshold: int = 2,
    model_usability: bool = False,
    distribution_plot: str = "histogram",
    decimals: int = 2,
    sep_mark: str = ",",
    dec_mark: str = ".",
    compact: bool = False,
    pattern: str | None = None
) -> Union[GT, pl.DataFrame]:
    """
    X-ray your data: comprehensive statistical analysis with quality assessment.
    
    This function provides deep insight into DataFrame structure and quality,
    revealing hidden issues, statistical properties, and data health indicators.
    Perfect for exploratory data analysis and data quality assessment.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to summarize.
    include : str, list[str], or None, default None
        Which data types to include in the analysis.
        - None (default): Only numeric columns (Int8, Int16, Int32, Int64, Float32, Float64)
        - 'all': All columns regardless of data type
        - 'numeric': Only numeric columns (same as None)
        - 'string': String-like columns (String, Categorical, Enum)
        - 'temporal': Only date/datetime columns
        - list of strings: Specific data type names (e.g., ['Float64', 'String'])
        Boolean columns are analyzed as 0/1 numerics (Mean = share of True);
        temporal columns report their earliest/latest timestamps.
        Raises ValueError when no columns match the selection.
    great_tables : bool, default True
        Whether to return a formatted Great Tables object (True) or standard
        Polars DataFrame output (False).
    expanded : bool, default False
        If True, shows all available statistics. If False, shows only essential
        metrics: dtype, count, null_count, mean, std, min, 25%, 50%, 75%, max,
        iqr, pct_missing, n_outliers, skew.
        For string/categorical columns the essential view also includes the top
        value, its frequency, and the min/median/avg/max value length; the
        expanded view adds the mode share %, the top-3 values, and a sample of
        distinct values. Numeric-only statistics are omitted entirely when none
        of the analyzed columns are numeric (e.g. include='string').
    title : str | None, optional
        Custom title for the Great Tables output. If None, uses default titles:
        "🔬 DataFrame X-ray" (minimal) or "🔬 Expanded Statistics" (expanded).
        Only applies when great_tables=True.
    percentiles : list[float] | None, optional
        Custom percentiles to calculate. Default: [0.25, 0.5, 0.75].
        Example: [0.1, 0.25, 0.5, 0.75, 0.9] for additional quantiles.
    outlier_method : str, default "iqr"
        Method for outlier detection:
        - "iqr": Interquartile range method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        - "percentile": Use custom percentile bounds from outlier_bounds
        - "zscore": Z-score method (mean ± 3*std)
    outlier_bounds : list[float] | None, optional
        Custom percentile bounds for outlier detection when method="percentile".
        Example: [0.05, 0.95] for 5th and 95th percentiles.
    corr_target : str | None, optional
        Target column for correlation analysis. Must be a numeric column.
        Shows correlation between each numeric column and the target.
        The accompanying bars use a fixed -1 to 1 scale, so bar length is
        comparable across columns and across separate tables.
    normality_test : str, default "shapiro"
        Statistical test for normality (requires scipy):
        - "shapiro": Shapiro-Wilk test (good for small/medium samples)
        - "anderson": Anderson-Darling test (sensitive to tail behavior)
        - "ks": Kolmogorov-Smirnov test vs normal distribution
    uniformity_test : str, default "ks"
        Statistical test for uniformity (requires scipy):
        - "ks": Kolmogorov-Smirnov test vs uniform distribution
        - "chi2": Chi-square goodness of fit test
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
        Minimum score to flag column as "shaky" for parametric models.
    model_usability : bool, default False
        Include sophisticated model usability scoring with weighted flags and recommendations.
        Adds columns: usability_flags, usability_score, recommendation
        (in both minimal and expanded mode).
    distribution_plot : str, default "histogram"
        Type of distribution visualization for numeric columns:
        - "histogram": Bar-based histogram (default) without markers
    decimals : int, default 2
        Number of decimal places for numeric formatting in Great Tables output.
        Plain Polars DataFrame output retains full numeric precision.
    sep_mark : str, default ","
        Thousands separator mark for numeric formatting (e.g., "1,000").
    dec_mark : str, default "."
        Decimal mark for numeric formatting (e.g., "1.23").
    compact : bool, default False
        If True, large numbers are auto-scaled with suffixes (e.g., "10K", "1.5M").
    pattern : str | None, optional
        Text pattern for decorating formatted values (e.g., "[{x}]").

    Returns
    -------
    Union[GT, pl.DataFrame]
        Either a Great Tables object (if great_tables=True) or a Polars DataFrame
        (if great_tables=False) containing the comprehensive summary statistics.

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
    >>> table = ps.xray(df)  # Only shows price, volume, rating (numeric columns)
    >>> table.show()
    
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

    Notes
    -----
    The shakiness score combines multiple data quality indicators:
    - High missingness (> threshold)
    - Constant/quasi-constant values
    - Extreme skewness
    - High outlier percentage
    - Failed normality tests
    
    Columns with shakiness_score >= shakiness_threshold are flagged as "⚠ SHAKY"
    for potential issues with parametric statistical models.
    
    When model_usability=True, additional sophisticated quality assessment is performed:
    - Weighted flag system (HM, MM, ID, BN, CV, EO, ES, EK, NN, ZH, UC)
    - 0-100 usability score with actionable recommendations
    - Based on polarsight's proven model usability framework
    
    Statistical tests require scipy. If not available, tests are skipped
    with informative messages.
    """
    # Start timing for performance measurement
    start_time = time.perf_counter()
    
    # Memory usage will be calculated in _format_memory_usage() using unit parameter
    
    # Set default percentiles
    if percentiles is None:
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
        raise ValueError("outlier_bounds must be provided when outlier_method='percentile'")

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

    if distribution_plot not in ["histogram"]:
        raise ValueError("distribution_plot must be 'histogram'")

    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    
    # Validate correlation target
    if corr_target:
        if corr_target not in df.columns:
            raise ValueError(f"Target column '{corr_target}' not found in DataFrame")
        target_dtype = df.schema[corr_target]
        if not target_dtype.is_numeric():
            raise ValueError(f"Target column '{corr_target}' must be numeric, got {target_dtype}")
    
    # Filter columns based on include parameter
    all_cols = _get_columns_to_analyze(df, include)
    if not all_cols:
        if include is None or include == 'numeric':
            raise ValueError(
                "No numeric columns found to analyze. "
                "Pass include='all' to analyze all column types, or include='string' for string columns."
            )
        raise ValueError(f"No columns matched include={include!r} in this DataFrame.")

    # Calculate comprehensive statistics for all columns
    stats_data = []

    schema = df.schema

    # IQR needs both quartiles. When the requested percentiles omit either one
    # the statistic is never computed, so no branch may seed it with None -
    # doing so would leave an all-null IQR column in the output.
    iqr_available = 0.25 in percentiles and 0.75 in percentiles

    # Batch all target correlations in a single pass instead of one query per column
    corr_map: dict[str, float | None] = {}
    if corr_target:
        corr_exprs = []
        for c in all_cols:
            if c == corr_target:
                continue
            if schema[c].is_numeric():
                corr_exprs.append(pl.corr(corr_target, c).alias(c))
            elif schema[c] == pl.Boolean:
                corr_exprs.append(pl.corr(pl.col(corr_target), pl.col(c).cast(pl.UInt8)).alias(c))
        if corr_exprs:
            try:
                corr_row = df.select(corr_exprs)
                corr_map = {c: corr_row[c][0] for c in corr_row.columns}
            except Exception:
                corr_map = {}

    for col in all_cols:
        dtype = schema[col]
        is_bool = dtype == pl.Boolean
        # Booleans are analyzed as 0/1 numerics (Mean = share of True, etc.)
        is_numeric = dtype.is_numeric() or is_bool
        series = df[col]
        series_clean = series.drop_nulls()
        if is_bool:
            series_clean = series_clean.cast(pl.UInt8)
        n_total = len(series)
        n_valid = len(series_clean)
        n_missing = series.null_count()
        pct_missing = (n_missing / n_total * 100) if n_total > 0 else 0

        # Initialize column stats
        col_stats = {
            'column': col,
            'dtype': str(dtype),
            'count': n_valid,
            'null_count': n_missing,
            'pct_missing': pct_missing,
        }

        # Basic counts and ratios (nulls excluded) - use approx for large datasets
        is_large_dataset = n_total > 300000
        try:
            # Use approximate count for large datasets (>300k rows) for better performance
            if is_large_dataset:
                n_unique = series_clean.approx_n_unique()
            else:
                n_unique = series_clean.n_unique()
        except Exception:
            # Fallback - count via groupby (more reliable for problematic dtypes)
            n_unique = series_clean.value_counts().height

        # Use appropriate column name based on dataset size
        n_unique_col = 'n_unique(approx)' if is_large_dataset else 'n_unique'
        col_stats[n_unique_col] = n_unique
        col_stats['uniqueness_ratio'] = n_unique / n_valid if n_valid > 0 else 0.0

        # Duplicate analysis for historized datasets (among non-null values;
        # clamped because approx_n_unique may slightly overestimate)
        n_duplicates = max(0, n_valid - n_unique)
        pct_duplicates = n_duplicates / n_valid * 100 if n_valid > 0 else 0.0
        col_stats['n_duplicates'] = n_duplicates
        col_stats['pct_duplicates'] = pct_duplicates

        # Dominant-value share (set exactly by the string branch; computed for
        # small/medium columns otherwise) - feeds quasi-constant detection.
        mode_share = None

        if is_numeric and n_valid > 0:
            # Numeric-specific statistics
            
            # Basic descriptive stats
            quantile_stats = _calculate_quantiles(series_clean, percentiles)
            col_stats.update(quantile_stats)
            
            mean_val = float(series_clean.mean())
            std_val = float(series_clean.std()) if n_valid > 1 else None
            try:
                min_val = float(series_clean.min())
                max_val = float(series_clean.max())
            except (ValueError, TypeError):
                # Handle cases where min/max operations fail with mixed types
                min_val = max_val = 0.0
            
            col_stats['mean'] = mean_val
            col_stats['std'] = std_val
            col_stats['min'] = min_val
            col_stats['max'] = max_val
            
            # IQR
            if iqr_available:
                q25 = quantile_stats.get('25%')
                q75 = quantile_stats.get('75%')
                col_stats['iqr'] = (
                    q75 - q25 if q25 is not None and q75 is not None else None
                )
            
            # Zero/positive/negative counts
            n_zero = int((series_clean == 0).sum())
            n_pos = int((series_clean > 0).sum())
            n_neg = int((series_clean < 0).sum())
            
            col_stats['n_zero'] = n_zero
            col_stats['pct_zero'] = n_zero / n_valid * 100 if n_valid > 0 else 0.0
            col_stats['pct_pos'] = n_pos / n_valid * 100 if n_valid > 0 else 0.0
            col_stats['pct_neg'] = n_neg / n_valid * 100 if n_valid > 0 else 0.0
            
            # Skewness (always calculated for default view)
            if n_valid > 2:
                skew_val = float(series_clean.skew())
                col_stats['skew'] = skew_val
            else:
                col_stats['skew'] = None
            
            # Distribution plot data for nanoplots
            if n_valid > 0:
                distribution_data = []
                try:
                    if distribution_plot == "histogram":
                        # Calculate optimal number of bins (max 12 for nanoplots)
                        n_bins = min(12, max(5, math.isqrt(n_valid)))

                        if min_val == max_val:
                            # All values are the same
                            distribution_data = [n_valid]
                        else:
                            histogram = series_clean.hist(bin_count=n_bins)
                            distribution_data = histogram.get_column("count").to_list()
                    
                    
                    col_stats['distribution_plot'] = distribution_data
                except Exception:
                    # Fallback if distribution calculation fails
                    col_stats['distribution_plot'] = []
            else:
                col_stats['distribution_plot'] = []
            
            # Model usability relies on these metrics even when they are not
            # included in the minimal output table.
            if expanded or model_usability:
                # MAD (Median Absolute Deviation)
                if n_valid > 0:
                    median_val = float(series_clean.median())
                    mad = (series_clean - median_val).abs().median()
                    col_stats['mad'] = float(mad)
                
                # Kurtosis (expanded mode only) - Polars' native excess kurtosis
                # (matches scipy.stats.kurtosis) without materializing another array
                if n_valid > 2:
                    kurt_val = series_clean.kurtosis()
                    col_stats['kurtosis'] = float(kurt_val) if kurt_val is not None else None
                
                # Optimal dtype suggestion
                col_stats['opt_dtype'] = _suggest_optimal_dtype(series_clean, dtype)
                
                # Statistical tests (if scipy available)
                if _check_scipy_availability() and n_valid > 3:
                    if n_unique <= 1:
                        col_stats['normality_test'] = "N/A (constant data)"
                        col_stats['uniformity_test'] = "N/A (constant data)"
                    else:
                        normality_result = _test_normality(series_clean, normality_test)
                        col_stats['normality_test'] = normality_result

                        uniformity_result = _test_uniformity(series_clean, uniformity_test)
                        col_stats['uniformity_test'] = uniformity_result
            
            # Outlier detection (reuse already-computed quartiles for the iqr method)
            n_outliers = _count_outliers(
                series_clean, outlier_method, outlier_bounds,
                q25=quantile_stats.get('25%'), q75=quantile_stats.get('75%'),
            )
            pct_outliers = (n_outliers / n_valid * 100) if n_valid > 0 else 0
            col_stats['n_outliers'] = n_outliers
            col_stats['pct_outliers'] = pct_outliers
            
            # Correlation with target (precomputed in a single batched pass)
            if corr_target:
                if col == corr_target:
                    col_stats['correlation'] = None
                    col_stats['correlation_plot'] = None  # No plot for target column
                else:
                    corr_val = corr_map.get(col)
                    col_stats['correlation'] = corr_val
                    # Single value for horizontal bar nanoplot (fixed -1 to 1 scale)
                    col_stats['correlation_plot'] = corr_val

            # String/temporal-specific columns are N/A for numeric columns
            for stat in _STRING_ALL_COLS + _TEMPORAL_COLS:
                col_stats[stat] = None

        else:
            # ── Non-numeric columns ──────────────────────────────────────
            is_string = _is_stringy(dtype)
            is_temporal = dtype.is_temporal()

            # Temporal min/max are set by the temporal branch below
            col_stats['earliest'] = None
            col_stats['latest'] = None

            if is_string and n_valid > 0:
                # String-specific statistics (Categorical/Enum analyzed as strings)
                if dtype not in (pl.String, pl.Utf8):
                    series_clean = series_clean.cast(pl.String)
                lengths = series_clean.str.len_chars()

                # Top value (mode) and its frequency
                value_counts = series_clean.value_counts().sort('count', descending=True)
                val_col = value_counts.columns[0]
                cnt_col = value_counts.columns[1]
                vc_height = value_counts.height
                top_freq = int(value_counts[cnt_col][0])

                # Defaults: top value + length spread
                col_stats['top'] = _truncate(value_counts[val_col][0])
                col_stats['top_freq'] = top_freq
                col_stats['min_length'] = int(lengths.min())
                col_stats['median_length'] = float(lengths.median())
                col_stats['avg_length'] = float(lengths.mean())
                col_stats['max_length'] = int(lengths.max())

                # Expanded: dominant-category share, top-3, and a spread sample
                mode_share = top_freq / n_valid  # reused for quasi-constant detection
                col_stats['mode_share'] = mode_share * 100
                col_stats['top_3'] = ", ".join(
                    f"{_truncate(value_counts[val_col][i], 20)} ({int(value_counts[cnt_col][i])})"
                    for i in range(min(3, vc_height))
                )
                # Sample of distinct values spread across the frequency range
                # (most common / mid / rarest) - meaningful at any cardinality.
                sample_idx = sorted({0, vc_height // 2, vc_height - 1})[:3]
                col_stats['sample_vals'] = ", ".join(
                    _truncate(value_counts[val_col][i], 20) for i in sample_idx
                )

                # Distribution plot - top category frequencies as bar chart
                n_bars = min(12, vc_height)
                col_stats['distribution_plot'] = value_counts[cnt_col][:n_bars].to_list()
            elif is_temporal and n_valid > 0:
                # Temporal columns: earliest/latest timestamps
                col_stats['earliest'] = str(series_clean.min())
                col_stats['latest'] = str(series_clean.max())
                for stat in _STRING_ALL_COLS:
                    col_stats[stat] = None
                col_stats['distribution_plot'] = []
            else:
                for stat in _STRING_ALL_COLS:
                    col_stats[stat] = None
                col_stats['distribution_plot'] = []

            # Numeric-only statistics are N/A for non-numeric columns
            numeric_only_stats = ['mean', 'std', 'min', 'max']
            if iqr_available:
                numeric_only_stats.append('iqr')
            for stat in numeric_only_stats:
                col_stats[stat] = None
            for p in percentiles:
                label = _percentile_to_label(p)
                col_stats[label] = None
            col_stats['skew'] = None
            col_stats['n_outliers'] = None
            col_stats['pct_outliers'] = None
            col_stats['n_zero'] = None
            col_stats['pct_zero'] = None
            col_stats['pct_pos'] = None
            col_stats['pct_neg'] = None

            if expanded or model_usability:
                col_stats['mad'] = None
                col_stats['kurtosis'] = None
                col_stats['opt_dtype'] = _suggest_optimal_dtype(series_clean, dtype)
                col_stats['normality_test'] = "N/A (non-numeric)"
                col_stats['uniformity_test'] = "N/A (non-numeric)"

            # Correlation with target for non-numeric columns
            if corr_target:
                if col == corr_target:
                    col_stats['correlation'] = None
                    col_stats['correlation_plot'] = None
                else:
                    col_stats['correlation'] = None
                    col_stats['correlation_plot'] = None
        
        # Dominant-value share for quasi-constant detection. The string branch
        # sets it from its value_counts; otherwise it needs its own pass, so only
        # compute it for small/medium columns - very large columns fall back to
        # the cardinality heuristic for speed.
        if mode_share is None and n_valid > 0 and not is_large_dataset:
            try:
                mode_share = int(series_clean.value_counts().get_column("count").max()) / n_valid
            except Exception:
                mode_share = None

        # Calculate shakiness score
        shakiness_score = _calculate_shakiness_score(
            col_stats, missing_threshold, constant_threshold,
            skew_threshold, kurtosis_threshold, outlier_threshold,
            mode_share=mode_share,
        )
        col_stats['shakiness_score'] = shakiness_score
        col_stats['quality_flag'] = "⚠ SHAKY" if shakiness_score >= shakiness_threshold else "✓ OK"
        
        # Model usability evaluation (if requested)
        if model_usability:
            usability_result = _evaluate_column_usability(
                col_stats,
                has_correlation=bool(corr_target)
            )
            col_stats['usability_flags'] = usability_result['flag_string']
            col_stats['usability_score'] = usability_result['score']
            col_stats['recommendation'] = usability_result['recommendation']
        
        stats_data.append(col_stats)
    
    # Create DataFrame
    summary_df = pl.DataFrame(stats_data)

    # Correlation is recorded for every analyzed column so the summary schema
    # stays stable, but nothing correlates with the target when no analyzed
    # column is numeric besides the target itself (e.g. include='string').
    # Drop the pair rather than render two empty columns.
    if corr_target and 'correlation' in summary_df.columns:
        if summary_df['correlation'].null_count() == summary_df.height:
            summary_df = summary_df.drop(
                [c for c in ('correlation', 'correlation_plot') if c in summary_df.columns]
            )

    # Percentile labels (numeric-only columns)
    percentile_labels = [_percentile_to_label(p) for p in percentiles]

    # When no analyzed column is numeric (e.g. include='string', or a frame with
    # no numeric columns), the numeric-only stats are all empty - drop them so
    # the table isn't padded with columns of None. Booleans count as numeric
    # since they are analyzed as 0/1.
    has_numeric = any(schema[c].is_numeric() or schema[c] == pl.Boolean for c in all_cols)
    numeric_only_cols = (
        ['mean', 'std', 'min', 'max', 'iqr', 'skew', 'kurtosis', 'mad',
         'n_zero', 'pct_zero', 'pct_pos', 'pct_neg', 'n_outliers', 'pct_outliers',
         'normality_test', 'uniformity_test', 'correlation', 'correlation_plot']
        + percentile_labels
    )

    # Apply column filtering based on expanded mode
    if expanded:
        final_df = summary_df
        if not has_numeric:
            final_df = final_df.drop([c for c in numeric_only_cols if c in final_df.columns])
        # String/temporal stats are seeded as None on every column so the summary
        # schema stays stable regardless of dtypes. Drop the ones no analyzed
        # column populated - minimal mode already does this via has_string_data.
        unpopulated = [
            c for c in _STRING_ALL_COLS + _TEMPORAL_COLS
            if c in final_df.columns and final_df[c].null_count() == final_df.height
        ]
        if unpopulated:
            final_df = final_df.drop(unpopulated)
    else:
        # Minimal mode - only essential columns (percentile labels generated dynamically)
        essential_cols = ['column', 'dtype', 'count', 'null_count']
        if has_numeric:
            essential_cols += ['mean', 'std', 'min'] + percentile_labels + ['max', 'iqr']
        essential_cols += ['pct_missing']
        if has_numeric:
            essential_cols += ['n_outliers', 'skew']

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

        essential_cols.append('distribution_plot')

        # Only include the essential columns that exist in the dataframe
        available_cols = [c for c in essential_cols if c in summary_df.columns]

        # Add correlation at the very end if specified
        if corr_target and 'correlation' in summary_df.columns:
            available_cols.append('correlation')
        if corr_target and 'correlation_plot' in summary_df.columns:
            available_cols.append('correlation_plot')

        # Model usability columns are appended in minimal mode too when requested
        if model_usability:
            available_cols.extend(c for c in _USABILITY_COLS if c in summary_df.columns)

        final_df = summary_df.select(available_cols)

    # Return standard DataFrame if great_tables=False
    if not great_tables:
        return final_df

    # Build Great Tables object
    if expanded:
        # Full statistics mode
        # Calculate timing
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        return _build_expanded_gt_table(final_df, df.height, df.width, df, execution_time_ms, corr_target, percentiles, decimals, sep_mark, dec_mark, compact, pattern, title, model_usability, distribution_plot)
    else:
        # Calculate timing
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        return _build_minimal_gt_table(final_df, df.height, df.width, df, execution_time_ms, corr_target, decimals, sep_mark, dec_mark, compact, pattern, title, model_usability, distribution_plot)


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
_STRING_DEFAULT_COLS = ['top', 'top_freq', 'min_length', 'median_length', 'avg_length', 'max_length']
_STRING_EXPANDED_COLS = ['mode_share', 'top_3', 'sample_vals']
_STRING_ALL_COLS = _STRING_DEFAULT_COLS + _STRING_EXPANDED_COLS

# Temporal stat column names (earliest/latest timestamps)
_TEMPORAL_COLS = ['earliest', 'latest']

# Model usability column names (appended when model_usability=True)
_USABILITY_COLS = ['usability_flags', 'usability_score', 'recommendation']


def _truncate(value: object, limit: int = 30) -> str:
    """Stringify and truncate a value with an ellipsis for table display."""
    s = str(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _calculate_quantiles(series: pl.Series, percentiles: list[float]) -> dict:
    """Calculate quantiles for a series."""
    quantile_stats = {}
    for p in percentiles:
        label = _percentile_to_label(p)
        if len(series) > 0:
            val = float(series.quantile(p))
            quantile_stats[label] = val
        else:
            quantile_stats[label] = None
    return quantile_stats


def _suggest_optimal_dtype(series: pl.Series, current_dtype) -> str:
    """Suggest optimal data type for a series."""
    if len(series) == 0:
        return str(current_dtype)

    # Check if boolean - only materialize unique values when there are at most
    # two of them (avoids building a Python set from high-cardinality columns)
    if series.n_unique() <= 2:
        unique_vals = set(series.unique().to_list())
        if unique_vals.issubset({0, 1}) or unique_vals.issubset({True, False}):
            return "Bool"
    
    if current_dtype.is_integer():
        try:
            min_val = int(series.min())
            max_val = int(series.max())
        except (ValueError, TypeError):
            # Handle cases where min/max return non-integer types
            return str(current_dtype)
        
        # Check if fits in smaller integer types
        if -128 <= min_val <= 127 and -128 <= max_val <= 127:
            return "Int8"
        elif -32768 <= min_val <= 32767 and -32768 <= max_val <= 32767:
            return "Int16"
        elif -2147483648 <= min_val <= 2147483647:
            return "Int32"
        else:
            return "Int64"
    
    elif current_dtype.is_float():
        # For floats, check if values could be represented as integers.
        # Drop nulls/NaNs/Infs and use tolerance to avoid float equality pitfalls.
        non_null = series.drop_nulls()
        finite_mask = (~non_null.is_nan()) & (~non_null.is_infinite())
        finite = non_null.filter(finite_mask)

        if finite.len() == 0:
            return str(current_dtype)

        try:
            max_abs_diff = (finite - finite.round(0)).abs().max()
            if max_abs_diff is not None and float(max_abs_diff) <= 1e-9:
                return "Int64"
        except Exception:
            # Keep float recommendation if integer-likeness check fails.
            pass

        return "Float32"  # Usually sufficient for most data
    
    elif current_dtype == pl.String:
        # Check if categorical would be better
        try:
            # Use approximate count for large datasets (>300k rows)
            if len(series) > 300000:
                n_unique = series.approx_n_unique()
            else:
                n_unique = series.n_unique()
        except Exception:
            # Fallback for problematic dtypes
            n_unique = series.drop_nulls().value_counts().height
        
        n_total = len(series)
        if n_unique / n_total < 0.5:  # Less than 50% unique
            return "Categorical"
        return "String"
    
    return str(current_dtype)


def _numeric_values(data: pl.Series | Iterable[float]) -> list[float]:
    """Materialize numeric input without requiring NumPy."""
    values = data.to_list() if isinstance(data, pl.Series) else list(data)
    return [
        numeric_value
        for value in values
        if math.isfinite(numeric_value := float(value))
    ]


def _normal_ks_goodness_of_fit(values: list[float]):
    """Run a fitted-normal KS test on a deterministic bounded sample."""
    if len(values) > _NORMALITY_MAX_SAMPLE_SIZE:
        step = len(values) / _NORMALITY_MAX_SAMPLE_SIZE
        sample = [
            values[int(index * step)]
            for index in range(_NORMALITY_MAX_SAMPLE_SIZE)
        ]
    else:
        sample = values

    result = stats.goodness_of_fit(
        stats.norm,
        sample,
        statistic="ks",
        n_mc_samples=_NORMALITY_MONTE_CARLO_SAMPLES,
        random_state=0,
    )
    return result, len(sample)


def _test_normality(data: pl.Series | Iterable[float], test_type: str) -> str:
    """Perform normality test and return formatted result."""
    if not _check_scipy_availability():
        return "N/A (scipy not available)"

    values = _numeric_values(data)
    if len(values) < 3:
        return "N/A (insufficient data)"
    
    try:
        if test_type == "shapiro":
            if len(values) > 5000:
                # Shapiro-Wilk is unreliable for large samples, so fall back to
                # a fitted-normal KS goodness-of-fit test.
                result, sample_size = _normal_ks_goodness_of_fit(values)
                p_value = result.pvalue
                test_name = (
                    "Kolmogorov-Smirnov Monte Carlo "
                    f"(sample={sample_size}, n>5000)"
                )
            else:
                stat, p_value = stats.shapiro(values)
                test_name = "Shapiro-Wilk"

        elif test_type == "anderson":
            if "method" in inspect.signature(stats.anderson).parameters:
                result = stats.anderson(
                    values,
                    dist="norm",
                    method="interpolate",
                )
                is_normal = result.pvalue > 0.05
            else:
                result = stats.anderson(values, dist="norm")
                is_normal = result.statistic < result.critical_values[2]
            p_value = None  # not reported for A-D
            test_name = "Anderson-Darling"

        elif test_type == "ks":
            result, sample_size = _normal_ks_goodness_of_fit(values)
            p_value = result.pvalue
            test_name = (
                "Kolmogorov-Smirnov Monte Carlo "
                f"(sample={sample_size})"
            )
        
        # Format result
        if test_type == "anderson":
            result_str = "NORMAL" if is_normal else "NON-NORMAL"
            return f"{result_str} ({test_name})"
        else:
            alpha = 0.05
            is_normal = p_value > alpha
            result_str = "NORMAL" if is_normal else "NON-NORMAL"
            return f"{result_str} ({test_name}, p={p_value:.3f})"
            
    except Exception as e:
        return f"Error ({test_type}): {str(e)[:20]}"


def _test_uniformity(data: pl.Series | Iterable[float], test_type: str) -> str:
    """Perform uniformity test and return formatted result."""
    if not _check_scipy_availability():
        return "N/A (scipy not available)"

    values = _numeric_values(data)
    if len(values) < 5:
        return "N/A (insufficient data)"
    
    try:
        if test_type == "ks":
            # KS test against uniform distribution
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                return "N/A (constant data)"
            
            # Normalize to [0,1] for uniform test (asymptotic p-value for large n)
            value_range = max_val - min_val
            normalized = [(value - min_val) / value_range for value in values]
            ks_method = 'asymp' if len(values) > 5000 else 'auto'
            stat, p_value = stats.kstest(normalized, 'uniform', method=ks_method)
            test_name = "KS"
            
        elif test_type == "chi2":
            # Chi-square goodness of fit test
            # Create bins and expected frequencies
            n_bins = min(10, math.isqrt(len(values)))
            observed = (
                pl.Series("_values", values)
                .hist(bin_count=n_bins)
                .get_column("count")
                .to_list()
            )
            expected_count = len(values) / len(observed)
            
            # Remove bins with very low expected frequency
            if expected_count < 5 or len(observed) < 2:
                return "N/A (insufficient bins)"
            
            expected = [expected_count] * len(observed)
            stat, p_value = stats.chisquare(observed, expected)
            test_name = "Chi-square"
        
        # Format result
        alpha = 0.05
        is_uniform = p_value > alpha
        result_str = "UNIFORM" if is_uniform else "NON-UNIFORM"
        return f"{result_str} ({test_name}, p={p_value:.3f})"
        
    except Exception as e:
        return f"Error ({test_type}): {str(e)[:20]}"




def _calculate_shakiness_score(
    col_stats: dict, 
    missing_threshold: float,
    constant_threshold: float,
    skew_threshold: float,
    kurtosis_threshold: float,
    outlier_threshold: float,
    mode_share: float | None = None,
) -> int:
    """Calculate shakiness score based on data quality indicators."""
    score = 0

    # High missingness
    if col_stats.get('pct_missing', 0) > missing_threshold * 100:
        score += 1

    # Constant/quasi-constant: flag when a single value dominates the column.
    # Handle both exact and approximate N_Unique column names
    n_unique_val = col_stats.get('n_unique', col_stats.get('n_unique(approx)', 0))
    if n_unique_val == 1:
        score += 1
    elif mode_share is not None:
        # Exact dominant-value share: flag if the most common value covers at
        # least `constant_threshold` (e.g. 0.99 = 99%) of the non-null values.
        if mode_share >= constant_threshold:
            score += 1

    uniqueness_ratio = col_stats.get('uniqueness_ratio', 1)
    
    # ID-like (too many unique values)
    if uniqueness_ratio > 0.95:  # More than 95% unique
        score += 1
    
    # Extreme skewness
    skewness = col_stats.get('skew')
    if skewness is not None and abs(skewness) > skew_threshold:
        score += 1
    
    # High kurtosis
    kurtosis = col_stats.get('kurtosis')
    if kurtosis is not None and abs(kurtosis) > kurtosis_threshold:
        score += 1
    
    # Outlier-heavy
    pct_outliers = col_stats.get('pct_outliers', 0)
    if pct_outliers is not None and pct_outliers > outlier_threshold * 100:
        score += 1
    
    # Failed normality test
    normality_test = col_stats.get('normality_test', '')
    if 'NON-NORMAL' in normality_test:
        score += 1
    
    return score


# Model Usability Functions (inspired by polarsight)

def _check_missing_values_usability(stats: dict) -> set[str]:
    """Check for missing value flags using polarsight thresholds."""
    flags = set()
    null_pct = stats.get('pct_missing', 0)
    
    if null_pct > 90.0:  # High missing (>90%)
        flags.add("HM")
    elif null_pct > 50.0:  # Moderate missing (>50%)
        flags.add("MM")
    
    return flags


def _is_likely_id_column(col_name: str, stats: dict) -> bool:
    """
    Detect likely identifier columns from tokenized names and cardinality.

    Floating-point measures are not classified as IDs based on uniqueness alone.
    """
    tokenized_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", col_name)
    name_tokens = set(re.findall(r"[a-z0-9]+", tokenized_name.lower()))
    has_id_name = bool(name_tokens & {"id", "key", "code", "num", "number", "no"})
    
    # Get statistics
    n_unique = stats.get('n_unique', stats.get('n_unique(approx)', 0))
    n_total = stats.get('count', 0) + stats.get('null_count', 0)  # Total including nulls
    uniqueness_ratio = stats.get('uniqueness_ratio', 0)
    
    if n_total == 0:
        return False
    
    # High cardinality check (even with duplicates, many unique values suggests ID)
    high_cardinality = n_unique > 1000  # More than 1000 unique values
    
    # High-uniqueness floats are usually measurements, not identifiers.
    if not has_id_name and uniqueness_ratio > 0.95:
        if str(stats.get("dtype", "")).startswith(("Float", "Decimal")):
            return False
        return True
    
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
    n_unique_val = stats.get('n_unique', stats.get('n_unique(approx)', 0))
    count = stats.get('count', 1)
    col_name = stats.get('column', '')
    
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
    outlier_pct = stats.get('pct_outliers') or 0
    if outlier_pct > 10.0:  # Extreme outliers (>10%)
        flags.add("EO")
    
    # Check skew
    skew = stats.get('skew')
    if skew is not None and abs(skew) > 3.0:  # Extreme skew (|skew| > 3)
        flags.add("ES")
    
    # Check kurtosis
    kurtosis = stats.get('kurtosis')
    if kurtosis is not None and abs(kurtosis) > 7.0:  # Extreme kurtosis (|kurtosis| > 7)
        flags.add("EK")
    
    # Check normality
    normality_test = stats.get('normality_test', '')
    if 'NON-NORMAL' in normality_test:  # Non-normal distribution
        flags.add("NN")
    
    # Check zero values (None for non-numeric columns)
    zero_pct = stats.get('pct_zero') or 0
    if zero_pct > 80.0:  # High zero values (>80%)
        flags.add("ZH")
    
    return flags


def _check_correlation_reliability_usability(stats: dict, has_correlation: bool) -> set[str]:
    """Check if correlation values are reliable using polarsight logic."""
    flags = set()
    
    # If column is non-normal and we're showing correlations, flag it
    if has_correlation and 'NON-NORMAL' in stats.get('normality_test', ''):
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
    stats: dict,
    has_correlation: bool = False
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


def _count_outliers(
    series: pl.Series,
    method: str,
    bounds: list[float] | None,
    q25: float | None = None,
    q75: float | None = None,
) -> int:
    """Count outliers in a series using specified method.

    For the ``iqr`` method, precomputed ``q25``/``q75`` quartiles may be passed
    in to avoid recalculating quantiles the caller already has.
    """
    if len(series) == 0:
        return 0

    if method == "iqr":
        if q25 is None:
            q25 = series.quantile(0.25)
        if q75 is None:
            q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
    elif method == "percentile":
        lower_bound = series.quantile(bounds[0])
        upper_bound = series.quantile(bounds[1])
        
    elif method == "zscore":
        mean_val = series.mean()
        std_val = series.std()
        if std_val is None or std_val == 0:
            return 0
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
    
    # Count values outside bounds using boolean indexing
    mask = (series < lower_bound) | (series > upper_bound)
    return int(mask.sum())


def _sanitize_nanoplot_column(
    summary_df: pl.DataFrame,
    column: str,
    *,
    list_payload: bool
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
_CORR_MISSING_TEXT = "—"


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
        f'width:{_CORR_TRACK_WIDTH_PX}px;height:{_CORR_TRACK_HEIGHT_PX}px;'
        f'margin:0 auto;background:{_CORR_TRACK_COLOR};border-radius:2px;">'
        f'<div style="position:absolute;left:{half - 0.5}px;top:0;width:1px;'
        f'height:{_CORR_TRACK_HEIGHT_PX}px;background:{_CORR_ZERO_TICK_COLOR};"></div>'
        f'<div class="ps-corr-bar" style="position:absolute;left:{offset}px;'
        f'top:{bar_top}px;width:{length}px;height:{_CORR_BAR_HEIGHT_PX}px;'
        f'background:{color};border-radius:1px;"></div>'
        f'</div>'
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

    corr_cols = [c for c in ("correlation", "correlation_plot") if c in summary_df.columns]

    if has_corr_data and "correlation_plot" in summary_df.columns:
        gt_table = gt_table.fmt(_correlation_bar_html, columns="correlation_plot")

    return (
        gt_table
        .sub_missing(columns=["correlation"], missing_text=_CORR_MISSING_TEXT)
        .tab_spanner(label=f"Correlation with '{corr_target}'", columns=corr_cols)
        .cols_align(align="center", columns=corr_cols)
    )


def _float_format_columns(summary_df: pl.DataFrame) -> list[str]:
    """Return scalar float columns that should honor the decimals option."""
    return [
        column
        for column, dtype in summary_df.schema.items()
        if dtype.is_float() and column != "correlation_plot"
    ]


def _format_nanoplot_value(value: float) -> str:
    """Format a numeric nanoplot label with three significant digits."""
    return format(value, ".3g")


def _apply_nanoplots(gt_table: GT, has_hist_data: bool, distribution_plot: str) -> GT:
    """Apply histogram nanoplot formatting shared by both table modes."""
    # Histogram nanoplots for the distribution_plot column
    if has_hist_data:
        try:
            from great_tables import nanoplot_options
            if distribution_plot == "histogram":
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
                    )
                )
        except ImportError:
            # If nanoplot_options not available, use basic nanoplot
            gt_table = gt_table.fmt_nanoplot(columns="distribution_plot", plot_type="bar")
        except Exception:
            # If nanoplot formatting fails, continue without it
            pass

    return gt_table


def _build_minimal_gt_table(
    summary_df: pl.DataFrame,
    n_rows: int,
    n_cols: int,
    df: pl.DataFrame,
    execution_ms: float,
    corr_target: str | None,
    decimals: int,
    sep_mark: str,
    dec_mark: str,
    compact: bool,
    pattern: str | None,
    title: str | None,
    model_usability: bool = False,
    distribution_plot: str = "histogram"
) -> GT:
    """Build minimal Great Tables object."""
    summary_df, has_hist_data = _sanitize_nanoplot_column(summary_df, "distribution_plot", list_payload=True)
    summary_df, has_corr_data = _sanitize_nanoplot_column(summary_df, "correlation_plot", list_payload=False)

    # Determine column organization (detect percentile columns dynamically)
    pct_cols = sorted(
        [c for c in summary_df.columns if _is_percentile_label(c)],
        key=lambda x: float(x[:-1])
    )
    basic_cols = ["dtype", "count", "null_count", "mean", "std", "min"] + pct_cols + ["max"]
    essential_cols = ["iqr", "pct_missing", "n_outliers", "skew"]
    string_cols = list(_STRING_DEFAULT_COLS)
    temporal_cols = list(_TEMPORAL_COLS)
    quality_cols = list(_USABILITY_COLS) if model_usability else []

    # Filter to existing columns and ensure all are strings
    basic_cols = [str(c) for c in basic_cols if c in summary_df.columns]
    essential_cols = [str(c) for c in essential_cols if c in summary_df.columns]
    # Only include string/temporal stats columns if there's at least one non-null value
    string_cols = [str(c) for c in string_cols
                   if c in summary_df.columns and summary_df[c].null_count() < len(summary_df)]
    temporal_cols = [str(c) for c in temporal_cols
                     if c in summary_df.columns and summary_df[c].null_count() < len(summary_df)]
    quality_cols = [str(c) for c in quality_cols if c in summary_df.columns]
    
    try:
        # Use custom title or default
        table_title = title if title is not None else "🔬 DataFrame X-ray"
        
        gt_table = (
            GT(summary_df)
            .tab_header(
                title=table_title,
                subtitle=f"Dataset: {n_rows:,} rows × {n_cols} columns ({_format_memory_usage(df)} in memory) - X-rayed in {execution_ms:.0f} ms"
            )
        )
    except Exception as e:
        # If GT creation fails, return a basic table without advanced formatting
        raise ValueError(f"Great Tables formatting failed. Try using great_tables=False. Error: {e}")
    
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
        gt_table = gt_table.tab_spanner(label="Quality Assessment", columns=quality_cols)

    # Format integer columns (filter to those that actually exist)
    int_cols = [c for c in ["count", "null_count", "n_outliers", "top_freq", "min_length", "max_length"]
                if c in summary_df.columns]
    float_cols = _float_format_columns(summary_df)

    gt_table = (
        gt_table
        .fmt_integer(columns=int_cols, sep_mark=sep_mark)
        .fmt_number(
            columns=float_cols,
            decimals=decimals,
            drop_trailing_zeros=True,  # 1.00 -> 1, but 1.58 keeps its decimals
            sep_mark=sep_mark,
            dec_mark=dec_mark,
            compact=compact,
            **({"pattern": pattern} if pattern is not None else {})
        )
    )
    
    # Alignment - free-text usability columns are left-aligned
    center_cols = [str(c) for c in (basic_cols + essential_cols
                                    + [c for c in string_cols if c != "top"]
                                    + temporal_cols
                                    + [c for c in quality_cols if c == "usability_score"])]
    left_cols = (["column"] + [c for c in ["top"] if c in string_cols]
                 + [c for c in quality_cols if c != "usability_score"])

    gt_table = (
        gt_table
        .cols_align(align="center", columns=center_cols)
        .cols_align(align="left", columns=left_cols)
        .tab_options(
            table_font_size="13px",
            heading_background_color="#f8f9fa",
            column_labels_background_color="#e9ecef"
        )
    )

    gt_table = _apply_correlation_columns(gt_table, corr_target, summary_df, has_corr_data)
    gt_table = _apply_nanoplots(gt_table, has_hist_data, distribution_plot)

    return gt_table


def _build_expanded_gt_table(
    summary_df: pl.DataFrame,
    n_rows: int,
    n_cols: int,
    df: pl.DataFrame,
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
    distribution_plot: str = "histogram"
) -> GT:
    """Build expanded Great Tables object with all statistics."""
    summary_df, has_hist_data = _sanitize_nanoplot_column(summary_df, "distribution_plot", list_payload=True)
    summary_df, has_corr_data = _sanitize_nanoplot_column(summary_df, "correlation_plot", list_payload=False)

    # Organize columns by category
    basic_cols = ["dtype", "count", "mean", "std", "min", "max"]
    quantile_cols = [str(_percentile_to_label(p)) for p in percentiles if _percentile_to_label(p) in summary_df.columns]
    
    # Handle both exact and approximate N_Unique column names
    n_unique_cols = [c for c in ["n_unique", "n_unique(approx)"] if c in summary_df.columns]
    
    distribution_cols = ["iqr", "skew", "kurtosis", "mad", "distribution_plot"]
    count_cols = ["null_count", "pct_missing"] + n_unique_cols + ["uniqueness_ratio", "n_duplicates", "pct_duplicates", "n_zero", "pct_zero", "pct_pos", "pct_neg"]
    outlier_cols = ["n_outliers", "pct_outliers"]
    string_stat_cols = list(_STRING_ALL_COLS)
    temporal_stat_cols = list(_TEMPORAL_COLS)
    test_cols = ["normality_test", "uniformity_test"]
    quality_cols = ["opt_dtype", "shakiness_score", "quality_flag"]
    if model_usability:
        quality_cols.extend(_USABILITY_COLS)

    # Filter to existing columns and ensure all are strings
    basic_cols = [str(c) for c in basic_cols if c in summary_df.columns]
    distribution_cols = [str(c) for c in distribution_cols if c in summary_df.columns]
    count_cols = [str(c) for c in count_cols if c in summary_df.columns]
    outlier_cols = [str(c) for c in outlier_cols if c in summary_df.columns]
    # Only include string/temporal stats columns if there's at least one non-null value
    string_stat_cols = [str(c) for c in string_stat_cols
                        if c in summary_df.columns and summary_df[c].null_count() < len(summary_df)]
    temporal_stat_cols = [str(c) for c in temporal_stat_cols
                          if c in summary_df.columns and summary_df[c].null_count() < len(summary_df)]
    test_cols = [str(c) for c in test_cols if c in summary_df.columns]
    quality_cols = [str(c) for c in quality_cols if c in summary_df.columns]
    
    try:
        # Use custom title or default
        table_title = title if title is not None else "🔬 Expanded Statistics"
        
        gt_table = (
            GT(summary_df)
            .tab_header(
                title=table_title,
                subtitle=f"Dataset: {n_rows:,} rows × {n_cols} columns ({_format_memory_usage(df)} in memory) - X-rayed in {execution_ms:.0f} ms"
            )
        )
    except Exception as e:
        # If GT creation fails, return a basic table without advanced formatting
        raise ValueError(f"Great Tables formatting failed. Try using great_tables=False. Error: {e}")
    
    # Add spanners only for non-empty column groups
    if basic_cols:
        gt_table = gt_table.tab_spanner(label="Basic Statistics", columns=basic_cols)
    if quantile_cols:
        gt_table = gt_table.tab_spanner(label="Quantiles", columns=quantile_cols)
    if distribution_cols:
        gt_table = gt_table.tab_spanner(label="Distribution", columns=distribution_cols)
    if count_cols:
        gt_table = gt_table.tab_spanner(label="Counts & Ratios", columns=count_cols)
    if outlier_cols:
        gt_table = gt_table.tab_spanner(label="Outliers", columns=outlier_cols)
    if string_stat_cols:
        gt_table = gt_table.tab_spanner(label="String Statistics", columns=string_stat_cols)
    if temporal_stat_cols:
        gt_table = gt_table.tab_spanner(label="Temporal", columns=temporal_stat_cols)
    if test_cols:
        gt_table = gt_table.tab_spanner(label="Statistical Tests", columns=test_cols)
    if quality_cols:
        gt_table = gt_table.tab_spanner(label="Quality Assessment", columns=quality_cols)
    
    # Build format column lists (only include columns that exist)
    int_fmt_cols = [c for c in ["count", "null_count"] + n_unique_cols + ["n_duplicates", "n_zero", "n_outliers", "top_freq", "min_length", "max_length", "shakiness_score"] if c in summary_df.columns]
    float_fmt_cols = _float_format_columns(summary_df)
    
    gt_table = (
        gt_table
        .fmt_integer(columns=int_fmt_cols, sep_mark=sep_mark)
        .fmt_number(
            columns=float_fmt_cols,
            decimals=decimals,
            drop_trailing_zeros=True,  # 1.00 -> 1, but 1.58 keeps its decimals
            sep_mark=sep_mark,
            dec_mark=dec_mark,
            compact=compact,
            **({"pattern": pattern} if pattern is not None else {})
        )
    )

    # Alignment - free-text string columns are left-aligned, the rest centered
    text_string_cols = {"top", "top_3", "sample_vals"}
    center_cols = [str(c) for c in (basic_cols + quantile_cols + distribution_cols + count_cols + outlier_cols + [c for c in string_stat_cols if c not in text_string_cols] + temporal_stat_cols + ["shakiness_score"] + (["usability_score"] if model_usability and "usability_score" in summary_df.columns else []))]
    left_cols = [str(c) for c in (["column", "opt_dtype", "quality_flag"] + [c for c in string_stat_cols if c in text_string_cols] + (["usability_flags", "recommendation"] if model_usability else []) + test_cols)]

    gt_table = (
        gt_table
        .cols_align(align="center", columns=center_cols)
        .cols_align(align="left", columns=left_cols)
        .tab_options(
            table_font_size="12px",
            heading_background_color="#f8f9fa",
            column_labels_background_color="#e9ecef"
        )
    )

    gt_table = _apply_correlation_columns(gt_table, corr_target, summary_df, has_corr_data)
    gt_table = _apply_nanoplots(gt_table, has_hist_data, distribution_plot)

    return gt_table
