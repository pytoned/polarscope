from __future__ import annotations
import polars as pl
from great_tables import GT
from typing import Any, Union
import numpy as np
import time

# Optional scipy imports - lazy loaded to avoid import warnings
SCIPY_AVAILABLE = None  # Will be checked when needed
stats = None


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
        # Only string/text columns
        return [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.String, pl.Utf8)]
    
    elif include == 'temporal':
        # Only date/datetime columns
        return [c for c, dt in zip(df.columns, df.dtypes) if dt.is_temporal()]
    
    elif isinstance(include, list):
        # Specific data type names
        include_types = set(include)
        return [c for c, dt in zip(df.columns, df.dtypes) if str(dt) in include_types]
    
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
        - 'string': Only string/text columns
        - 'temporal': Only date/datetime columns
        - list of strings: Specific data type names (e.g., ['Float64', 'String'])
    great_tables : bool, default True
        Whether to return a formatted Great Tables object (True) or standard
        Polars DataFrame output (False).
    expanded : bool, default False
        If True, shows all available statistics. If False, shows only essential
        metrics: dtype, count, null_count, mean, std, min, 25%, 50%, 75%, max, 
        iqr, pct_missing, n_outliers, skew.
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
        Adds columns: Usability_Flags, Usability_Score, Recommendation.
    distribution_plot : str, default "histogram"
        Type of distribution visualization for numeric columns:
        - "histogram": Bar-based histogram (default) without markers
    decimals : int, default 2
        Number of decimal places for numeric formatting in Great Tables output.
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
        percentiles = sorted(set(percentiles))
        if any(p < 0 or p > 1 for p in percentiles):
            raise ValueError("percentiles values must be between 0 and 1")
    
    # Validate parameters
    if outlier_method not in ["iqr", "percentile", "zscore"]:
        raise ValueError("outlier_method must be 'iqr', 'percentile', or 'zscore'")
    
    if outlier_method == "percentile" and not outlier_bounds:
        raise ValueError("outlier_bounds must be provided when outlier_method='percentile'")
    
    if outlier_bounds and len(outlier_bounds) != 2:
        raise ValueError("outlier_bounds must be a list of exactly 2 values")
    
    if normality_test not in ["shapiro", "anderson", "ks"]:
        raise ValueError("normality_test must be 'shapiro', 'anderson', or 'ks'")
    
    if uniformity_test not in ["ks", "chi2"]:
        raise ValueError("uniformity_test must be 'ks' or 'chi2'")

    if distribution_plot not in ["histogram"]:
        raise ValueError("distribution_plot must be 'histogram'")
    
    # Validate correlation target
    if corr_target:
        if corr_target not in df.columns:
            raise ValueError(f"Target column '{corr_target}' not found in DataFrame")
        target_dtype = df.schema[corr_target]
        if not target_dtype.is_numeric():
            raise ValueError(f"Target column '{corr_target}' must be numeric, got {target_dtype}")
    
    # Filter columns based on include parameter
    all_cols = _get_columns_to_analyze(df, include)
    
    # Calculate comprehensive statistics for all columns
    stats_data = []
    
    schema = df.schema
    for col in all_cols:
        dtype = schema[col]
        is_numeric = dtype.is_numeric()
        series = df[col]
        series_clean = series.drop_nulls()
        n_total = len(series)
        n_valid = len(series_clean)
        n_missing = series.null_count()
        pct_missing = (n_missing / n_total * 100) if n_total > 0 else 0
        
        # Initialize column stats
        col_stats = {
            'Column': col,
            'Dtype': str(dtype),
            'Count': n_valid,
            'null_count': n_missing,
            'Pct_Missing': round(pct_missing, 2)
        }
        
        # Basic counts and ratios - use approx for large datasets
        is_large_dataset = len(series) > 300000
        try:
            # Use approximate count for large datasets (>300k rows) for better performance
            if is_large_dataset:
                n_unique = series.approx_n_unique()
            else:
                n_unique = series.n_unique()
        except Exception:
            # Fallback - count via groupby (more reliable for problematic dtypes)
            n_unique = series.drop_nulls().value_counts().height
        
        # Use appropriate column name based on dataset size
        n_unique_col = 'N_Unique(approx)' if is_large_dataset else 'N_Unique'
        col_stats[n_unique_col] = n_unique
        col_stats['Uniqueness_Ratio'] = round(n_unique / n_total, 4) if n_total > 0 else 0
        
        # Duplicate analysis for historized datasets
        n_duplicates = n_total - n_unique
        pct_duplicates = round((n_duplicates / n_total * 100), 2) if n_total > 0 else 0
        col_stats['N_Duplicates'] = n_duplicates
        col_stats['Pct_Duplicates'] = pct_duplicates
        
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
            
            col_stats['Mean'] = round(mean_val, 3)
            col_stats['std'] = round(std_val, 3) if std_val is not None else None
            col_stats['Min'] = round(min_val, 3)
            col_stats['Max'] = round(max_val, 3)
            
            # IQR
            if 0.25 in percentiles and 0.75 in percentiles:
                q25 = quantile_stats.get('25%')
                q75 = quantile_stats.get('75%') 
                if q25 is not None and q75 is not None:
                    col_stats['IQR'] = round(q75 - q25, 3)
            
            # Zero/positive/negative counts
            n_zero = int((series_clean == 0).sum())
            n_pos = int((series_clean > 0).sum())
            n_neg = int((series_clean < 0).sum())
            
            col_stats['N_Zero'] = n_zero
            col_stats['Pct_Zero'] = round(n_zero / n_valid * 100, 2) if n_valid > 0 else 0
            col_stats['Pct_Pos'] = round(n_pos / n_valid * 100, 2) if n_valid > 0 else 0
            col_stats['Pct_Neg'] = round(n_neg / n_valid * 100, 2) if n_valid > 0 else 0
            
            # Skewness (always calculated for default view)
            if n_valid > 2:
                skew_val = float(series_clean.skew())
                col_stats['skew'] = round(skew_val, 3)
            else:
                col_stats['skew'] = None
            
            # Distribution plot data for nanoplots
            if n_valid > 0:
                distribution_data = []
                try:
                    if distribution_plot == "histogram":
                        # Calculate optimal number of bins (max 12 for nanoplots)
                        n_bins = min(12, max(5, int(np.sqrt(n_valid))))

                        # Reuse the min/max already computed above for the range
                        # Create bin edges
                        if min_val == max_val:
                            # All values are the same
                            distribution_data = [n_valid]
                        else:
                            bin_edges = np.linspace(min_val, max_val, n_bins + 1)
                            # Calculate histogram using numpy
                            counts, _ = np.histogram(series_clean.to_numpy(), bins=bin_edges)
                            distribution_data = counts.tolist()
                    
                    
                    col_stats['Distribution_Plot'] = distribution_data
                except Exception:
                    # Fallback if distribution calculation fails
                    col_stats['Distribution_Plot'] = []
            else:
                col_stats['Distribution_Plot'] = []
            
            # Advanced statistics (expanded mode)
            if expanded:
                # MAD (Median Absolute Deviation)
                if n_valid > 0:
                    median_val = float(series_clean.median())
                    mad = (series_clean - median_val).abs().median()
                    col_stats['MAD'] = round(float(mad), 3)
                
                # Kurtosis (expanded mode only)
                if n_valid > 2:
                    # Calculate kurtosis (excess kurtosis)
                    kurt_val = _calculate_kurtosis(series_clean.to_numpy())
                    col_stats['Kurtosis'] = round(kurt_val, 3)
                
                # Optimal dtype suggestion
                col_stats['Opt_Dtype'] = _suggest_optimal_dtype(series_clean, dtype)
                
                # Statistical tests (if scipy available)
                if _check_scipy_availability() and n_valid > 3:
                    normality_result = _test_normality(series_clean.to_numpy(), normality_test)
                    col_stats['Normality_Test'] = normality_result
                    
                    uniformity_result = _test_uniformity(series_clean.to_numpy(), uniformity_test)
                    col_stats['Uniformity_Test'] = uniformity_result
            
            # Outlier detection (reuse already-computed quartiles for the iqr method)
            n_outliers = _count_outliers(
                series_clean, outlier_method, outlier_bounds,
                q25=quantile_stats.get('25%'), q75=quantile_stats.get('75%'),
            )
            pct_outliers = (n_outliers / n_valid * 100) if n_valid > 0 else 0
            col_stats['N_Outliers'] = n_outliers
            col_stats['Pct_Outliers'] = round(pct_outliers, 2)
            
            # Correlation with target
            if corr_target:
                if col == corr_target:
                    col_stats['Correlation'] = '-'  # Target column shows '-' instead of None
                    col_stats['Correlation_Plot'] = None  # No plot for target column
                else:
                    try:
                        corr_val = df.select([pl.corr(corr_target, col)]).item()
                        col_stats['Correlation'] = round(corr_val, 3) if corr_val is not None else None
                        # Create single value for horizontal bar nanoplot (fixed -1 to 1 scale)
                        # Round to 3 decimal places for better performance and readability
                        col_stats['Correlation_Plot'] = round(corr_val, 3) if corr_val is not None else 0.0
                    except Exception:
                        col_stats['Correlation'] = None
                        col_stats['Correlation_Plot'] = None

            # String-specific columns are N/A for numeric columns
            col_stats['Top'] = None
            col_stats['Top_Freq'] = None
            col_stats['Avg_Length'] = None

        else:
            # ── Non-numeric columns ──────────────────────────────────────
            is_string = dtype in (pl.String, pl.Utf8)

            if is_string and n_valid > 0:
                # String-specific statistics
                lengths = series_clean.str.len_chars()

                # Top value (mode) and its frequency
                value_counts = series_clean.value_counts().sort('count', descending=True)
                val_col = value_counts.columns[0]
                cnt_col = value_counts.columns[1]
                top_val = str(value_counts[val_col][0])
                col_stats['Top'] = top_val if len(top_val) <= 30 else top_val[:27] + "..."
                col_stats['Top_Freq'] = int(value_counts[cnt_col][0])
                col_stats['Avg_Length'] = round(float(lengths.mean()), 1)

                # Distribution plot - top category frequencies as bar chart
                n_bars = min(12, value_counts.height)
                col_stats['Distribution_Plot'] = value_counts[cnt_col][:n_bars].to_list()
            else:
                col_stats['Top'] = None
                col_stats['Top_Freq'] = None
                col_stats['Avg_Length'] = None
                col_stats['Distribution_Plot'] = []

            # Numeric-only statistics are N/A for non-numeric columns
            for stat in ['Mean', 'std', 'Min', 'Max', 'IQR']:
                col_stats[stat] = None
            for p in percentiles:
                label = _percentile_to_label(p)
                col_stats[label] = None
            col_stats['skew'] = None
            col_stats['N_Outliers'] = None
            col_stats['Pct_Outliers'] = None
            col_stats['N_Zero'] = None
            col_stats['Pct_Zero'] = None
            col_stats['Pct_Pos'] = None
            col_stats['Pct_Neg'] = None

            if expanded:
                col_stats['MAD'] = None
                col_stats['Kurtosis'] = None
                col_stats['Opt_Dtype'] = _suggest_optimal_dtype(series_clean, dtype)
                col_stats['Normality_Test'] = "N/A (non-numeric)"
                col_stats['Uniformity_Test'] = "N/A (non-numeric)"

            # Correlation with target for non-numeric columns
            if corr_target:
                if col == corr_target:
                    col_stats['Correlation'] = '-'
                    col_stats['Correlation_Plot'] = None
                else:
                    col_stats['Correlation'] = None
                    col_stats['Correlation_Plot'] = None
        
        # Dominant-value share for quasi-constant detection. An exact mode share
        # needs a value_counts pass, so only compute it for small/medium columns;
        # very large columns fall back to the cardinality heuristic for speed.
        mode_share = None
        if n_valid > 0 and not is_large_dataset:
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
        col_stats['Shakiness_Score'] = shakiness_score
        col_stats['Quality_Flag'] = "⚠ SHAKY" if shakiness_score >= shakiness_threshold else "✓ OK"
        
        # Model usability evaluation (if requested)
        if model_usability:
            usability_result = _evaluate_column_usability(
                col_stats,
                has_correlation=bool(corr_target)
            )
            col_stats['Usability_Flags'] = usability_result['flag_string']
            col_stats['Usability_Score'] = usability_result['score']
            col_stats['Recommendation'] = usability_result['recommendation']
        
        stats_data.append(col_stats)
    
    # Create DataFrame
    summary_df = pl.DataFrame(stats_data)
    
    # Apply column filtering based on expanded mode
    if expanded:
        final_df = summary_df
    else:
        # Minimal mode - only essential columns (percentile labels generated dynamically)
        percentile_labels = [_percentile_to_label(p) for p in percentiles]
        essential_cols = (['Column', 'Dtype', 'Count', 'null_count', 'Mean', 'std', 'Min']
                         + percentile_labels
                         + ['Max', 'IQR', 'Pct_Missing', 'N_Outliers', 'skew'])
        
        # Add string-specific columns only when string columns are present
        string_stat_cols = ['Top', 'Top_Freq', 'Avg_Length']
        has_string_data = any(
            c in summary_df.columns and summary_df[c].null_count() < len(summary_df)
            for c in string_stat_cols
        )
        if has_string_data:
            essential_cols.extend(string_stat_cols)
        
        essential_cols.append('Distribution_Plot')
        
        # Only include the essential columns that exist in the dataframe
        available_cols = [c for c in essential_cols if c in summary_df.columns]
        
        # Add correlation at the very end if specified
        if corr_target and 'Correlation' in summary_df.columns:
            available_cols.append('Correlation')
        if corr_target and 'Correlation_Plot' in summary_df.columns:
            available_cols.append('Correlation_Plot')
        
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
    if p == 0.5:
        return "50%"  # Keep 50% instead of "Median" for consistency
    else:
        return f"{int(p*100)}%"


def _calculate_quantiles(series: pl.Series, percentiles: list[float]) -> dict:
    """Calculate quantiles for a series."""
    quantile_stats = {}
    for p in percentiles:
        label = _percentile_to_label(p)
        if len(series) > 0:
            val = float(series.quantile(p))
            quantile_stats[label] = round(val, 3)
        else:
            quantile_stats[label] = None
    return quantile_stats


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis (kurtosis - 3)."""
    if len(data) < 4:
        return np.nan
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return np.nan
    
    # Calculate fourth moment
    fourth_moment = np.mean((data - mean) ** 4)
    kurtosis = fourth_moment / (std ** 4)
    
    # Return excess kurtosis (subtract 3 for normal distribution)
    return kurtosis - 3


def _suggest_optimal_dtype(series: pl.Series, current_dtype) -> str:
    """Suggest optimal data type for a series."""
    if len(series) == 0:
        return str(current_dtype)
    
    # Check if boolean
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


def _test_normality(data: np.ndarray, test_type: str) -> str:
    """Perform normality test and return formatted result."""
    if not _check_scipy_availability():
        return "N/A (scipy not available)"
    
    if len(data) < 3:
        return "N/A (insufficient data)"
    
    try:
        if test_type == "shapiro":
            if len(data) > 5000:
                # Shapiro-Wilk test not reliable for large samples
                return "N/A (sample too large)"
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
            
        elif test_type == "anderson":
            result = stats.anderson(data, dist='norm')
            # Anderson-Darling uses critical values, not p-values
            # Use 5% significance level (index 2 in critical_values)
            is_normal = result.statistic < result.critical_values[2]
            p_value = None  # A-D doesn't give exact p-value
            test_name = "Anderson-Darling"
            
        elif test_type == "ks":
            # Lilliefors test (KS test with estimated parameters)
            stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            test_name = "Kolmogorov-Smirnov"
        
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


def _test_uniformity(data: np.ndarray, test_type: str) -> str:
    """Perform uniformity test and return formatted result."""
    if not _check_scipy_availability():
        return "N/A (scipy not available)"
    
    if len(data) < 5:
        return "N/A (insufficient data)"
    
    try:
        if test_type == "ks":
            # KS test against uniform distribution
            min_val, max_val = np.min(data), np.max(data)
            if min_val == max_val:
                return "N/A (constant data)"
            
            # Normalize to [0,1] for uniform test
            normalized = (data - min_val) / (max_val - min_val)
            stat, p_value = stats.kstest(normalized, 'uniform')
            test_name = "KS"
            
        elif test_type == "chi2":
            # Chi-square goodness of fit test
            # Create bins and expected frequencies
            n_bins = min(10, int(np.sqrt(len(data))))
            observed, bin_edges = np.histogram(data, bins=n_bins)
            expected = np.full(n_bins, len(data) / n_bins)
            
            # Remove bins with very low expected frequency
            mask = expected >= 5
            if np.sum(mask) < 2:
                return "N/A (insufficient bins)"
            
            stat, p_value = stats.chisquare(observed[mask], expected[mask])
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
    if col_stats.get('Pct_Missing', 0) > missing_threshold * 100:
        score += 1

    # Constant/quasi-constant: flag when a single value dominates the column.
    uniqueness_ratio = col_stats.get('Uniqueness_Ratio', 1)
    # Handle both exact and approximate N_Unique column names
    n_unique_val = col_stats.get('N_Unique', col_stats.get('N_Unique(approx)', 0))
    if uniqueness_ratio == 0 or n_unique_val == 1:
        score += 1
    elif mode_share is not None:
        # Exact dominant-value share: flag if the most common value covers at
        # least `constant_threshold` (e.g. 0.99 = 99%) of the non-null values.
        if mode_share >= constant_threshold:
            score += 1
    elif uniqueness_ratio < (1 - constant_threshold):
        # Fallback heuristic when mode share is unavailable (very large columns
        # using approximate cardinality).
        score += 1
    
    # ID-like (too many unique values)
    if uniqueness_ratio > 0.95:  # More than 95% unique
        score += 1
    
    # Extreme skewness
    skewness = col_stats.get('skew')
    if skewness is not None and abs(skewness) > skew_threshold:
        score += 1
    
    # High kurtosis
    kurtosis = col_stats.get('Kurtosis')
    if kurtosis is not None and abs(kurtosis) > kurtosis_threshold:
        score += 1
    
    # Outlier-heavy
    pct_outliers = col_stats.get('Pct_Outliers', 0)
    if pct_outliers is not None and pct_outliers > outlier_threshold * 100:
        score += 1
    
    # Failed normality test
    normality_test = col_stats.get('Normality_Test', '')
    if 'NON-NORMAL' in normality_test:
        score += 1
    
    return score


# Model Usability Functions (inspired by polarsight)

def _check_missing_values_usability(stats: dict) -> set[str]:
    """Check for missing value flags using polarsight thresholds."""
    flags = set()
    null_pct = stats.get('Pct_Missing', 0)
    
    if null_pct > 90.0:  # High missing (>90%)
        flags.add("HM")
    elif null_pct > 50.0:  # Moderate missing (>50%)
        flags.add("MM")
    
    return flags


def _is_likely_id_column(col_name: str, stats: dict) -> bool:
    """
    Advanced ID column detection that considers multiple factors:
    1. Column name patterns (ID, _id, etc.)
    2. High cardinality even with duplicates (historized data)
    3. Sequential patterns in numeric IDs
    4. Format patterns in string IDs
    """
    col_name_lower = col_name.lower()
    
    # Check for ID-like column names
    id_name_patterns = ['id', '_id', 'key', '_key', 'code', '_code', 'num', '_num', 'no', '_no']
    has_id_name = any(pattern in col_name_lower for pattern in id_name_patterns)
    
    # Get statistics
    n_unique = stats.get('N_Unique', stats.get('N_Unique(approx)', 0))
    n_total = stats.get('Count', 0) + stats.get('null_count', 0)  # Total including nulls
    uniqueness_ratio = stats.get('Uniqueness_Ratio', 0)
    
    if n_total == 0:
        return False
    
    # High cardinality check (even with duplicates, many unique values suggests ID)
    high_cardinality = n_unique > 1000  # More than 1000 unique values
    
    # Traditional high uniqueness ratio (for non-ID named columns, be strict)
    # But exclude obvious numeric columns that happen to be unique
    if not has_id_name and uniqueness_ratio > 0.95:
        # Check if this looks like a numeric column that's just unique by chance
        # (e.g., revenue, price, etc. that happen to have no duplicates)
        if col_name_lower in ['revenue', 'price', 'amount', 'value', 'cost', 'sales', 'profit', 'income']:
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
    n_unique_val = stats.get('N_Unique', stats.get('N_Unique(approx)', 0))
    count = stats.get('Count', 1)
    col_name = stats.get('Column', '')
    
    if count > 0:
        # Use the uniqueness ratio from stats if available, otherwise calculate it
        # uniqueness_ratio = stats.get('Uniqueness_Ratio', 0) * 100  # Not used in this function
        
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
    outlier_pct = stats.get('Pct_Outliers') or 0
    if outlier_pct > 10.0:  # Extreme outliers (>10%)
        flags.add("EO")
    
    # Check skew
    skew = stats.get('skew')
    if skew is not None and abs(skew) > 3.0:  # Extreme skew (|skew| > 3)
        flags.add("ES")
    
    # Check kurtosis
    kurtosis = stats.get('Kurtosis')
    if kurtosis is not None and abs(kurtosis) > 7.0:  # Extreme kurtosis (|kurtosis| > 7)
        flags.add("EK")
    
    # Check normality
    normality_test = stats.get('Normality_Test', '')
    if 'NON-NORMAL' in normality_test:  # Non-normal distribution
        flags.add("NN")
    
    # Check zero values (None for non-numeric columns)
    zero_pct = stats.get('Pct_Zero') or 0
    if zero_pct > 80.0:  # High zero values (>80%)
        flags.add("ZH")
    
    return flags


def _check_correlation_reliability_usability(stats: dict, has_correlation: bool) -> set[str]:
    """Check if correlation values are reliable using polarsight logic."""
    flags = set()
    
    # If column is non-normal and we're showing correlations, flag it
    if has_correlation and 'NON-NORMAL' in stats.get('Normality_Test', ''):
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
        "NN": 1.5,  # Non-normal distribution (p-value < 0.01)
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
        if std_val == 0:
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
                    if isinstance(item, (int, float, np.number)) and np.isfinite(float(item)):
                        numeric_values.append(float(item))
                if numeric_values:
                    has_valid_payload = True
                    cleaned.append(numeric_values)
                else:
                    cleaned.append(None)
            elif isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
                has_valid_payload = True
                cleaned.append([float(value)])
            else:
                cleaned.append(None)
    else:
        for value in values:
            if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
                has_valid_payload = True
                cleaned.append(float(value))
            else:
                cleaned.append(None)

    return summary_df.with_columns(pl.Series(column, cleaned)), has_valid_payload


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
    summary_df, has_hist_data = _sanitize_nanoplot_column(summary_df, "Distribution_Plot", list_payload=True)
    summary_df, has_corr_data = _sanitize_nanoplot_column(summary_df, "Correlation_Plot", list_payload=False)

    # Determine column organization (detect percentile columns dynamically)
    pct_cols = sorted(
        [c for c in summary_df.columns if c.endswith('%') and c[:-1].isdigit()],
        key=lambda x: int(x[:-1])
    )
    basic_cols = ["Dtype", "Count", "null_count", "Mean", "std", "Min"] + pct_cols + ["Max"]
    essential_cols = ["IQR", "Pct_Missing", "N_Outliers", "skew"]
    string_cols = ["Top", "Top_Freq", "Avg_Length"]
    plot_cols = ["Distribution_Plot"]
    quality_cols = []
    
    # Filter to existing columns and ensure all are strings
    basic_cols = [str(c) for c in basic_cols if c in summary_df.columns]
    essential_cols = [str(c) for c in essential_cols if c in summary_df.columns]
    # Only include string stats columns if there's at least one non-null value
    string_cols = [str(c) for c in string_cols
                   if c in summary_df.columns and summary_df[c].null_count() < len(summary_df)]
    plot_cols = [str(c) for c in plot_cols if c in summary_df.columns]
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
    if quality_cols:
        gt_table = gt_table.tab_spanner(label="Quality", columns=quality_cols)
    
    # Format integer columns (filter to those that actually exist)
    int_cols = [c for c in ["Count", "null_count", "N_Outliers", "Top_Freq"] if c in summary_df.columns]
    num_cols = [c for c in ["Mean", "std", "Min"] + pct_cols + ["Max", "IQR", "skew"] if c in summary_df.columns]
    
    gt_table = (
        gt_table
        .fmt_integer(columns=int_cols, sep_mark=sep_mark)
        .fmt_number(
            columns=num_cols, 
            decimals=decimals, 
            sep_mark=sep_mark, 
            dec_mark=dec_mark,
            compact=compact,
            **({"pattern": pattern} if pattern is not None else {})
        )
        .fmt_number(columns=["Pct_Missing"], decimals=1, sep_mark=sep_mark, dec_mark=dec_mark)
    )
    
    # Format string-specific columns
    if "Avg_Length" in summary_df.columns and "Avg_Length" in string_cols:
        gt_table = gt_table.fmt_number(columns=["Avg_Length"], decimals=1, sep_mark=sep_mark, dec_mark=dec_mark)
    
    # Alignment
    center_cols = [str(c) for c in (basic_cols + essential_cols + [c for c in string_cols if c != "Top"] + quality_cols)]
    left_cols = ["Column"] + ([c for c in ["Top"] if c in string_cols])
    
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
    
    # Add correlation if specified
    if corr_target and "Correlation" in summary_df.columns:
        gt_table = (
            gt_table
            .tab_spanner(label=f"Correlation with '{corr_target}'", columns=["Correlation"])
            # Note: Skip fmt_number for Correlation since it contains both numbers and '-' string
            .cols_align(align="center", columns=["Correlation"])
        )
    
    # Add histogram nanoplots for numeric columns (if Histogram column exists)
    if has_hist_data:
        try:
            from great_tables import nanoplot_options
            if distribution_plot == "histogram":
                gt_table = gt_table.fmt_nanoplot(
                    columns="Distribution_Plot",
                    plot_type="bar",
                    options=nanoplot_options(
                        data_bar_stroke_width=0,  # No gaps between bars (like histogram)
                        data_bar_fill_color="#4A90E2",
                        show_data_line=False,
                        show_data_area=False
                    )
                )
        except ImportError:
            # If nanoplot_options not available, use basic nanoplot
            plot_type = "bar"
            gt_table = gt_table.fmt_nanoplot(columns="Distribution_Plot", plot_type=plot_type)
        except Exception:
            # If nanoplot formatting fails, continue without it
            pass
    
    # Add correlation nanoplots (horizontal bars with fixed -1 to 1 scale)
    if has_corr_data:
        try:
            from great_tables import nanoplot_options
            gt_table = gt_table.fmt_nanoplot(
                columns="Correlation_Plot",
                plot_type="bar",
                expand_x=[-1, 1],  # Fixed scale from -1 to 1
                options=nanoplot_options(
                    data_bar_stroke_width=1,
                    data_bar_fill_color="#4A90E2",
                    data_bar_negative_fill_color="#E24A4A",  # Red for negative correlations
                    show_data_line=False,
                    show_data_area=False
                )
            )
        except ImportError:
            # If nanoplot_options not available, use basic nanoplot
            gt_table = gt_table.fmt_nanoplot(columns="Correlation_Plot", plot_type="bar")
        except Exception:
            # If nanoplot formatting fails, continue without it
            pass
    
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
    summary_df, has_hist_data = _sanitize_nanoplot_column(summary_df, "Distribution_Plot", list_payload=True)
    summary_df, has_corr_data = _sanitize_nanoplot_column(summary_df, "Correlation_Plot", list_payload=False)

    # Organize columns by category
    basic_cols = ["Dtype", "Count", "Mean", "std", "Min", "Max"]
    quantile_cols = [str(_percentile_to_label(p)) for p in percentiles if _percentile_to_label(p) in summary_df.columns]
    
    # Handle both exact and approximate N_Unique column names
    n_unique_cols = [c for c in ["N_Unique", "N_Unique(approx)"] if c in summary_df.columns]
    
    distribution_cols = ["IQR", "skew", "Kurtosis", "MAD", "Distribution_Plot"]
    count_cols = ["null_count", "Pct_Missing"] + n_unique_cols + ["Uniqueness_Ratio", "N_Duplicates", "Pct_Duplicates", "N_Zero", "Pct_Zero", "Pct_Pos", "Pct_Neg"]
    outlier_cols = ["N_Outliers", "Pct_Outliers"]
    string_stat_cols = ["Top", "Top_Freq", "Avg_Length"]
    test_cols = ["Normality_Test", "Uniformity_Test"]
    quality_cols = ["Opt_Dtype", "Shakiness_Score", "Quality_Flag"]
    if model_usability:
        quality_cols.extend(["Usability_Flags", "Usability_Score", "Recommendation"])
    
    # Filter to existing columns and ensure all are strings
    basic_cols = [str(c) for c in basic_cols if c in summary_df.columns]
    distribution_cols = [str(c) for c in distribution_cols if c in summary_df.columns]
    count_cols = [str(c) for c in count_cols if c in summary_df.columns]
    outlier_cols = [str(c) for c in outlier_cols if c in summary_df.columns]
    # Only include string stats columns if there's at least one non-null value
    string_stat_cols = [str(c) for c in string_stat_cols
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
    if test_cols:
        gt_table = gt_table.tab_spanner(label="Statistical Tests", columns=test_cols)
    if quality_cols:
        gt_table = gt_table.tab_spanner(label="Quality Assessment", columns=quality_cols)
    
    # Build format column lists (only include columns that exist)
    int_fmt_cols = [c for c in ["Count", "null_count"] + n_unique_cols + ["N_Duplicates", "N_Zero", "N_Outliers", "Top_Freq", "Shakiness_Score"] + (["Usability_Score"] if model_usability and "Usability_Score" in summary_df.columns else []) if c in summary_df.columns]
    num_fmt_cols = [c for c in ["Mean", "std", "Min", "Max", "IQR", "MAD"] + quantile_cols if c in summary_df.columns]
    
    gt_table = (
        gt_table
        .fmt_integer(columns=int_fmt_cols, sep_mark=sep_mark)
        .fmt_number(
            columns=num_fmt_cols, 
            decimals=decimals, 
            sep_mark=sep_mark, 
            dec_mark=dec_mark,
            compact=compact,
            **({"pattern": pattern} if pattern is not None else {})
        )
        .fmt_number(columns=[c for c in ["skew", "Kurtosis"] if c in summary_df.columns], decimals=3, sep_mark=sep_mark, dec_mark=dec_mark)
        .fmt_number(columns=[c for c in ["Pct_Missing", "Pct_Duplicates", "Pct_Zero", "Pct_Pos", "Pct_Neg", "Pct_Outliers"] if c in summary_df.columns], decimals=1, sep_mark=sep_mark, dec_mark=dec_mark)
        .fmt_number(columns=[c for c in ["Uniqueness_Ratio"] if c in summary_df.columns], decimals=4, sep_mark=sep_mark, dec_mark=dec_mark)
    )
    
    # Format string-specific columns
    if "Avg_Length" in summary_df.columns and "Avg_Length" in string_stat_cols:
        gt_table = gt_table.fmt_number(columns=["Avg_Length"], decimals=1, sep_mark=sep_mark, dec_mark=dec_mark)
    
    # Alignment
    center_cols = [str(c) for c in (basic_cols + quantile_cols + distribution_cols + count_cols + outlier_cols + [c for c in string_stat_cols if c != "Top"] + ["Shakiness_Score"] + (["Usability_Score"] if model_usability and "Usability_Score" in summary_df.columns else []))]
    left_cols = [str(c) for c in (["Column", "Opt_Dtype", "Quality_Flag"] + ([c for c in ["Top"] if c in string_stat_cols]) + (["Usability_Flags", "Recommendation"] if model_usability else []) + test_cols)]
    
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
    
    # Add correlation if specified
    if corr_target and "Correlation" in summary_df.columns:
        gt_table = (
            gt_table
            .tab_spanner(label=f"Correlation with '{corr_target}'", columns=["Correlation"])
            # Note: Skip fmt_number for Correlation since it contains both numbers and '-' string
            .cols_align(align="center", columns=["Correlation"])
        )
    
    # Add histogram nanoplots for numeric columns (if Histogram column exists)
    if has_hist_data:
        try:
            from great_tables import nanoplot_options
            if distribution_plot == "histogram":
                gt_table = gt_table.fmt_nanoplot(
                    columns="Distribution_Plot",
                    plot_type="bar",
                    options=nanoplot_options(
                        data_bar_stroke_width=0,  # No gaps between bars (like histogram)
                        data_bar_fill_color="#4A90E2",
                        show_data_line=False,
                        show_data_area=False
                    )
                )
        except ImportError:
            # If nanoplot_options not available, use basic nanoplot
            plot_type = "bar"
            gt_table = gt_table.fmt_nanoplot(columns="Distribution_Plot", plot_type=plot_type)
        except Exception:
            # If nanoplot formatting fails, continue without it
            pass
    
    # Add correlation nanoplots (horizontal bars with fixed -1 to 1 scale)
    if has_corr_data:
        try:
            from great_tables import nanoplot_options
            gt_table = gt_table.fmt_nanoplot(
                columns="Correlation_Plot",
                plot_type="bar",
                expand_x=[-1, 1],  # Fixed scale from -1 to 1
                options=nanoplot_options(
                    data_bar_stroke_width=1,
                    data_bar_fill_color="#4A90E2",
                    data_bar_negative_fill_color="#E24A4A",  # Red for negative correlations
                    show_data_line=False,
                    show_data_area=False
                )
            )
        except ImportError:
            # If nanoplot_options not available, use basic nanoplot
            gt_table = gt_table.fmt_nanoplot(columns="Correlation_Plot", plot_type="bar")
        except Exception:
            # If nanoplot formatting fails, continue without it
            pass
    
    return gt_table
