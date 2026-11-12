"""Plotting helpers built on the Plotly and Altair backends.

Every plotting function returns the backend's own figure object rather than a
polarscope wrapper, so the full backend API stays available without polarscope
re-exposing it. Options that are not explicit parameters here are reached in one
of two ways.

Per figure, by chaining onto the result::

    ps.missingval_plot(df).update_layout(template="plotly_dark")

Globally, using the backend's own theme setting::

    import plotly.io as pio
    pio.templates.default = "plotly_dark"    # or alt.themes.enable(...)

Both are honoured because these functions only ever set the properties they
manage (title, axis titles, width, height) and leave the rest to the backend.
A Plotly template styles only properties the figure has not set explicitly, so
the correlation heatmaps keep their built-in ``colorscale``; override that with
``update_traces(colorscale=...)`` if needed.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import polars as pl
import polars.selectors as cs

from ._helpers import literal_name_map, require_dataframe

# Moved to polarscope.clean; re-exported here for backward compatibility.
from .clean import convert_datatypes, drop_missing  # noqa: F401

# ---------- helpers ----------

_VALID_BACKENDS = ("plotly", "altair")
_BACKEND_ERROR = "backend must be 'plotly' or 'altair'"
_ALTAIR_INSTALL_ERROR = (
    "The Altair backend requires the optional 'altair' dependency. "
    "Install it with `pip install \"polarscope[altair]\"`."
)


def _require_altair():
    """Import Altair or raise an actionable optional-dependency error."""
    try:
        import altair as alt
    except ImportError as exc:
        raise ImportError(_ALTAIR_INSTALL_ERROR) from exc
    return alt


def _validate_backend(backend: str) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(_BACKEND_ERROR)
    if backend == "altair":
        _require_altair()


def _apply_chart_size(chart, width, height):
    properties = {}
    if width is not None:
        properties["width"] = width
    if height is not None:
        properties["height"] = height
    return chart.properties(**properties) if properties else chart


def _empty_plot(message: str, backend: str, width, height):
    """Return an empty backend-specific figure that explains why it is empty."""
    if backend == "plotly":
        import plotly.graph_objects as go

        figure = go.Figure()
        figure.update_layout(title=message, width=width, height=height)
        return figure

    alt = _require_altair()
    chart = (
        alt.Chart(alt.Data(values=[{"message": message}]))
        .mark_text()
        .encode(text="message:N")
    )
    return _apply_chart_size(chart, width, height)


def _altair_data(rows: list[dict]):
    """Wrap already aggregated rows (bins, categories, matrix cells) for Altair.

    Altair's default data transformer refuses DataFrames with more than 5,000
    rows when a chart is serialized (``MaxRowsError``), which a 71-column
    correlation matrix already exceeds. Inline values are not subject to that
    check, so these charts render at any size without touching the user's
    global ``alt.data_transformers`` setting. Non-finite floats become None so
    the spec stays valid JSON.
    """
    alt = _require_altair()
    values = [
        {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    return alt.InlineData(values=values)


def _literal_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]]:
    """Rename pattern-like column names (``"*"``, ``"^a.*$"``) to safe stand-ins.

    ``pl.col(name)`` expands such names as a wildcard or regex. Build
    expressions with ``mapping.get(name, name)`` on the returned frame and keep
    showing the original names.
    """
    mapping = literal_name_map(df.columns)
    return (df.rename(mapping) if mapping else df), mapping


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    try:
        return list(df.select(cs.numeric()).columns)
    except Exception:
        return [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]

def _ensure_columns(df: pl.DataFrame, columns: Iterable[str] | None) -> list[str]:
    if columns is None:
        return _numeric_columns(df)

    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return []

    try:
        numeric_set = set(df.select(cs.numeric()).columns)
        return [c for c in valid_cols if c in numeric_set]
    except Exception:
        dtypes = dict(zip(df.columns, df.dtypes))
        return [c for c in valid_cols if dtypes[c].is_numeric()]


def _pairwise_correlations(
    df: pl.DataFrame,
    pairs: Sequence[tuple[str, str]],
    method: str,
) -> list[float | None]:
    """Correlation of each ``(a, b)`` column pair, computed in one ``select``.

    For Spearman, every column without nulls is ranked once and a pair of such
    columns gets the Pearson correlation of their ranks. That is exactly what
    ``pl.corr(method="spearman")`` computes (both rank NaN highest), without
    re-ranking both columns for every pair. ``pl.corr`` drops the rows where
    either column is null before ranking, so pairs involving a column with
    nulls keep the per-pair Spearman.
    """
    if not pairs:
        return []
    work, mapping = _literal_columns(df)
    names = list(dict.fromkeys(name for pair in pairs for name in pair))
    alias = {name: f"c{i}" for i, name in enumerate(names)}
    work = work.select(
        [pl.col(mapping.get(name, name)).alias(alias[name]) for name in names]
    )

    ranked: set[str] = set()
    if method == "spearman":
        null_counts = work.null_count().row(0)
        ranked = {c for c, nulls in zip(work.columns, null_counts) if nulls == 0}
        work = work.with_columns(
            [pl.col(c).rank("average").alias(f"r{c}") for c in work.columns if c in ranked]
        )

    expressions = []
    for k, (a, b) in enumerate(pairs):
        x, y = alias[a], alias[b]
        if x in ranked and y in ranked:
            expression = pl.corr(f"r{x}", f"r{y}", method="pearson")
        else:
            expression = pl.corr(x, y, method=method)
        expressions.append(expression.alias(f"p{k}"))

    return [
        float(value) if value is not None else None
        for value in work.select(expressions).row(0)
    ]


def _correlation_matrix(
    df: pl.DataFrame,
    columns: Sequence[str],
    method: str,
) -> list[list[float | None]]:
    """Compute a symmetric correlation matrix with Polars expressions."""
    positions = [
        (row_idx, col_idx)
        for row_idx in range(len(columns))
        for col_idx in range(row_idx, len(columns))
    ]
    values = _pairwise_correlations(
        df, [(columns[i], columns[j]) for i, j in positions], method
    )
    matrix: list[list[float | None]] = [
        [None for _ in columns] for _ in columns
    ]
    for (row_idx, col_idx), correlation in zip(positions, values):
        matrix[row_idx][col_idx] = correlation
        matrix[col_idx][row_idx] = correlation

    return matrix


def _nice_bins(
    lo: float, hi: float, max_bins: int, min_step: float
) -> tuple[float, float, float, int]:
    """Choose bins for ``[lo, hi]`` the way Vega-Lite's ``maxbins`` does.

    Returns ``(start, stop, step, precision)``: a "nice" step (1, 2 or 5 times
    a power of ten, at least ``min_step``) that gives at most about
    ``max_bins`` bins, ``start``/``stop`` rounded outwards to multiples of it,
    and the number of decimals to round bin edges to.
    """
    span = (hi - lo) or abs(lo) or 1.0
    level = math.ceil(math.log10(max_bins))
    step = max(min_step, 10.0 ** (math.floor(math.log10(span) + 0.5) - level))
    while math.ceil(span / step) > max_bins:
        step *= 10
    for divisor in (5, 2):
        if step / divisor >= min_step and span / (step / divisor) <= max_bins:
            step /= divisor
    precision = 0 if step >= 1 else int(-math.log10(step)) + 1
    step = round(step, precision)
    start = math.floor(lo / step + 10.0 ** (-precision - 1)) * step
    if lo < start:
        start -= step
    stop = math.ceil(hi / step) * step
    if stop <= start:
        stop = start + step
    return round(start, precision), round(stop, precision), step, precision


def _format_edge(value: float, precision: int) -> str:
    return f"{value:.{precision}f}" if abs(value) < 1e15 else f"{value:.6g}"


def _histogram(s: pl.Series, max_bins: int) -> dict:
    """Bin a numeric Series in Polars; return one entry per bin.

    Nulls, NaN and infinite values are left out. Bins follow Vega-Lite's
    ``maxbins`` rule (what the Altair backend always used). Integer columns get
    whole-number bins centred on the integers, as Plotly does for integer data,
    and the largest value gets its own bin instead of sharing the last one.
    """
    is_integer = s.dtype.is_integer()
    # Some Series methods run as pl.col(<series name>) expressions, so a name
    # like "^a.*$" would be read as a regex: use a fixed name.
    values = s.rename("value").cast(pl.Float64)
    values = values.filter(values.is_finite())
    if values.len() == 0:
        return {"start": [], "end": [], "center": [], "count": [], "label": [], "step": 1.0}

    lo, hi = values.min(), values.max()
    start, stop, step, precision = _nice_bins(
        lo, hi + 1 if is_integer else hi, max_bins, 1.0 if is_integer else 0.0
    )
    n_bins = max(1, round(stop / step - start / step))
    index = (
        ((values - start) / step + 1e-14).floor().clip(0, n_bins - 1).cast(pl.Int64)
    )
    counts = [0] * n_bins
    for bin_index, count in index.to_frame("bin").group_by("bin").len().iter_rows():
        counts[bin_index] = count

    # "+ 0.0" turns a rounded -0.0 into 0.0.
    edges = [round(start + i * step, precision) + 0.0 for i in range(n_bins + 1)]
    if is_integer:
        # [a, b) holds the integers a .. b - 1; draw it from a - 0.5 to b - 0.5.
        labels = [
            f"{int(a)}" if step == 1 else f"{int(a)} to {int(b) - 1}"
            for a, b in zip(edges, edges[1:])
        ]
        edges = [edge - 0.5 for edge in edges]
    else:
        decimals = max(0, -math.floor(math.log10(step) + 1e-9))
        labels = [
            f"[{_format_edge(a, decimals)}, {_format_edge(b, decimals)})"
            for a, b in zip(edges, edges[1:])
        ]
    return {
        "start": edges[:-1],
        "end": edges[1:],
        "center": [round((a + b) / 2, precision + 1) for a, b in zip(edges, edges[1:])],
        "count": counts,
        "label": labels,
        "step": step,
    }


# ---------- Plotly backends ----------

def _dist_plot_plotly(hist: dict, column: str, width, height):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Bar(
        x=hist["center"], y=hist["count"], width=hist["step"], customdata=hist["label"],
        hovertemplate="%{customdata}<br>Count: %{y}<extra></extra>",
    )])
    fig.update_layout(title=f"Distribution: {column}", xaxis_title=column, yaxis_title="Count",
                      bargap=0, width=width, height=height)
    return fig

def _missingval_plot_plotly(cols: List[str], ratios: List[float], counts: List[int], width, height, normalize: bool = False):
    import plotly.graph_objects as go
    if normalize:
        # x-axis = ratio, labels = percentage strings
        x_values = ratios
        text_values = [f"{r*100:.1f}%" for r in ratios]
        x_title = "Share of missing values"
    else:
        # x-axis = absolute counts, labels = count integers
        x_values = counts
        text_values = [str(c) for c in counts]
        x_title = "Number of missing values"
    fig = go.Figure(data=[go.Bar(y=cols, x=x_values, orientation="h", text=text_values, textposition="outside")])
    fig.update_layout(title="Missing values per column", xaxis_title=x_title,
                      yaxis_title="Columns", width=width, height=height)
    return fig

def _cat_plot_plotly(cat_data: dict, width, height):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    n_cols = len(cat_data)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=list(cat_data.keys()),
        # 0.1 of the width per gap, but never more than 30% in total: Plotly
        # rejects spacings above 1 / (cols - 1), i.e. from 12 columns on.
        horizontal_spacing=min(0.1, 0.3 / max(n_cols - 1, 1))
    )
    
    for i, (col_name, data) in enumerate(cat_data.items(), 1):
        # Combine top and bottom values
        all_values = data['top_values'] + data['bottom_values']
        all_counts = data['top_counts'] + data['bottom_counts']
        
        if all_values:
            # Create colors (blue for top, red for bottom)
            colors = ['steelblue'] * len(data['top_values']) + ['indianred'] * len(data['bottom_values'])
            
            fig.add_trace(
                go.Bar(
                    x=all_values,
                    y=all_counts,
                    marker_color=colors,
                    text=all_counts,
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                ),
                row=1, col=i
            )
            
            # Update x-axis for this subplot
            fig.update_xaxes(tickangle=45, row=1, col=i)
            fig.update_yaxes(title_text="Count" if i == 1 else "", row=1, col=i)
    
    fig.update_layout(
        title="Categorical Value Frequencies",
        width=width or max(800, n_cols * 300),
        height=height or 500,
        showlegend=False
    )
    
    return fig

def _corr_heatmap_plotly_enhanced(row_labels: list, col_labels: list, mat: List[List[float]], annotate: bool, method: str, target: str, width, height):
    import plotly.graph_objects as go
    
    # Handle None values for filtered correlations
    display_mat = []
    text_mat = []
    for row in mat:
        display_row = []
        text_row = []
        for val in row:
            if val is None:
                display_row.append(0)  # Show as neutral color
                text_row.append("")    # No text
            else:
                display_row.append(val)
                text_row.append(f"{val:.3f}" if annotate else "")
        display_mat.append(display_row)
        text_mat.append(text_row)
    
    # Create enhanced heatmap
    fig = go.Figure(data=go.Heatmap(
        z=display_mat,
        x=col_labels,
        y=row_labels,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title=f"{method.title()}<br>Correlation"),
        text=text_mat,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>' +
                     f'{method.title()} Correlation: %{{z:.3f}}<extra></extra>'
    ))
    
    # Set appropriate title
    if target:
        title = f"{method.title()} Correlation with '{target}'"
    else:
        title = f"{method.title()} Correlation Matrix"
    
    fig.update_layout(
        title=title,
        width=width or 700,
        height=height or 600,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def _corr_plot_plotly(columns: list[str], corr_matrix: list, method: str, clustered: bool, width, height):
    import plotly.graph_objects as go
    
    # Enhanced interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=columns,
        y=columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title=f"{method.title()}<br>Correlation"),
        text=[[f"{val:.3f}" for val in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>' +
                     f'{method.title()} Correlation: %{{z:.3f}}<extra></extra>'
    ))
    
    title = f"{method.title()} Correlation Matrix"
    if clustered:
        title += " (Clustered)"
    
    fig.update_layout(
        title=title,
        width=width or 700,
        height=height or 600,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

# ---------- Altair backends ----------

def _corr_heatmap_altair_enhanced(row_labels: list, col_labels: list, mat: List[List[float]], annotate: bool, method: str, target: str, width, height):
    alt = _require_altair()
    
    # Prepare data for altair - handle None values
    data = []
    for i, row_name in enumerate(row_labels):
        for j, col_name in enumerate(col_labels):
            val = mat[i][j]
            if val is not None:
                data.append({
                    "row": row_name,
                    "col": col_name,
                    "correlation": val
                })
    
    if not data:
        return _empty_plot("No correlations to display", "altair", width or 400, height or 300)
    
    df_data = _altair_data(data)
    
    # Create base heatmap
    base = alt.Chart(df_data).mark_rect().encode(
        x=alt.X('col:N', title='Variables'),
        y=alt.Y('row:N', title='Variables'),
        color=alt.Color('correlation:Q', 
                       scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                       legend=alt.Legend(title=f'{method.title()} Correlation')),
        tooltip=['row:N', 'col:N', 'correlation:Q']
    )
    
    # Set title
    if target:
        title = f"{method.title()} Correlation with '{target}'"
    else:
        title = f"{method.title()} Correlation Matrix"
    
    base = _apply_chart_size(base.properties(title=title), width, height)
    
    # Add text annotations if requested
    if annotate:
        text = alt.Chart(df_data).mark_text(
            color='white',
            fontSize=10
        ).encode(
            x='col:N', 
            y='row:N', 
            text=alt.Text('correlation:Q', format='.3f')
        )
        text = _apply_chart_size(text, width, height)
        return base + text
    
    return base

def _dist_plot_altair(hist: dict, column: str, width, height):
    alt = _require_altair()
    rows = [
        {"bin_start": a, "bin_end": b, "count": n, "bin": label}
        for a, b, n, label in zip(hist["start"], hist["end"], hist["count"], hist["label"])
    ]
    # Fixed field names: the user's column name is only ever a title, so
    # Vega-Lite never parses it as a field (":" types, "." / "[]" nesting).
    chart = alt.Chart(_altair_data(rows)).mark_bar().encode(
        x=alt.X(field="bin_start", type="quantitative", bin="binned", title=column),
        x2=alt.X2(field="bin_end"),
        y=alt.Y(field="count", type="quantitative", title="Count"),
        tooltip=[
            alt.Tooltip(field="bin", type="nominal", title=column),
            alt.Tooltip(field="count", type="quantitative", title="Count"),
        ],
    ).properties(title=f"Distribution: {column}")
    return _apply_chart_size(chart, width, height)

def _missingval_plot_altair(cols: List[str], ratios: List[float], counts: List[int], width, height, normalize: bool = False):
    alt = _require_altair()
    if normalize:
        x_field = "ratio"
        x_title = "Share of missing values"
        text_values = [f"{r*100:.1f}%" for r in ratios]
    else:
        x_field = "count"
        x_title = "Number of missing values"
        text_values = [str(c) for c in counts]
    data = [{"column": c, "ratio": r, "count": n, "text_value": t} for c, r, n, t in zip(cols, ratios, counts, text_values)]
    df_data = _altair_data(data)
    base = alt.Chart(df_data).mark_bar().encode(
        y=alt.Y("column:N", sort=None, title="Columns"),
        x=alt.X(f"{x_field}:Q", title=x_title),
        tooltip=["column:N", "count:Q", "ratio:Q"]
    ).properties(title="Missing values per column")
    base = _apply_chart_size(base, width, height)
    text = alt.Chart(df_data).mark_text(align="left", baseline="middle", dx=3).encode(
        y="column:N", x=f"{x_field}:Q", text="text_value:N"
    )
    text = _apply_chart_size(text, width, height)
    return base + text

def _cat_plot_altair(cat_data: dict, width, height):
    alt = _require_altair()
    
    # Prepare data for all columns
    all_data = []
    for col_name, data in cat_data.items():
        # Add top values
        for val, count in zip(data['top_values'], data['top_counts']):
            all_data.append({
                'column': col_name,
                'value': str(val),
                'count': count,
                'type': 'top'
            })
        # Add bottom values
        for val, count in zip(data['bottom_values'], data['bottom_counts']):
            all_data.append({
                'column': col_name,
                'value': str(val),
                'count': count,
                'type': 'bottom'
            })
    
    if not all_data:
        return _empty_plot("No categorical values to display", "altair", width, height)
    
    df_data = _altair_data(all_data)
    
    chart = alt.Chart(df_data).mark_bar().encode(
        x=alt.X('value:N', title='Categories'),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color('type:N', 
                       scale=alt.Scale(domain=['top', 'bottom'], 
                                     range=['steelblue', 'indianred']),
                       legend=alt.Legend(title="Category Type")),
        tooltip=['column:N', 'value:N', 'count:Q', 'type:N'],
        facet=alt.Facet('column:N', title='Categorical Variables')
    ).properties(
        title="Categorical Value Frequencies"
    ).resolve_scale(
        x='independent'
    )
    
    chart_width = width // len(cat_data) if width is not None else None
    return _apply_chart_size(chart, chart_width, height)

def _corr_plot_altair(columns: list[str], corr_matrix: list, method: str, width, height):
    alt = _require_altair()
    
    # Prepare data for altair
    data = []
    for i, row_name in enumerate(columns):
        for j, col_name in enumerate(columns):
            data.append({
                "row": row_name,
                "col": col_name,
                "correlation": corr_matrix[i][j]
            })
    
    df_data = _altair_data(data)
    
    chart = alt.Chart(df_data).mark_rect().encode(
        x=alt.X('col:N', title='Variables'),
        y=alt.Y('row:N', title='Variables'),
        color=alt.Color('correlation:Q', 
                       scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                       legend=alt.Legend(title=f'{method.title()} Correlation')),
        tooltip=['row:N', 'col:N', 'correlation:Q']
    ).properties(
        title=f"{method.title()} Correlation Matrix"
    )
    
    return _apply_chart_size(chart, width, height)

# ---------- Public APIs ----------


def corr_heatmap(
    df: pl.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    split: str | None = None,
    threshold: float | None = None,
    target: str | None = None,
    method: str = "pearson",
    annotate: bool = True,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
):
    """
    Generate a color-encoded correlation heatmap to visualize relationships between numeric columns.
    
    This function creates a comprehensive correlation heatmap with advanced filtering and 
    targeting options, similar to klib's corr_mat function but optimized for Polars DataFrames.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing numeric columns to analyze. Non-numeric columns are automatically excluded.
    columns : Sequence[str] | None, optional
        Specific columns to include in correlation analysis. If None, uses all numeric columns.
    split : str | None, optional
        Type of correlation filtering to apply. Options:
        - None: Show all correlations between feature columns
        - "pos": Show only positive correlations above threshold
        - "neg": Show only negative correlations below -threshold  
        - "high": Show correlations where abs(correlation) > threshold
        - "low": Show correlations where abs(correlation) < threshold
    threshold : float | None, default None
        Correlation threshold value between 0 and 1. Used with split parameter:
        - For "pos"/"neg": minimum absolute correlation to display
        - For "high"/"low": threshold for filtering correlations
        - If None: 0.3 when split is "high" or "low", otherwise 0.0. An
          explicit value, including 0.0, is always used as given.
    target : str | None, optional
        Target column for correlation analysis. When specified, shows correlations 
        between each feature column and the target column only.
    method : str, default "pearson"
        Correlation calculation method:
        - "pearson": Linear correlation (assumes normal distribution)
        - "spearman": Rank-based correlation (monotonic relationships)
    annotate : bool, default True
        Whether to display correlation values as text annotations on each cell.
    width : int | None, optional
        Plot width in pixels. If None, uses backend default.
    height : int | None, optional
        Plot height in pixels. If None, uses backend default.
    backend : str, default "plotly"
        Visualization backend:
        - "plotly": Interactive heatmap with hover details (default)
        - "altair": Grammar of graphics heatmap

    Returns
    -------
    Figure object
        Correlation heatmap visualization (plotly Figure or altair Chart).
        Chain backend methods onto it to customize further - see the module
        docstring.
    """
    require_dataframe(df, "corr_heatmap")

    # Validate parameters
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be 'pearson' or 'spearman'")
    
    if split and split not in ["pos", "neg", "high", "low"]:
        raise ValueError("split must be None, 'pos', 'neg', 'high', or 'low'")
    
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    
    _validate_backend(backend)
    
    # Default threshold: 0.3 for high/low splits, 0.0 otherwise
    if threshold is None:
        threshold = 0.3 if split in ["high", "low"] else 0.0
    
    # Get numeric columns
    if target:
        # Target correlation mode - ensure target exists and is numeric
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        target_dtype = df.schema[target]
        if not target_dtype.is_numeric():
            raise ValueError(f"Target column '{target}' must be numeric, got {target_dtype}")
        
        # Get all other numeric columns for correlation with target
        all_numeric = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
        cols = [c for c in all_numeric if c != target]
        
        if columns:
            # Filter to specified columns (excluding target)
            cols = [c for c in cols if c in columns]
        
        if len(cols) == 0:
            raise ValueError("No numeric columns available for target correlation")
        
        correlations = [
            value if value is not None else 0.0
            for value in _pairwise_correlations(
                df, [(target, col) for col in cols], method
            )
        ]
        
        # Create target correlation matrix (1 row)
        mat = [correlations]
        correlation_cols = cols
        correlation_rows = [target]
        
    else:
        # Standard correlation matrix mode
        cols = _ensure_columns(df, columns)
        if len(cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation matrix")
        
        mat = _correlation_matrix(df, cols, method)
        correlation_cols = cols
        correlation_rows = cols
    
    # Apply split filtering if specified
    if split:
        filtered_mat = []
        for i, row in enumerate(mat):
            filtered_row = []
            for j, val in enumerate(row):
                if split == "pos" and val > threshold:
                    filtered_row.append(val)
                elif split == "neg" and val < -threshold:
                    filtered_row.append(val)
                elif split == "high" and abs(val) > threshold:
                    filtered_row.append(val)
                elif split == "low" and abs(val) < threshold:
                    filtered_row.append(val)
                else:
                    filtered_row.append(None)
            filtered_mat.append(filtered_row)
        mat = filtered_mat
    
    # Check if we have any data to plot after filtering
    if split and all(all(val is None for val in row) for row in mat):
        return _empty_plot(
            f'No correlations found with split="{split}" and threshold={threshold}',
            backend,
            width,
            height,
        )
    
    # Generate plot with appropriate backend
    if backend == "plotly":
        return _corr_heatmap_plotly_enhanced(correlation_rows, correlation_cols, mat, annotate, method, target, width, height)
    elif backend == "altair":
        return _corr_heatmap_altair_enhanced(correlation_rows, correlation_cols, mat, annotate, method, target, width, height)
    else:
        raise ValueError(_BACKEND_ERROR)

def dist_plot(
    df: pl.DataFrame,
    column: str | None = None,
    *,
    bins: int = 30,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
):
    """
    Create a distribution plot (histogram) for a numeric column.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing the column to plot.
    column : str | None, optional
        Name of the numeric column to plot. If None, uses the first numeric column.
    bins : int, default 30
        Maximum number of histogram bins. The bin width is rounded to a "nice"
        value (1, 2 or 5 times a power of ten; whole numbers for integer
        columns), so the plot may show fewer bins. The data are binned in
        Polars and only the per-bin counts go into the figure, so its size
        does not grow with the number of rows. Nulls, NaN and infinite values
        are left out.
    width : int | None, optional
        Width of the plot in pixels. If None, uses backend default.
    height : int | None, optional
        Height of the plot in pixels. If None, uses backend default.
    backend : str, default "plotly"
        Plotting backend to use. Options: "plotly", "altair".

    Returns
    -------
    Figure object
        The distribution plot figure (plotly Figure or altair Chart).
        Chain backend methods onto it to customize further - see the module
        docstring.
    """
    require_dataframe(df, "dist_plot")
    _validate_backend(backend)
    if bins < 1:
        raise ValueError("bins must be at least 1")

    cols = _numeric_columns(df)
    if column is None:
        if not cols:
            return _empty_plot("No numeric column found", backend, width, height)
        column = cols[0]
    elif column not in df.columns:
        return _empty_plot(f"Column '{column}' not found", backend, width, height)
    else:
        if not df[column].dtype.is_numeric():
            return _empty_plot(f"Column '{column}' is not numeric", backend, width, height)

    # get_column takes the name literally, unlike pl.col("*") or pl.col("^a.*$").
    hist = _histogram(df.get_column(column), bins)

    if backend == "plotly":
        return _dist_plot_plotly(hist, column, width, height)
    elif backend == "altair":
        return _dist_plot_altair(hist, column, width, height)
    else:
        raise ValueError(_BACKEND_ERROR)


def missingval_plot(
    df: pl.DataFrame,
    *,
    sort: str = "desc",
    normalize: bool = False,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
):
    """
    Create a horizontal bar plot showing missing values per column.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to analyze for missing values.
    sort : str, default "desc"
        How to sort columns by missing value count. Options: "desc", "asc", "none".
    normalize : bool, default False
        If True, display percentages on the x-axis. If False, display absolute counts.
    width : int | None, optional
        Width of the plot in pixels. If None, uses backend default.
    height : int | None, optional
        Height of the plot in pixels. If None, uses backend default.
    backend : str, default "plotly"
        Plotting backend to use. Options: "plotly", "altair".

    Returns
    -------
    Figure object
        The missing values plot figure (plotly Figure or altair Chart).
        Chain backend methods onto it to customize further, e.g.
        ``ps.missingval_plot(df).update_layout(template="plotly_dark")`` -
        see the module docstring.
    """
    require_dataframe(df, "missingval_plot")
    cols = list(df.columns)
    if sort not in {"desc", "asc", "none"}:
        raise ValueError("sort must be 'desc', 'asc', or 'none'")

    _validate_backend(backend)

    if not cols:
        return _empty_plot("No columns", backend, width, height)

    work, _ = _literal_columns(df)
    missing_expressions = []
    for column, dtype in work.schema.items():
        missing = pl.col(column).is_null()
        if dtype.is_float():
            missing = missing | pl.col(column).is_nan()
        missing_expressions.append(missing.sum().alias(column))

    null_counts_row = work.select(missing_expressions)
    total = df.height
    counts = [int(count) for count in null_counts_row.row(0)]
    ratios = [cnt / max(total, 1) for cnt in counts]

    order = list(range(len(cols)))
    if sort == "desc":
        order.sort(key=lambda i: ratios[i], reverse=True)
    elif sort == "asc":
        order.sort(key=lambda i: ratios[i])

    cols_o  = [cols[i] for i in order]
    ratios_o = [ratios[i] for i in order]
    counts_o = [counts[i] for i in order]

    if backend == "plotly":
        return _missingval_plot_plotly(cols_o, ratios_o, counts_o, width, height, normalize)
    elif backend == "altair":
        return _missingval_plot_altair(cols_o, ratios_o, counts_o, width, height, normalize)
    else:
        raise ValueError(_BACKEND_ERROR)


def cat_plot(
    df: pl.DataFrame,
    *,
    top: int = 10,
    bottom: int = 10,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
):
    """
    Create categorical value frequency plots showing top and bottom categories.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing categorical columns.
    top : int, default 10
        Number of most frequent categories to show per column.
    bottom : int, default 10  
        Number of least frequent categories to show per column.
    width : int | None, optional
        Width of the plot in pixels. If None, uses backend default.
    height : int | None, optional
        Height of the plot in pixels. If None, uses backend default.
    backend : str, default "plotly"
        Plotting backend to use. Options: "plotly", "altair".

    Returns
    -------
    Figure object
        The categorical plot figure (plotly Figure or altair Chart).
        Chain backend methods onto it to customize further - see the module
        docstring.
    """
    require_dataframe(df, "cat_plot")
    if top < 0 or bottom < 0:
        raise ValueError("top and bottom must be non-negative")

    _validate_backend(backend)

    # Get categorical columns (string/categorical types)
    cat_cols = [
        c
        for c, dt in zip(df.columns, df.dtypes)
        if dt == pl.String or isinstance(dt, (pl.Categorical, pl.Enum))
    ]
    
    if not cat_cols:
        return _empty_plot("No categorical columns found", backend, width, height)
    
    # Calculate value counts for each categorical column
    work, mapping = _literal_columns(df)
    cat_data = {}
    for col in cat_cols:
        # Own aliases: value_counts() calls its count field "count", which
        # collides with a column of that name. Ties are ordered by value so
        # the plot is the same on every run.
        value_counts = (
            work.select(pl.col(mapping.get(col, col)).alias("value"))
            .group_by("value")
            .agg(pl.len().alias("count"))
            .sort(
                [pl.col("count"), pl.col("value").cast(pl.String)],
                descending=[True, False],
                nulls_last=True,
            )
        )

        # Slice in Polars; only the plotted values become Python lists. The
        # bottom block starts after the top block, so no value appears twice.
        top_part = value_counts.head(top)
        if bottom > 0:
            bottom_part = value_counts.slice(max(top, value_counts.height - bottom))
        else:
            bottom_part = value_counts.clear()

        cat_data[col] = {
            'top_values': top_part.get_column("value").to_list(),
            'top_counts': top_part.get_column("count").to_list(),
            'bottom_values': bottom_part.get_column("value").to_list(),
            'bottom_counts': bottom_part.get_column("count").to_list()
        }
    
    if backend == "plotly":
        return _cat_plot_plotly(cat_data, width, height)
    elif backend == "altair":
        return _cat_plot_altair(cat_data, width, height)
    else:
        raise ValueError(_BACKEND_ERROR)




def corr_plot(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    *,
    method: str = "pearson",
    interactive: bool = True,
    clustered: bool = False,
    width: int | None = None,
    height: int | None = None,
    backend: str = "plotly",
) -> object:
    """
    Create enhanced correlation plots with multiple visualization options.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame.
    columns : list[str] | None, optional
        Specific columns to include. If None, uses all numeric columns.
    method : str, default "pearson"
        Correlation method ("pearson", "spearman").
    interactive : bool, default True
        Deprecated compatibility parameter. The explicit ``backend`` selection
        controls which plotting library is used.
    clustered : bool, default False
        Whether to cluster correlations by similarity (requires scipy).
    width : int | None, optional
        Plot width in pixels.
    height : int | None, optional
        Plot height in pixels.
    backend : str, default "plotly"
        Plotting backend ("plotly", "altair").

    Returns
    -------
    Figure object
        Enhanced correlation plot (plotly Figure or altair Chart).
        Chain backend methods onto it to customize further - see the module
        docstring.
    """
    require_dataframe(df, "corr_plot")
    _validate_backend(backend)

    # Get numeric columns
    if columns is None:
        columns = _numeric_columns(df)
    else:
        columns = _ensure_columns(df, columns)
    
    if len(columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation plot")
    
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    corr_matrix = _correlation_matrix(df, columns, method)
    
    # Clustering logic
    if clustered:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform
            
            # Replace NaN with 0 in correlation matrix before computing distances
            # NaN occurs when a column has no variance or all nulls
            corr_clean = [
                [
                    float(value)
                    if value is not None and math.isfinite(float(value))
                    else 0.0
                    for value in row
                ]
                for row in corr_matrix
            ]
            distance_matrix = [
                [
                    0.0 if row_idx == col_idx else 1 - abs(value)
                    for col_idx, value in enumerate(row)
                ]
                for row_idx, row in enumerate(corr_clean)
            ]
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method='average')
            cluster_order = [int(index) for index in leaves_list(linkage_matrix)]
            
            corr_matrix = [
                [corr_matrix[row_idx][col_idx] for col_idx in cluster_order]
                for row_idx in cluster_order
            ]
            columns = [columns[i] for i in cluster_order]
            
        except ImportError:
            print("scipy not installed, skipping clustering")
        except Exception as exc:
            print(f"Clustering failed ({exc.__class__.__name__}: {exc}), skipping clustering")
    
    if backend == "plotly":
        return _corr_plot_plotly(columns, corr_matrix, method, clustered, width, height)
    elif backend == "altair":
        return _corr_plot_altair(columns, corr_matrix, method, width, height)
    else:
        raise ValueError(_BACKEND_ERROR)


