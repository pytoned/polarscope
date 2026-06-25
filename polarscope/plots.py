
from __future__ import annotations
from typing import Iterable, Sequence, List
import math
import polars as pl
import polars.selectors as cs

# ---------- helpers ----------

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

# ---------- Plotly backends ----------

def _dist_plot_plotly(s: pl.Series, column: str, bins: int, width, height):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Histogram(x=s.to_numpy(), nbinsx=bins)])
    fig.update_layout(title=f"Distribution: {column}", xaxis_title=column, yaxis_title="Count",
                      width=width, height=height)
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
        horizontal_spacing=0.1
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

def _corr_plot_plotly(columns: list[str], corr_matrix: list, method: str, interactive: bool, clustered: bool, width, height):
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
    import altair as alt
    
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
        # Return empty chart with message
        return alt.Chart(pl.DataFrame({'message': ['No correlations to display']})).mark_text(
            text='No correlations to display'
        ).properties(width=width or 400, height=height or 300)
    
    df_data = pl.DataFrame(data)
    
    # Create base heatmap
    base = alt.Chart(df_data).mark_rect().encode(
        x=alt.X('col:N', title='Variables'),
        y=alt.Y('row:N', title='Variables'),
        color=alt.Color('correlation:Q', 
                       scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                       legend=alt.Legend(title=f'{method.title()} Correlation')),
        tooltip=['row', 'col', 'correlation']
    )
    
    # Set title
    if target:
        title = f"{method.title()} Correlation with '{target}'"
    else:
        title = f"{method.title()} Correlation Matrix"
    
    base = base.properties(title=title)
    if width: base = base.properties(width=width)
    if height: base = base.properties(height=height)
    
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
        if width: text = text.properties(width=width)
        if height: text = text.properties(height=height)
        return base + text
    
    return base

def _dist_plot_altair(s: pl.Series, column: str, bins: int, width, height):
    import altair as alt
    df_data = pl.DataFrame({column: s})
    chart = alt.Chart(df_data).mark_bar().encode(
        x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=bins)),
        y="count()",
        tooltip=[column]
    ).properties(title=f"Distribution: {column}")
    if width:  chart = chart.properties(width=width)
    if height: chart = chart.properties(height=height)
    return chart

def _missingval_plot_altair(cols: List[str], ratios: List[float], counts: List[int], width, height, normalize: bool = False):
    import altair as alt
    if normalize:
        x_field = "ratio"
        x_title = "Share of missing values"
        text_values = [f"{r*100:.1f}%" for r in ratios]
    else:
        x_field = "count"
        x_title = "Number of missing values"
        text_values = [str(c) for c in counts]
    data = [{"column": c, "ratio": r, "count": n, "text_value": t} for c, r, n, t in zip(cols, ratios, counts, text_values)]
    df_data = pl.DataFrame(data)
    base = alt.Chart(df_data).mark_bar().encode(
        y=alt.Y("column:N", sort=None, title="Columns"),
        x=alt.X(f"{x_field}:Q", title=x_title),
        tooltip=["column", "count", "ratio"]
    ).properties(title="Missing values per column")
    if width:  base = base.properties(width=width)
    if height: base = base.properties(height=height)
    text = alt.Chart(df_data).mark_text(align="left", baseline="middle", dx=3).encode(
        y="column:N", x=f"{x_field}:Q", text="text_value:N"
    )
    if width:  text = text.properties(width=width)
    if height: text = text.properties(height=height)
    return base + text

def _cat_plot_altair(cat_data: dict, width, height):
    import altair as alt
    
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
        # Return empty chart
        return alt.Chart(pl.DataFrame({'x': [0], 'y': [0]})).mark_point()
    
    df_data = pl.DataFrame(all_data)
    
    chart = alt.Chart(df_data).mark_bar().encode(
        x=alt.X('value:N', title='Categories'),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color('type:N', 
                       scale=alt.Scale(domain=['top', 'bottom'], 
                                     range=['steelblue', 'indianred']),
                       legend=alt.Legend(title="Category Type")),
        tooltip=['column', 'value', 'count', 'type'],
        facet=alt.Facet('column:N', title='Categorical Variables')
    ).properties(
        title="Categorical Value Frequencies"
    ).resolve_scale(
        x='independent'
    )
    
    if width: chart = chart.properties(width=width // len(cat_data))
    if height: chart = chart.properties(height=height)
    
    return chart

def _corr_plot_altair(columns: list[str], corr_matrix: list, method: str, width, height):
    import altair as alt
    
    # Prepare data for altair
    data = []
    for i, row_name in enumerate(columns):
        for j, col_name in enumerate(columns):
            data.append({
                "row": row_name,
                "col": col_name,
                "correlation": corr_matrix[i][j]
            })
    
    df_data = pl.DataFrame(data)
    
    chart = alt.Chart(df_data).mark_rect().encode(
        x=alt.X('col:N', title='Variables'),
        y=alt.Y('row:N', title='Variables'),
        color=alt.Color('correlation:Q', 
                       scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                       legend=alt.Legend(title=f'{method.title()} Correlation')),
        tooltip=['row', 'col', 'correlation']
    ).properties(
        title=f"{method.title()} Correlation Matrix"
    )
    
    if width: chart = chart.properties(width=width)
    if height: chart = chart.properties(height=height)
    
    return chart

# ---------- Public APIs ----------

_VALID_BACKENDS = ("plotly", "altair")
_BACKEND_ERROR = "backend must be 'plotly' or 'altair'"


def corr_heatmap(
    df: pl.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    split: str | None = None,
    threshold: float = 0.0,
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
    threshold : float, default 0.0
        Correlation threshold value between 0 and 1. Used with split parameter:
        - For "pos"/"neg": minimum absolute correlation to display
        - For "high"/"low": threshold for filtering correlations
        - Default becomes 0.3 when split is "high" or "low"
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
    """
    # Validate parameters
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be 'pearson' or 'spearman'")
    
    if split and split not in ["pos", "neg", "high", "low"]:
        raise ValueError("split must be None, 'pos', 'neg', 'high', or 'low'")
    
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    
    if backend not in _VALID_BACKENDS:
        raise ValueError(_BACKEND_ERROR)
    
    # Set default threshold for high/low splits
    if split in ["high", "low"] and threshold == 0.0:
        threshold = 0.3
    
    # Get numeric columns
    if target:
        # Target correlation mode - ensure target exists and is numeric
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        target_dtype = df.select(pl.col(target)).dtypes[0]
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
        
        # Calculate target correlations
        correlations = []
        for col in cols:
            if method == "pearson":
                corr_val = df.select([pl.corr(target, col)]).item()
            else:  # spearman
                # Rank-based correlation
                ranked_df = df.select([
                    pl.col(target).rank().alias('target_rank'),
                    pl.col(col).rank().alias('col_rank')
                ])
                corr_val = ranked_df.select([pl.corr('target_rank', 'col_rank')]).item()
            
            correlations.append(corr_val if corr_val is not None else 0.0)
        
        # Create target correlation matrix (1 row)
        mat = [correlations]
        correlation_cols = cols
        correlation_rows = [target]
        
    else:
        # Standard correlation matrix mode
        cols = _ensure_columns(df, columns)
        if len(cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation matrix")
        
        # Calculate correlation matrix
        df_numeric = df.select(cols)
        
        if method == "pearson":
            corr_df = df_numeric.corr()
        else:  # spearman
            # Rank all columns then correlate
            ranked_df = df_numeric.select([
                pl.col(c).rank().alias(c) for c in cols
            ])
            corr_df = ranked_df.corr()
        
        mat = corr_df.to_numpy().tolist()
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
        if backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(
                title=f'No correlations found with split="{split}" and threshold={threshold}',
                width=width, height=height
            )
            return fig
        elif backend == "altair":
            import altair as alt
            chart = alt.Chart(pl.DataFrame({"message": ["No correlations found"]})).mark_text(
                text=f'No correlations found with split="{split}" and threshold={threshold}'
            )
            if width: chart = chart.properties(width=width)
            if height: chart = chart.properties(height=height)
            return chart
    
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
        Number of histogram bins to use.
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
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(_BACKEND_ERROR)

    cols = _numeric_columns(df)
    if column is None:
        if not cols:
            if backend == "plotly":
                import plotly.graph_objects as go
                fig = go.Figure(); fig.update_layout(title="No numeric column found",
                                                     width=width, height=height); return fig
            else:
                import altair as alt
                chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
                if width:  chart = chart.properties(width=width)
                if height: chart = chart.properties(height=height)
                return chart
        column = cols[0]
    elif column not in df.columns:
        if backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure(); fig.update_layout(title=f"Column '{column}' not found",
                                                 width=width, height=height); return fig
        else:
            import altair as alt
            chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
            if width:  chart = chart.properties(width=width)
            if height: chart = chart.properties(height=height)
            return chart
    else:
        if not df[column].dtype.is_numeric():
            if backend == "plotly":
                import plotly.graph_objects as go
                fig = go.Figure(); fig.update_layout(title=f"Column '{column}' is not numeric",
                                                     width=width, height=height); return fig
            else:
                import altair as alt
                chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
                if width:  chart = chart.properties(width=width)
                if height: chart = chart.properties(height=height)
                return chart

    s = df.select(pl.col(column).drop_nulls()).to_series()
    if s.dtype not in (pl.Float32, pl.Float64):
        s = s.cast(pl.Float64)

    if backend == "plotly":
        return _dist_plot_plotly(s, column, bins, width, height)
    elif backend == "altair":
        return _dist_plot_altair(s, column, bins, width, height)
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
    """
    cols = list(df.columns)
    if sort not in {"desc", "asc", "none"}:
        raise ValueError("sort must be 'desc', 'asc', or 'none'")

    if backend not in _VALID_BACKENDS:
        raise ValueError(_BACKEND_ERROR)

    if not cols:
        if backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure(); fig.update_layout(title="No columns", width=width, height=height); return fig
        else:
            import altair as alt
            chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
            if width:  chart = chart.properties(width=width)
            if height: chart = chart.properties(height=height)
            return chart

    null_counts_row = df.select([pl.col(c).is_null().sum().alias(c) for c in cols])
    total = df.height
    counts = [int(null_counts_row.select(pl.col(c)).item()) for c in cols]
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
    """
    if top < 0 or bottom < 0:
        raise ValueError("top and bottom must be non-negative")

    if backend not in _VALID_BACKENDS:
        raise ValueError(_BACKEND_ERROR)

    # Get categorical columns (string/categorical types)
    cat_cols = [c for c, dt in zip(df.columns, df.dtypes) 
                if dt in (pl.String, pl.Utf8, pl.Categorical)]
    
    if not cat_cols:
        # No categorical columns found
        if backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(title="No categorical columns found", width=width, height=height)
            return fig
        else:
            import altair as alt
            chart = alt.Chart(pl.DataFrame({"values": []})).mark_bar()
            if width:  chart = chart.properties(width=width)
            if height: chart = chart.properties(height=height)
            return chart
    
    # Calculate value counts for each categorical column
    cat_data = {}
    for col in cat_cols:
        value_counts = (df.select(pl.col(col).value_counts(sort=True))
                       .unnest(col)
                      )

        counts = value_counts.get_column("count").to_list()
        values = value_counts.get_column(col).to_list()

        top_values = values[:top] if top > 0 else []
        top_counts = counts[:top] if top > 0 else []

        if bottom > 0:
            bottom_pairs = list(zip(values[-bottom:], counts[-bottom:]))
            top_value_keys = {repr(v) for v in top_values}
            bottom_pairs = [(v, c) for v, c in bottom_pairs if repr(v) not in top_value_keys]
            bottom_values = [v for v, _ in bottom_pairs]
            bottom_counts = [c for _, c in bottom_pairs]
        else:
            bottom_values = []
            bottom_counts = []
            
        cat_data[col] = {
            'top_values': top_values,
            'top_counts': top_counts,
            'bottom_values': bottom_values,
            'bottom_counts': bottom_counts
        }
    
    if backend == "plotly":
        return _cat_plot_plotly(cat_data, width, height)
    elif backend == "altair":
        return _cat_plot_altair(cat_data, width, height)
    else:
        raise ValueError(_BACKEND_ERROR)


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

        # Skip columns with nulls (keeps function behavior predictable).
        if series.null_count() > 0:
            continue

        if str_to_cat and dtype in (pl.String, pl.Utf8):
            unique_count = series.n_unique()
            total_count = len(series)
            if (
                unique_count <= max_cardinality
                and unique_count / total_count <= categorical_threshold
            ):
                transformations.append(pl.col(col).cast(pl.Categorical).alias(col))
            continue

        if downcast_ints and dtype.is_integer():
            min_val = series.min()
            max_val = series.max()

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
        Specific columns to consider for row dropping, or specific rows
        to consider for column dropping.

    Returns
    -------
    pl.DataFrame
        DataFrame with missing values dropped.
    """
    if thresh is not None and not 0.0 <= thresh <= 1.0:
        raise ValueError("thresh must be between 0 and 1")

    if subset:
        missing_subset = [c for c in subset if c not in df.columns]
        if missing_subset:
            raise ValueError(f"subset contains unknown columns: {missing_subset}")

    if axis == "rows":
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
                
    elif axis == "columns":
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
        
    else:
        raise ValueError("axis must be 'rows' or 'columns'")


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
        Whether to create interactive plots (plotly) or static (altair).
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
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(_BACKEND_ERROR)

    # Get numeric columns
    if columns is None:
        columns = _numeric_columns(df)
    else:
        columns = _ensure_columns(df, columns)
    
    if len(columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation plot")
    
    # Calculate correlation matrix
    df_numeric = df.select(columns)
    
    if method == "pearson":
        corr_df = df_numeric.corr()
    elif method == "spearman":
        ranked_df = df_numeric.select([
            pl.col(c).rank().alias(c) for c in columns
        ])
        corr_df = ranked_df.corr()
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    
    # Convert to matrix format
    corr_matrix = corr_df.to_numpy()
    
    # Clustering logic
    if clustered:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            import numpy as np
            
            # Replace NaN with 0 in correlation matrix before computing distances
            # NaN occurs when a column has no variance or all nulls
            corr_clean = np.nan_to_num(corr_matrix, nan=0.0)
            distance_matrix = 1 - np.abs(corr_clean)
            # Ensure diagonal is exactly 0 (avoid floating-point noise)
            np.fill_diagonal(distance_matrix, 0.0)
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method='average')
            cluster_order = leaves_list(linkage_matrix)
            
            corr_matrix = corr_matrix[cluster_order][:, cluster_order]
            columns = [columns[i] for i in cluster_order]
            
        except ImportError:
            print("scipy not installed, skipping clustering")
        except Exception as exc:
            print(f"Clustering failed ({exc.__class__.__name__}: {exc}), skipping clustering")
    
    # Choose appropriate backend based on interactivity preference
    if interactive and backend != "plotly":
        backend = "plotly"  # Force plotly for interactivity
    
    if backend == "plotly":
        return _corr_plot_plotly(columns, corr_matrix.tolist(), method, interactive, clustered, width, height)
    elif backend == "altair":
        return _corr_plot_altair(columns, corr_matrix.tolist(), method, width, height)
    else:
        raise ValueError(_BACKEND_ERROR)


def data_cleaning(
    df: pl.DataFrame,
    *,
    drop_missing_thresh: float = 0.9,
    optimize_dtypes: bool = True,
    remove_duplicates: bool = True,
    outlier_method: str | None = "iqr",
    outlier_threshold: float = 1.5,
    categorical_threshold: float = 0.5,
    max_cardinality: int = 50,
) -> pl.DataFrame:
    """
    Comprehensive data cleaning pipeline.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame to clean.
    drop_missing_thresh : float, default 0.9
        Threshold for dropping columns with missing values (keep if >= thresh).
    optimize_dtypes : bool, default True
        Whether to optimize data types for memory efficiency.
    remove_duplicates : bool, default True
        Whether to remove duplicate rows.
    outlier_method : str | None, default "iqr"
        Method for outlier detection ("iqr", "zscore", None).
    outlier_threshold : float, default 1.5
        Threshold for outlier detection (IQR multiplier or Z-score).
    categorical_threshold : float, default 0.5
        Threshold for converting strings to categorical.
    max_cardinality : int, default 50
        Maximum unique values for categorical conversion.

    Returns
    -------
    pl.DataFrame
        Cleaned DataFrame.
    """
    result = df.clone()
    
    print(f"Starting data cleaning: {result.shape}")
    
    # 1. Remove columns with too many missing values
    result = drop_missing(result, axis="columns", thresh=drop_missing_thresh)
    print(f"After dropping sparse columns: {result.shape}")
    
    # 2. Remove duplicate rows
    if remove_duplicates:
        before_rows = len(result)
        result = result.unique()
        after_rows = len(result)
        if before_rows != after_rows:
            print(f"Removed {before_rows - after_rows} duplicate rows")
    
    # 3. Handle outliers in numeric columns
    if outlier_method:
        numeric_cols = _numeric_columns(result)
        
        for col in numeric_cols:
            series = result[col]
            
            if outlier_method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
                
                result = result.with_columns(
                    pl.when((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
                
            elif outlier_method == "zscore":
                mean_val = series.mean()
                std_val = series.std()
                
                if std_val > 0:
                    result = result.with_columns(
                        pl.when(((pl.col(col) - mean_val) / std_val).abs() > outlier_threshold)
                        .then(None)
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
    
    # 4. Optimize data types
    if optimize_dtypes:
        result = convert_datatypes(
            result,
            categorical_threshold=categorical_threshold,
            max_cardinality=max_cardinality
        )
        print(f"Data types optimized")
    
    print(f"Data cleaning complete: {result.shape}")
    return result
