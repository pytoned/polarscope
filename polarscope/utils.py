
from __future__ import annotations

from typing import Any

import polars as pl


def as_dataframe(data: Any, *, name: str = "df") -> pl.DataFrame:
    """Accept a DataFrame or LazyFrame; reject everything else with a clear error.

    LazyFrames are collected so public helpers can share one eager code path.
    Pandas objects are rejected on purpose: this package never converts to or
    from pandas.
    """
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    raise TypeError(
        f"{name} must be a Polars DataFrame or LazyFrame, got {type(data).__name__}. "
        "Pandas objects are not supported; convert with pl.from_pandas() first."
    )


def save_fig(obj, path: str, *, scale: float = 1.0):
    """
    Save a figure/chart to disk.

    - Plotly Figure     -> .write_image(path, scale=scale)  (requires kaleido installed for static images)
    - Altair Chart      -> .save(path)
    - Matplotlib Figure -> .savefig(path, dpi=300)  (if encountered)

    Parameters
    ----------
    obj : Plotly Figure | Altair Chart | Matplotlib Figure
    path : str
        Output filepath with extension (.png, .pdf, .svg, .html, etc.)
    scale : float
        Scaling factor for Plotly static images.
    """
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if hasattr(obj, "savefig"):  # Matplotlib
        obj.savefig(path, dpi=300, bbox_inches="tight")
        return
    if hasattr(obj, "write_html") and hasattr(obj, "write_image"):
        # HTML export doesn't require kaleido; static image does
        if ext == "html":
            obj.write_html(path, include_plotlyjs="cdn")
        else:
            obj.write_image(path, scale=scale)
        return
    if hasattr(obj, "save"):
        obj.save(path)
        return
    raise TypeError(f"Don't know how to save object of type {type(obj).__name__}")
