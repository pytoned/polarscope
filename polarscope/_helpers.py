"""Small helpers shared by the public modules."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def require_dataframe(df: object, func: str) -> None:
    """Raise a clear TypeError unless ``df`` is an eager Polars DataFrame."""
    if isinstance(df, pl.LazyFrame):
        raise TypeError(
            f"{func}() needs a DataFrame; call .collect() on the LazyFrame first "
            "(of the polarscope functions, only xray() accepts LazyFrames)."
        )
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"{func}() expects a Polars DataFrame, got {type(df).__name__}")


def is_pattern_name(name: str) -> bool:
    """True for column names pl.col() does not take literally.

    ``pl.col("*")`` selects every column and ``pl.col("^...$")`` is a regular
    expression.
    """
    return name == "*" or (name.startswith("^") and name.endswith("$"))


def literal_name_map(columns: Sequence[str]) -> dict[str, str]:
    """Safe stand-ins for pattern-like column names (see ``is_pattern_name``).

    Rename the frame with this mapping before building expressions from user
    column names, and map back for display. ``pl.nth()`` is no alternative:
    Polars 1.25 resolves it to the name and then expands that name as a
    pattern again.
    """
    taken = set(columns)
    mapping: dict[str, str] = {}
    for i, name in enumerate(columns):
        if is_pattern_name(name):
            safe = f"__ps_col_{i}"
            while safe in taken:
                safe += "_"
            taken.add(safe)
            mapping[name] = safe
    return mapping
