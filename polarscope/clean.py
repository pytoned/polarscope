
from __future__ import annotations
import re
import unicodedata
import polars as pl

def _normalize(s: str, case: str = "lower", ascii_only: bool = True) -> str:
    s = s.strip()
    if ascii_only:
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"[\s\-]+", "_", s)
    if case == "lower":
        s = s.lower()
    elif case == "upper":
        s = s.upper()
    return s

def clean_column_names(
    df: pl.DataFrame,
    *,
    case: str = "lower",
    ascii_only: bool = True,
    dedupe: bool = True,
) -> pl.DataFrame:
    old = list(df.columns)
    seen: dict[str, int] = {}
    new: list[str] = []
    for name in old:
        base = _normalize(str(name), case=case, ascii_only=ascii_only)
        if not base:
            base = "col"
        if dedupe:
            if base not in seen:
                seen[base] = 0
                new_name = base
            else:
                seen[base] += 1
                new_name = f"{base}_{seen[base]}"
        else:
            new_name = base
        new.append(new_name)
    mapping = dict(zip(old, new))
    return df.rename(mapping)
