
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
