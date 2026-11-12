
from __future__ import annotations

import os
from pathlib import Path


def save_fig(obj, path: str | os.PathLike[str], *, scale: float = 1.0, **kwargs):
    """
    Save a figure/chart to disk.

    - Plotly Figure     -> .write_html(path) for ".html", otherwise
                           .write_image(path, scale=scale) (static images
                           require kaleido)
    - Altair Chart      -> .save(path)
    - Matplotlib Figure -> .savefig(path, dpi=300, bbox_inches="tight")  (if encountered)

    Plotly HTML files load plotly.js from a CDN by default, so they are small
    but need an internet connection to display. Pass
    ``include_plotlyjs=True`` to embed plotly.js (several MB) and get a
    self-contained file. Altair's HTML output likewise loads Vega from a CDN
    unless you pass ``inline=True`` (requires the ``vl-convert-python``
    package).

    Parameters
    ----------
    obj : Plotly Figure | Altair Chart | Matplotlib Figure
    path : str | os.PathLike
        Output filepath with extension (.png, .pdf, .svg, .html, etc.)
    scale : float
        Scaling factor for Plotly static images.
    **kwargs
        Passed on to the backend's save method (``write_html``,
        ``write_image``, ``save`` or ``savefig``), e.g.
        ``save_fig(fig, "plot.html", include_plotlyjs=True)``.
    """
    t = str(type(obj))
    path = os.fspath(path)
    ext = Path(path).suffix.lstrip(".").lower()
    if hasattr(obj, "savefig"):  # Matplotlib
        obj.savefig(path, **{"dpi": 300, "bbox_inches": "tight", **kwargs})
        return
    if "plotly" in t:
        # HTML export doesn't require kaleido; static image does
        if ext == "html":
            kwargs.setdefault("include_plotlyjs", "cdn")
            obj.write_html(path, **kwargs)
        else:
            obj.write_image(path, scale=scale, **kwargs)
        return
    # assume Altair
    obj.save(path, **kwargs)
