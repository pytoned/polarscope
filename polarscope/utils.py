
from __future__ import annotations

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
    t = str(type(obj))
    ext = path.split(".")[-1].lower()
    if hasattr(obj, "savefig"):  # Matplotlib
        obj.savefig(path, dpi=300, bbox_inches="tight")
        return
    if "plotly" in t:
        # HTML export doesn't require kaleido; static image does
        if ext == "html":
            obj.write_html(path, include_plotlyjs="cdn")
        else:
            obj.write_image(path, scale=scale)
        return
    # assume Altair
    obj.save(path)
