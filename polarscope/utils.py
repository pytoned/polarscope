
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


def fig_to_base64_png(fig) -> str:
    """
    Convert a Matplotlib Figure to a base64 PNG data URL string.
    Safe no-op for non-matplotlib objects (raises ValueError).
    """
    if not hasattr(fig, "savefig"):
        raise ValueError("fig_to_base64_png expects a Matplotlib Figure")
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"
