"""Shared publication typography, with a TeX-free option for compute nodes.

The values intentionally match ``PAPER_RC`` in
``analyses/create_four_panel_reproducibility_figure.py``.  Keeping other figure
generators behind this context avoids leaking rcParams into training or analysis.
RIBOUNMIX_PLOT_TEX=auto (default) uses TeX when latex and dvipng are available;
0 uses Matplotlib's bundled serif/math fonts; 1 explicitly requires TeX.
The fallback is not claimed to be identical to the publication's TeX glyphs.
"""

from __future__ import annotations

from functools import wraps
import os
import shutil
from typing import Any, Callable, TypeVar, cast
import warnings


LATEX_PAPER_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
    "font.size": 12.0,
    "axes.labelsize": 12.0,
    "axes.titlesize": 12.0,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 12.0,
    "legend.title_fontsize": 12.0,
    "legend.frameon": False,
    "grid.color": "#D8D8D8",
    "grid.linewidth": 0.45,
    "grid.alpha": 0.70,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "path.simplify": False,
    "savefig.dpi": 300,
}


_Function = TypeVar("_Function", bound=Callable[..., Any])


def publication_rc() -> dict[str, Any]:
    """Choose rendering only; numerical results and experiment settings are untouched."""
    mode = os.environ.get(
        "RIBOUNMIX_PLOT_TEX",
        os.environ.get("RIBOAI_PLOT_TEX", "auto"),
    ).strip().lower()
    if mode not in {"auto", "0", "1"}:
        raise ValueError("RIBOUNMIX_PLOT_TEX must be auto, 0 (built-in fonts), or 1 (require TeX).")
    missing = [] if mode == "0" else [
        name for name in ("latex", "dvipng") if shutil.which(name) is None
    ]
    if missing and mode == "1":
        raise RuntimeError(
            f"TeX plotting was requested but executables are missing: {', '.join(missing)}. "
            "Set RIBOUNMIX_PLOT_TEX=0 to render without external TeX."
        )
    use_tex = mode != "0" and not missing
    if missing:
        warnings.warn(
            f"Missing {', '.join(missing)}; using Matplotlib DejaVu Serif and built-in "
            "math text for plots. Numerical results are unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
    style = dict(LATEX_PAPER_RC)
    if not use_tex:
        style.update({
            "text.usetex": False,
            "text.latex.preamble": "",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
        })
    return style


def latex_paper_style(function: _Function) -> _Function:
    """Run one plotting function inside the canonical publication rc context."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        import matplotlib

        with matplotlib.rc_context(publication_rc()):
            return function(*args, **kwargs)

    return cast(_Function, wrapped)
