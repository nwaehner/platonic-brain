"""
_style.py — the exact serif / black 16×12 plot theme for the post-hoc interp studies.

Import this AFTER `_common` so it overrides `_common`'s 14×9 theme. `_interp_common`
imports it last, so any script that does `import _interp_common` inherits this theme
automatically. Call `apply_style()` again before a figure if some other module reset
rcParams in between.
"""

import matplotlib.pyplot as plt

COLOR = "black"

THEME = {
    "figure.dpi": 120,
    "figure.figsize": (16, 12),
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "legend.fontsize": "medium",
    "legend.title_fontsize": 14,
    "axes.titlesize": 14,
    "axes.labelsize": "large",
    "ytick.labelsize": 10,
    "xtick.labelsize": 10,
    "text.color": COLOR,
    "axes.labelcolor": COLOR,
    "xtick.color": COLOR,
    "ytick.color": COLOR,
    "grid.color": COLOR,
}


def apply_style():
    """(Re)apply the requested interp theme to the global matplotlib rcParams."""
    plt.rcParams.update(THEME)


apply_style()
