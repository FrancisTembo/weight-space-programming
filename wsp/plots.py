"""Tables and the two figures shown in the notebook."""
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from IPython.display import Markdown  # noqa: E402

CLEAN, EDITED, LINE = "#8a8f98", "#c2410c", "#1f2937"

Row: TypeAlias = dict[str, Any]
"""One result row, as produced by :mod:`wsp.detect` or :mod:`wsp.loss`."""


def _cell(value: Any, digits: int) -> str:
    """Format one table cell.

    Parameters
    ----------
    value : object
        The value to show.
    digits : int
        Decimal places for floats.

    Returns
    -------
    str
        A dash for None, the rounded number for a float, otherwise ``str(value)``.
    """
    if value is None:
        return "–"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: Sequence[Row], columns: Sequence[str] | None = None, digits: int = 3) -> Markdown:
    """Render rows as a Markdown table.

    Parameters
    ----------
    rows : sequence of dict
        One dict per table row.
    columns : sequence of str, optional
        Keys to show, in order. Defaults to the keys of the first row.
    digits : int, default 3
        Decimal places for floats.

    Returns
    -------
    IPython.display.Markdown
        The table. Its source text is in the ``data`` attribute.
    """
    columns = columns or list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(r.get(c), digits) for c in columns) + " |" for r in rows]
    return Markdown("\n".join(lines))


def screen_figure(clean_rows: Sequence[Row], edited_rows: Sequence[Row], threshold: float,
                  path: str | os.PathLike[str], title: str = "Spectral statistic per layer") -> Path:
    """Plot each layer's clean statistic, the statistic of edited layers, and the threshold.

    Parameters
    ----------
    clean_rows : sequence of dict
        Clean-model rows with ``layer`` and ``stat``, drawn as bars.
    edited_rows : sequence of dict
        Rows of edited layers with ``layer`` and ``stat``, drawn as dots.
    threshold : float
        Alarm threshold, drawn as a dashed line.
    path : str or path-like
        Where to save the PNG.
    title : str, default "Spectral statistic per layer"
        Figure title.

    Returns
    -------
    pathlib.Path
        ``path``. The vertical axis is logarithmic.
    """
    path = Path(path)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar([r["layer"] for r in clean_rows], [r["stat"] for r in clean_rows], color=CLEAN, label="clean layer")
    ax.scatter([r["layer"] for r in edited_rows], [r["stat"] for r in edited_rows], color=EDITED, zorder=3,
               s=45, label="same layer, edited")
    ax.axhline(threshold, color=LINE, linewidth=1, linestyle="--", label="alarm threshold")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("statistic")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xticks([r["layer"] for r in clean_rows])
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def sweep_figure(summary: Sequence[Row], path: str | os.PathLike[str]) -> Path:
    """Plot recall and mean precision against edit size, for both detectors.

    Parameters
    ----------
    summary : sequence of dict
        Rows from :func:`wsp.detect.summarise`, with ``detector``, ``size``, ``recall`` and
        ``mean_precision_at_64``.
    path : str or path-like
        Where to save the PNG.

    Returns
    -------
    pathlib.Path
        ``path``. The dashed and dotted lines mark the support (0.80) and challenge (0.50)
        bands.
    """
    path = Path(path)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, field, name in zip(axes, ("recall", "mean_precision_at_64"),
                               ("A. Screening: recall at the threshold", "B. Localisation: mean precision at 64")):
        for detector, colour in (("spectral", EDITED), ("balance", CLEAN)):
            rows = sorted((r for r in summary if r["detector"] == detector), key=lambda r: r["size"])
            ax.plot([r["size"] for r in rows], [r[field] for r in rows], marker="o", color=colour, label=detector)
        ax.axhline(0.80, color=LINE, linewidth=0.8, linestyle="--")
        ax.axhline(0.50, color=LINE, linewidth=0.8, linestyle=":")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("edit size (share of the layer's norm)")
        ax.set_title(name, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
