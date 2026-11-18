"""Two detectors that score a layer from its weights alone, and the rules used to judge them.

Neither score function is given the original weights or the true edited columns.
"""
import statistics
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from wsp import weights
from wsp.weights import Array, Mlps

TOP_VECTORS = 16        # singular vectors inspected by the spectral detector
N_NEURONS = 64          # neurons each detector names
SUPPORT = 0.80          # recall or precision at or above this supports a claim
CHALLENGE = 0.50        # below this challenges it
MAX_FALSE_ALARMS = 1
DETECTORS = ("spectral", "balance")

Row: TypeAlias = dict[str, Any]
"""One result row: plain Python values, ready to be written as JSON."""
CleanRows: TypeAlias = dict[str, list[Row]]
"""``{detector: screen(...) rows of the clean model}``."""


def spectral_score(down: Array) -> Row:
    """Score one layer by spectral localisation.

    A rank-one change confined to a few columns appears as a singular vector that sits on those
    columns. For each of the top :data:`TOP_VECTORS` right singular vectors ``v``, the inverse
    participation ratio ``IPR = sum(v**4)`` is ``1/n`` when ``v`` is spread over all ``n``
    neurons and ``1/m`` when it sits on ``m`` of them. The most localised vector is chosen.

    Parameters
    ----------
    down : ndarray, shape (d, n)
        The layer's ``down_proj`` matrix, one column per neuron.

    Returns
    -------
    dict
        ``stat`` : float
            The largest ``IPR * n`` among the top singular vectors.
        ``neurons`` : list of int
            Sorted indices of the :data:`N_NEURONS` largest absolute entries of the chosen
            vector.
        ``direction`` : list of float
            The chosen vector's left singular vector: the direction written to the output.
        ``vector_index`` : int
            Which singular vector was chosen (0 is the largest singular value).
        ``effective_neurons`` : float
            ``1 / IPR``: roughly how many neurons the chosen vector sits on.
    """
    U, _, Vt = np.linalg.svd(down, full_matrices=False)
    ipr = (Vt[:TOP_VECTORS] ** 4).sum(1)
    k = int(ipr.argmax())
    top = np.argsort(-np.abs(Vt[k]))[:N_NEURONS]
    return {
        "stat": float(ipr[k] * down.shape[1]),
        "neurons": sorted(int(i) for i in top),
        "direction": [float(x) for x in U[:, k]],
        "vector_index": k,
        "effective_neurons": float(1 / ipr[k]),     # how many neurons the chosen vector sits on
    }


def _balance_z(down: Array, gate: Array, up: Array) -> npt.NDArray[np.floating[Any]]:
    """Robust z-score of each neuron's departure from the layer's norm-balance relation.

    In trained weights a neuron's output norm tracks its input norms. The log output norm is
    fitted on the two log input norms by trimmed least squares (three rounds, each refitting
    without the worst 5% of residuals). An edit to the output column alone moves the neuron off
    that relation.

    Parameters
    ----------
    down : ndarray, shape (d, n)
        ``down_proj``, one column per neuron.
    gate : ndarray, shape (n, d)
        ``gate_proj``, one row per neuron.
    up : ndarray, shape (n, d)
        ``up_proj``, one row per neuron.

    Returns
    -------
    ndarray, shape (n,)
        Residual of each neuron, centred on the median and scaled by the median absolute
        deviation.
    """
    o = np.log(np.linalg.norm(down, axis=0))
    X = np.stack([np.log(np.linalg.norm(gate, axis=1)), np.log(np.linalg.norm(up, axis=1)),
                  np.ones(down.shape[1], dtype=o.dtype)], 1)
    keep = np.ones(len(o), bool)
    for _ in range(3):                                   # refit without the worst 5%
        coef, *_ = np.linalg.lstsq(X[keep], o[keep], rcond=None)
        r = o - X @ coef
        keep = np.abs(r) <= np.quantile(np.abs(r), 0.95)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826 + 1e-12
    return (r - med) / mad


def balance_score(down: Array, gate: Array, up: Array) -> Row:
    """Score one layer by norm balance, the alternative detector.

    Parameters
    ----------
    down : ndarray, shape (d, n)
        ``down_proj``, one column per neuron.
    gate : ndarray, shape (n, d)
        ``gate_proj``, one row per neuron.
    up : ndarray, shape (n, d)
        ``up_proj``, one row per neuron.

    Returns
    -------
    dict
        ``stat`` : float
            Mean of the :data:`N_NEURONS` largest absolute residual z-scores.
        ``neurons`` : list of int
            Sorted indices of those neurons.
    """
    z = np.abs(_balance_z(down, gate, up))
    top = np.argsort(-z)[:N_NEURONS]
    return {"stat": float(z[top].mean()), "neurons": sorted(int(i) for i in top)}


def _score(mlps: Mlps, layer: int, detector: str) -> Row:
    """Run one detector on one layer of a checkpoint.

    Parameters
    ----------
    mlps : Mlps
        The checkpoint's MLP weights.
    layer : int
        Layer to score.
    detector : {"spectral", "balance"}
        Which detector to run.

    Returns
    -------
    dict
        The detector's score, as returned by :func:`spectral_score` or :func:`balance_score`.

    Raises
    ------
    ValueError
        If ``detector`` is not a known name.
    """
    if detector == "spectral":
        return spectral_score(mlps[(layer, "down_proj")])
    if detector == "balance":
        return balance_score(*(mlps[(layer, p)] for p in weights.PARTS))
    raise ValueError(f"unknown detector {detector!r}")


def screen(mlps: Mlps, detector: str, layers: Iterable[int] = range(weights.N_LAYERS)) -> list[Row]:
    """Score several layers of a checkpoint with one detector.

    Parameters
    ----------
    mlps : Mlps
        The checkpoint's MLP weights.
    detector : {"spectral", "balance"}
        Which detector to run.
    layers : iterable of int, default all 24 layers
        Layers to score.

    Returns
    -------
    list of dict
        One row per layer, in the order given: the detector's score plus ``layer``.
    """
    return [{"layer": int(L), **_score(mlps, L, detector)} for L in layers]


def calibrate(clean_rows: list[Row], dev_layers: Iterable[int]) -> float:
    """Set the alarm threshold from clean development layers.

    Parameters
    ----------
    clean_rows : list of dict
        :func:`screen` rows of the clean model for one detector.
    dev_layers : iterable of int
        The development layers. Rows of other layers are ignored.

    Returns
    -------
    float
        The highest ``stat`` among the development layers. A layer alarms when its statistic is
        strictly greater than this.
    """
    return max(r["stat"] for r in clean_rows if r["layer"] in dev_layers)


def precision_at_64(neurons: Sequence[int], true_columns: Iterable[int]) -> float:
    """Share of the named neurons that were truly edited.

    Parameters
    ----------
    neurons : sequence of int
        Neurons a detector named (64 of them for the detectors here).
    true_columns : iterable of int
        Neurons that were truly edited.

    Returns
    -------
    float
        Between 0 and 1.
    """
    return len(set(neurons) & set(int(c) for c in true_columns)) / len(neurons)


def direction_cosine(direction: Sequence[float], true_direction: Sequence[float]) -> float:
    """Absolute cosine between a recovered direction and the true one.

    Parameters
    ----------
    direction : sequence of float
        Direction reported by the spectral detector.
    true_direction : sequence of float
        Direction of the actual edit.

    Returns
    -------
    float
        Between 0 and 1. Sign and length are ignored.
    """
    a, b = np.asarray(direction, dtype=np.float64), np.asarray(true_direction, dtype=np.float64)
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve.

    Parameters
    ----------
    scores : sequence of float
        One score per item; higher means more suspicious.
    labels : sequence of bool
        True for positives (edited), False for negatives (clean).

    Returns
    -------
    float
        Probability that a random positive scores above a random negative; ties count half.

    Raises
    ------
    ValueError
        If there is no positive or no negative.
    """
    s, positive = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    pos, neg = int(positive.sum()), int((~positive).sum())
    if pos == 0 or neg == 0:
        raise ValueError("auroc needs at least one positive and one negative")
    greater = (s[positive][:, None] > s[~positive][None, :]).sum()
    equal = (s[positive][:, None] == s[~positive][None, :]).sum()
    return float((greater + 0.5 * equal) / (pos * neg))


def rank_in_checkpoint(clean_rows: list[Row], edited_rows: list[Row]) -> dict[int, int]:
    """Rank each edited layer among all layers of its checkpoint.

    Parameters
    ----------
    clean_rows : list of dict
        :func:`screen` rows of the clean model for one detector, all layers.
    edited_rows : list of dict
        Rows of the edited layers. They replace the clean rows of the same layers.

    Returns
    -------
    dict of int to int
        ``{layer: rank}`` for each edited layer, where 1 is the most suspicious. Ties share the
        better rank (1, 1, 3).
    """
    stats = {r["layer"]: r["stat"] for r in clean_rows}
    stats.update({r["layer"]: r["stat"] for r in edited_rows})
    return {r["layer"]: 1 + sum(v > r["stat"] for L, v in stats.items() if L != r["layer"])
            for r in edited_rows}


def sweep(mlps: Mlps, clean_rows: CleanRows, layers: Iterable[int], sizes: Iterable[float],
          kinds: Sequence[str], seeds: Iterable[int]) -> list[Row]:
    """Score synthetic edits, one layer at a time, with both detectors.

    Each edit is generated once, with the seed
    ``(seed, layer, round(size * 1000), kinds.index(kind))``, and scored by both detectors. The
    true direction is worked out only after the detectors have returned.

    Parameters
    ----------
    mlps : Mlps
        Clean MLP weights.
    clean_rows : CleanRows
        Clean-model rows per detector, used to rank the edited layer.
    layers : iterable of int
        Layers to edit, one per synthetic checkpoint.
    sizes : iterable of float
        Edit sizes, as shares of the layer's Frobenius norm.
    kinds : sequence of str
        Direction kinds, each ``"svd"`` or ``"random"``.
    seeds : iterable of int
        Seeds; each gives a different edit per layer, size and kind.

    Returns
    -------
    list of dict
        One row per edit and detector, with ``detector``, ``layer``, ``size``, ``kind``,
        ``seed``, ``stat``, ``rank``, ``precision_at_64`` and ``direction_cosine`` (None for
        the balance detector).
    """
    rows = []
    for layer in layers:
        down = mlps[(layer, "down_proj")]
        for size in sizes:
            for kind in kinds:
                for seed in seeds:
                    edited, cols = weights.synthetic_edit(
                        down, (seed, layer, round(size * 1000), kinds.index(kind)), size, kind)
                    one = weights.apply_edits(mlps, {layer: edited})
                    scores = {d: _score(one, layer, d) for d in DETECTORS}
                    true_direction = weights.describe_edit(down, edited)["direction"]   # after scoring
                    for d, score in scores.items():
                        rows.append({
                            "detector": d, "layer": int(layer), "size": size, "kind": kind, "seed": seed,
                            "stat": score["stat"],
                            "rank": rank_in_checkpoint(clean_rows[d], [{"layer": layer, **score}])[layer],
                            "precision_at_64": precision_at_64(score["neurons"], cols),
                            "direction_cosine": (direction_cosine(score["direction"], true_direction)
                                                 if d == "spectral" else None),
                        })
    return rows


def summarise(sweep_rows: list[Row], clean_rows: CleanRows, thresholds: dict[str, float],
              dev_layers: Iterable[int]) -> list[Row]:
    """Summarise a sweep per detector and edit size.

    Parameters
    ----------
    sweep_rows : list of dict
        Rows from :func:`sweep`.
    clean_rows : CleanRows
        Clean-model rows per detector.
    thresholds : dict of str to float
        Alarm threshold per detector, from :func:`calibrate`.
    dev_layers : iterable of int
        Development layers. Clean layers outside this set are the negatives.

    Returns
    -------
    list of dict
        One row per detector and size, with ``n``, ``threshold``, ``recall`` (share of edits
        whose statistic is strictly above the threshold), ``mean_precision_at_64``,
        ``median_rank``, ``auroc_vs_clean`` (edited-layer statistics against clean layers
        outside development) and ``false_alarms`` (how many of those clean layers alarm).
    """
    out = []
    for d in sorted({r["detector"] for r in sweep_rows}):
        clean = [r["stat"] for r in clean_rows[d] if r["layer"] not in dev_layers]
        false_alarms = sum(s > thresholds[d] for s in clean)
        for size in sorted({r["size"] for r in sweep_rows if r["detector"] == d}):
            rows = [r for r in sweep_rows if r["detector"] == d and r["size"] == size]
            stats = [r["stat"] for r in rows]
            out.append({
                "detector": d, "size": size, "n": len(rows), "threshold": thresholds[d],
                "recall": sum(s > thresholds[d] for s in stats) / len(rows),
                "mean_precision_at_64": sum(r["precision_at_64"] for r in rows) / len(rows),
                "median_rank": float(statistics.median(r["rank"] for r in rows)),
                "auroc_vs_clean": auroc(stats + clean, [True] * len(stats) + [False] * len(clean)),
                "false_alarms": int(false_alarms),
            })
    return out


def _band(value: float) -> str:
    """Place a recall or precision in its decision band.

    Parameters
    ----------
    value : float
        A recall or a mean precision.

    Returns
    -------
    str
        ``"supported"`` at or above :data:`SUPPORT`, ``"challenged"`` below :data:`CHALLENGE`,
        otherwise ``"inconclusive"``.
    """
    return "supported" if value >= SUPPORT else "challenged" if value < CHALLENGE else "inconclusive"


def verdict(summary_row: Row) -> dict[str, str]:
    """Judge the two claims by the rules fixed before the held-out run.

    Parameters
    ----------
    summary_row : dict
        One row from :func:`summarise`, at the decision size.

    Returns
    -------
    dict of str to str
        ``screening``: judged on ``recall``, and challenged whenever ``false_alarms`` exceeds
        :data:`MAX_FALSE_ALARMS`. ``localisation``: judged on ``mean_precision_at_64``. Each is
        ``"supported"``, ``"challenged"`` or ``"inconclusive"``.
    """
    screening = _band(summary_row["recall"])
    if summary_row["false_alarms"] > MAX_FALSE_ALARMS:
        screening = "challenged"
    return {"screening": screening, "localisation": _band(summary_row["mean_precision_at_64"])}


def score_checkpoint(mlps: Mlps, clean_rows: CleanRows, thresholds: dict[str, float],
                     edits: dict[int, Array]) -> list[Row]:
    """Score one edited checkpoint with both detectors.

    The true columns and direction are worked out from the clean and edited arrays only after
    every detector has returned.

    Parameters
    ----------
    mlps : Mlps
        Clean MLP weights.
    clean_rows : CleanRows
        Clean-model rows per detector, all layers.
    thresholds : dict of str to float
        Alarm threshold per detector.
    edits : dict of int to ndarray
        ``{layer: edited down_proj}``: the checkpoint is ``mlps`` with these applied.

    Returns
    -------
    list of dict
        One row per detector and edited layer, with ``detector``, ``layer``, ``stat``,
        ``clean_stat``, ``threshold``, ``alarm``, ``rank``, ``precision_at_64``,
        ``direction_cosine`` (None for the balance detector), ``effective_neurons`` (None for
        the balance detector) and ``neurons``.
    """
    checkpoint = weights.apply_edits(mlps, edits)
    scored = {d: screen(checkpoint, d, layers=list(edits)) for d in DETECTORS}
    truth = {L: weights.describe_edit(mlps[(L, "down_proj")], edits[L]) for L in edits}
    rows = []
    for d in DETECTORS:
        ranks = rank_in_checkpoint(clean_rows[d], scored[d])
        clean = {r["layer"]: r["stat"] for r in clean_rows[d]}
        for r in scored[d]:
            L = r["layer"]
            rows.append({
                "detector": d, "layer": L, "stat": r["stat"], "clean_stat": clean[L],
                "threshold": thresholds[d], "alarm": bool(r["stat"] > thresholds[d]), "rank": ranks[L],
                "precision_at_64": precision_at_64(r["neurons"], truth[L]["columns"]),
                "direction_cosine": (direction_cosine(r["direction"], truth[L]["direction"])
                                     if d == "spectral" else None),
                "effective_neurons": r.get("effective_neurons"),
                "neurons": r["neurons"],
            })
    return rows


def summarise_by_layer(sweep_rows: list[Row], clean_rows: CleanRows, detector: str,
                       size: float) -> list[Row]:
    """Average sweep results per layer, for one detector and edit size.

    Parameters
    ----------
    sweep_rows : list of dict
        Rows from :func:`sweep`.
    clean_rows : CleanRows
        Clean-model rows per detector.
    detector : {"spectral", "balance"}
        Detector to report.
    size : float
        Edit size to report.

    Returns
    -------
    list of dict
        One row per layer that has matching sweep rows, sorted by layer, with ``layer``,
        ``clean_stat``, ``n``, ``mean_stat``, ``mean_precision_at_64`` and ``median_rank``.
    """
    clean = {r["layer"]: r["stat"] for r in clean_rows[detector]}
    out = []
    for layer in sorted({r["layer"] for r in sweep_rows}):
        rows = [r for r in sweep_rows if (r["detector"], r["size"], r["layer"]) == (detector, size, layer)]
        if rows:
            out.append({
                "layer": layer, "clean_stat": clean[layer], "n": len(rows),
                "mean_stat": sum(r["stat"] for r in rows) / len(rows),
                "mean_precision_at_64": sum(r["precision_at_64"] for r in rows) / len(rows),
                "median_rank": float(statistics.median(r["rank"] for r in rows)),
            })
    return out


def compare_neurons(neurons: Iterable[int], true_columns: Iterable[int]) -> Row:
    """Compare the neurons a detector named with the truly edited ones.

    Parameters
    ----------
    neurons : iterable of int
        Neurons the detector named.
    true_columns : iterable of int
        Neurons that were truly edited.

    Returns
    -------
    dict
        ``recovered`` : int
            How many named neurons were truly edited.
        ``missed`` : list of int
            Edited neurons the detector did not name.
        ``wrongly_named`` : list of int
            Named neurons that were not edited.
    """
    named, true = set(neurons), set(int(c) for c in true_columns)
    return {"recovered": len(named & true), "missed": sorted(true - named), "wrongly_named": sorted(named - true)}


def shortlist(clean_rows: list[Row], edited_rows: list[Row], n: int = 5) -> list[Row]:
    """List the most suspicious layers of a checkpoint.

    Parameters
    ----------
    clean_rows : list of dict
        :func:`screen` rows of the clean model for one detector, all layers.
    edited_rows : list of dict
        Rows of the edited layers. They replace the clean rows of the same layers.
    n : int, default 5
        How many layers to return.

    Returns
    -------
    list of dict
        The ``n`` layers with the highest statistic, each with ``rank``, ``layer``, ``stat``,
        ``edited`` and ``effective_neurons``. Ranks follow :func:`rank_in_checkpoint`: ties
        share the better rank.
    """
    edited = {r["layer"]: r for r in edited_rows}
    rows = sorted((edited.get(r["layer"], r) for r in clean_rows), key=lambda r: -r["stat"])
    return [{"rank": 1 + sum(o["stat"] > r["stat"] for o in rows),       # ties share the better rank
             "layer": r["layer"], "stat": r["stat"], "edited": r["layer"] in edited,
             "effective_neurons": r.get("effective_neurons")} for r in rows[:n]]
