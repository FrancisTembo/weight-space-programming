"""Teacher-forced next-token loss on bundled Swedish and English passages, for the base model
and for edited versions of it.

Runs on CPU in float32. No text is generated.
"""
import functools
import hashlib
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch

from wsp import io, weights
from wsp.weights import Array

PASSAGES = weights.ROOT / "data" / "passages.jsonl"
RANDOM_SEEDS = (0, 1, 2)
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 0
RESERVED_NAMES = {"base", "id", "lang", "doc", "tokens"}     # row columns an edit name must not overwrite

Passage: TypeAlias = dict[str, str]
"""One passage: ``id``, ``lang``, ``doc``, ``split`` and ``text``."""
Row: TypeAlias = dict[str, Any]
"""One scored passage: ``id``, ``lang``, ``doc``, ``tokens``, ``base`` and one loss per edit name."""
Edits: TypeAlias = dict[str, dict[int, Array]]
"""``{edit name: {layer: edited down_proj}}``."""


def load_passages(split: str | None = None) -> list[Passage]:
    """Load the bundled passages.

    Parameters
    ----------
    split : {"dev", "test"}, optional
        Keep only this split. By default all 136 passages are returned.

    Returns
    -------
    list of dict
        Passages with ``id``, ``lang`` (``"sv"`` or ``"en"``), ``doc``, ``split`` and ``text``,
        in file order.
    """
    rows = [json.loads(line) for line in PASSAGES.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if split is None or r["split"] == split]


@functools.lru_cache(maxsize=1)
def load_model() -> tuple[Any, torch.nn.Module]:
    """Load the pinned base checkpoint on CPU in float32.

    The result is cached, so every caller shares one model object.

    Returns
    -------
    tokenizer : transformers tokenizer
        The checkpoint's tokenizer.
    model : torch.nn.Module
        The model, in evaluation mode.

    Raises
    ------
    FileNotFoundError
        If the base weights are not in the Hugging Face cache.
    """
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    d = weights.base_dir()
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.float32).eval()
    return tok, model


def down_proj(model: torch.nn.Module, layer: int) -> torch.nn.Parameter:
    """Return the live MLP output weight of one layer.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model.
    layer : int
        Layer index, 0 to 23.

    Returns
    -------
    torch.nn.Parameter
        The ``down_proj`` weight. Writing to it changes the model.

    Raises
    ------
    KeyError
        If the model has no such parameter.
    """
    params = dict(model.named_parameters())
    name = weights.tensor_name(layer)
    for candidate in (name, name.removeprefix("model.")):
        if candidate in params:
            return params[candidate]
    raise KeyError(name)


@torch.no_grad()
def _passage_nll(model: torch.nn.Module, ids: torch.Tensor) -> float:
    """Mean next-token loss of one passage.

    Parameters
    ----------
    model : torch.nn.Module
        The model to score with.
    ids : torch.Tensor, shape (1, t)
        Token ids of the passage.

    Returns
    -------
    float
        Mean negative log-likelihood, in nats per token, of tokens 2 to ``t`` given the tokens
        before them.
    """
    logits = model(input_ids=ids).logits[0, :-1].float()
    return torch.nn.functional.cross_entropy(logits, ids[0, 1:]).item()


@torch.no_grad()
def score_passages(passages: list[Passage], edits: Edits) -> list[Row]:
    """Score every passage under the base weights and under each edit.

    Parameters
    ----------
    passages : list of dict
        Passages from :func:`load_passages`.
    edits : Edits
        ``{edit name: {layer: edited down_proj}}``. Each edit is applied on its own, to the base
        weights.

    Returns
    -------
    list of dict
        One row per passage, in order, with ``id``, ``lang``, ``doc``, ``tokens``, the loss
        under ``base`` and the loss under each edit name, in nats per token.

    Raises
    ------
    ValueError
        If an edit name is one of :data:`RESERVED_NAMES`, which would overwrite a row column.

    Notes
    -----
    The model is restored to the base weights before returning, including when scoring raises.
    """
    reserved = sorted(set(edits) & RESERVED_NAMES)
    if reserved:
        raise ValueError(f"reserved condition names: {reserved}")
    tok, model = load_model()
    layers = sorted({L for e in edits.values() for L in e})
    base = {L: down_proj(model, L).detach().clone() for L in layers}
    ids = [tok(p["text"], return_tensors="pt", add_special_tokens=False).input_ids for p in passages]
    rows = [{"id": p["id"], "lang": p["lang"], "doc": p["doc"], "tokens": int(x.shape[1])}
            for p, x in zip(passages, ids)]
    try:
        for name, edit in {"base": {}, **edits}.items():
            for L in layers:
                target = torch.from_numpy(edit[L]) if L in edit else base[L]
                down_proj(model, L).copy_(target)
            for row, x in zip(rows, ids):
                row[name] = _passage_nll(model, x)
    finally:
        for L in layers:
            down_proj(model, L).copy_(base[L])
    return rows


def paired_change(rows: list[Row], condition: str, lang: str) -> dict[str, float | int]:
    """Mean change in loss from base, for one edit and one language.

    Parameters
    ----------
    rows : list of dict
        Rows from :func:`score_passages`.
    condition : str
        Edit name to compare with ``base``.
    lang : {"sv", "en"}
        Language of the passages to include.

    Returns
    -------
    dict
        ``mean`` : float
            Mean of the per-passage loss differences, in nats per token.
        ``ci_low``, ``ci_high`` : float
            95% interval from :data:`BOOTSTRAPS` resamples of whole documents.
        ``n`` : int
            Number of passages.
        ``worse`` : int
            Number of passages whose loss rose.
    """
    sub = [r for r in rows if r["lang"] == lang]
    diffs = np.array([r[condition] - r["base"] for r in sub])
    by_doc: dict[str, list[float]] = {}
    for r, d in zip(sub, diffs):
        by_doc.setdefault(r["doc"], []).append(d)
    groups = [np.array(g) for g in by_doc.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return {
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": len(sub),
        "worse": int((diffs > 0).sum()),
    }


def worst_passages(rows: list[Row], passages: list[Passage], condition: str, lang: str,
                   n: int = 2) -> list[Row]:
    """Find the passages whose loss rose most under an edit.

    Parameters
    ----------
    rows : list of dict
        Rows from :func:`score_passages`.
    passages : list of dict
        The passages that were scored, for their text.
    condition : str
        Edit name to compare with ``base``.
    lang : {"sv", "en"}
        Language of the passages to include.
    n : int, default 2
        How many passages to return.

    Returns
    -------
    list of dict
        The ``n`` passages with the largest rise, each with ``id``, ``change`` (nats per token)
        and ``text`` (the first 300 characters).
    """
    text = {p["id"]: p["text"] for p in passages}
    sub = sorted((r for r in rows if r["lang"] == lang), key=lambda r: r["base"] - r[condition])
    return [{"id": r["id"], "change": r[condition] - r["base"], "text": text[r["id"]][:300]} for r in sub[:n]]


def _sha256(data: bytes) -> str:
    """Hex sha256 of some bytes.

    Parameters
    ----------
    data : bytes
        The bytes to hash.

    Returns
    -------
    str
        The hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def loss_key(split: str, edits: Edits) -> dict[str, Any]:
    """Describe everything a cached loss result depends on.

    Parameters
    ----------
    split : str
        The passage split being scored.
    edits : Edits
        The edits being scored.

    Returns
    -------
    dict
        The base revision, the split, hashes of the passage file and of every edited tensor,
        the random seeds, the dtype and device, and hashes of the source files ``loss.py`` and
        ``weights.py``. A change to any of these, including to the scoring code, makes a cached
        result stale.
    """
    here = Path(__file__).resolve().parent
    return {
        "revision": weights.BASE_REV,
        "split": split,
        "passages": _sha256(PASSAGES.read_bytes()),
        "edits": {name: {str(L): _sha256(np.ascontiguousarray(a).tobytes()) for L, a in sorted(e.items())}
                  for name, e in edits.items()},
        "random_seeds": list(RANDOM_SEEDS),
        "dtype": "float32",
        "device": "cpu",
        "source": {f: _sha256((here / f).read_bytes()) for f in ("loss.py", "weights.py")},
    }


def run_loss(split: str, force: bool = False) -> list[Row]:
    """Score one split under the published edit and three random matched edits.

    Slow (about 2 seconds per passage per condition on CPU), so the result is cached at
    ``results/loss_{split}.json`` under the key from :func:`loss_key`.

    Parameters
    ----------
    split : {"dev", "test"}
        The passage split to score.
    force : bool, default False
        Recompute even when a cached result with a matching key exists.

    Returns
    -------
    list of dict
        Rows from :func:`score_passages`, with losses under ``base``, ``published``,
        ``random0``, ``random1`` and ``random2``.
    """
    edit = weights.load_published_edit()
    edits = {"published": {L: edited for L, (_, edited) in edit.items()}}
    for seed in RANDOM_SEEDS:
        edits[f"random{seed}"] = weights.random_matched_edit(edit, seed)
    value: dict[str, Any] = io.cached(
        io.RESULTS / f"loss_{split}.json", loss_key(split, edits),
        lambda: {"threads": torch.get_num_threads(), "rows": score_passages(load_passages(split), edits)},
        force=force)
    return value["rows"]
