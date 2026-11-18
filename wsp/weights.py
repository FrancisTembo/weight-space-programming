"""Read MLP weights from the pinned base checkpoint, rebuild the published edit, and make
random and synthetic edits.

Only weight files are read here: no model is run.
"""
import hashlib
import json
import os
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

BASE_REPO = "Qwen/Qwen3.5-0.8B"
BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"
WEIGHTS_FILE = "model.safetensors-00001-of-00001.safetensors"
ROOT = Path(__file__).resolve().parents[1]
EDIT_DIR = ROOT / "data" / "edit_p50c"
N_LAYERS = 24
PARTS = ("down_proj", "gate_proj", "up_proj")

Array: TypeAlias = npt.NDArray[np.float32]
"""A float32 weight matrix."""
Bits: TypeAlias = npt.NDArray[np.uint16]
"""The bfloat16 bit patterns of a tensor, as stored in the checkpoint."""
Mlps: TypeAlias = dict[tuple[int, str], Array]
"""MLP weights keyed by ``(layer, part)``, with ``part`` one of :data:`PARTS`."""
Seed: TypeAlias = int | tuple[int, ...]
"""A seed accepted by :func:`numpy.random.default_rng`."""


def tensor_name(layer: int, part: str = "down_proj") -> str:
    """Name of one MLP weight tensor in the base checkpoint.

    Parameters
    ----------
    layer : int
        Layer index, 0 to 23.
    part : str, default "down_proj"
        One of :data:`PARTS`.

    Returns
    -------
    str
        The tensor's key in the safetensors file.
    """
    return f"model.language_model.layers.{layer}.mlp.{part}.weight"


def base_dir(cache_dir: str | os.PathLike[str] | None = None) -> str:
    """Folder of the pinned base snapshot in the Hugging Face cache.

    Parameters
    ----------
    cache_dir : str or path-like, optional
        Hugging Face hub cache to look in. Defaults to the configured cache.

    Returns
    -------
    str
        Path of the snapshot folder for :data:`BASE_REV`.

    Raises
    ------
    FileNotFoundError
        If the weights file of the pinned revision is not in the cache. The message names the
        revision and points to the download step in the README.
    """
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_dir = HF_HUB_CACHE
    d = os.path.join(cache_dir, "models--" + BASE_REPO.replace("/", "--"), "snapshots", BASE_REV)
    if not os.path.exists(os.path.join(d, WEIGHTS_FILE)):
        raise FileNotFoundError(
            f"Base weights not found: {BASE_REPO} at revision {BASE_REV} is not in {cache_dir}. "
            "Run the download step in README.md first.")
    return d


def _sha256(a: npt.NDArray[Any]) -> str:
    """Hex sha256 of an array's bytes in C order.

    Parameters
    ----------
    a : ndarray
        Any array.

    Returns
    -------
    str
        The hex digest.
    """
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _bits_to_f32(bits: Bits) -> Array:
    """Convert bfloat16 bit patterns to float32, exactly.

    Parameters
    ----------
    bits : ndarray of uint16
        The bfloat16 bit patterns.

    Returns
    -------
    ndarray of float32
        The same values. Every bfloat16 value is representable in float32.
    """
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _read_bits(names: Iterable[str]) -> dict[str, Bits]:
    """Read bfloat16 tensors of the base checkpoint as raw bit patterns.

    Parameters
    ----------
    names : iterable of str
        Tensor keys in the safetensors file.

    Returns
    -------
    dict of str to ndarray of uint16
        Bit patterns of each tensor, in its stored shape.

    Raises
    ------
    FileNotFoundError
        If the base weights are not in the cache.
    ValueError
        If a tensor is not stored as bfloat16.
    """
    path = os.path.join(base_dir(), WEIGHTS_FILE)
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        for name in names:
            info = head[name]
            if info["dtype"] != "BF16":
                raise ValueError(f"{name}: expected BF16, found {info['dtype']}")
            a, b = info["data_offsets"]
            f.seek(8 + n + a)
            out[name] = np.frombuffer(f.read(b - a), dtype=np.uint16).reshape(info["shape"]).copy()
    return out


def load_mlps() -> Mlps:
    """Load the MLP weights of all 24 language-model layers of the base checkpoint.

    Returns
    -------
    Mlps
        ``{(layer, part): float32 array}``. ``down_proj`` has shape (1024, 3584), with one
        column per neuron; ``gate_proj`` and ``up_proj`` have shape (3584, 1024), with one row
        per neuron.

    Raises
    ------
    FileNotFoundError
        If the base weights are not in the cache.
    """
    keys = [(L, p) for L in range(N_LAYERS) for p in PARTS]
    bits = _read_bits([tensor_name(L, p) for L, p in keys])
    return {(L, p): _bits_to_f32(bits[tensor_name(L, p)]) for L, p in keys}


def rebuild_tensor(base_bits: Bits, idx: npt.NDArray[np.integer[Any]], val: Bits,
                   base_sha256: str, edited_sha256: str) -> Bits:
    """Apply a sparse patch to a tensor's bit patterns, checking both ends against hashes.

    Parameters
    ----------
    base_bits : ndarray of uint16
        Bit patterns of the unedited tensor.
    idx : ndarray of int
        Flat indices of the entries the patch changes.
    val : ndarray of uint16
        New bit patterns for those entries.
    base_sha256 : str
        Expected sha256 of ``base_bits``.
    edited_sha256 : str
        Expected sha256 of the result.

    Returns
    -------
    ndarray of uint16
        Bit patterns of the edited tensor, in the shape of ``base_bits``.

    Raises
    ------
    ValueError
        If ``base_bits`` or the rebuilt tensor does not match its expected hash. Unverified
        weights are never returned.
    """
    if _sha256(base_bits) != base_sha256:
        raise ValueError("base tensor hash does not match the patch manifest")
    flat = base_bits.copy().reshape(-1)
    flat[idx] = val
    edited = flat.reshape(base_bits.shape)
    if _sha256(edited) != edited_sha256:
        raise ValueError("rebuilt tensor hash does not match the patch manifest")
    return edited


def load_published_edit() -> dict[int, tuple[Array, Array]]:
    """Rebuild the article's published 50-pass edit from its patch.

    Returns
    -------
    dict of int to (ndarray, ndarray)
        ``{layer: (base, edited)}`` float32 ``down_proj`` matrices, in manifest order
        (layer 15, then 19).

    Raises
    ------
    FileNotFoundError
        If the base weights are not in the cache.
    ValueError
        If the patch is not in sparse form, or any tensor fails its hash check.
    """
    manifest = json.loads((EDIT_DIR / "manifest.json").read_text())
    patch = np.load(EDIT_DIR / "edit.npz")
    names = [t["name"] for t in manifest["tensors"]]
    bits = _read_bits(names)
    out = {}
    for i, t in enumerate(manifest["tensors"]):
        if t["form"] != "sparse":
            raise ValueError(f"{t['name']}: unsupported patch form {t['form']}")
        edited = rebuild_tensor(bits[t["name"]], patch[f"t{i}_idx"], patch[f"t{i}_val"],
                                t["base_sha256"], t["edited_sha256"])
        layer = int(t["name"].split(".layers.")[1].split(".")[0])
        out[layer] = (_bits_to_f32(bits[t["name"]]), _bits_to_f32(edited))
    return out


def describe_edit(base: Array, edited: Array) -> dict[str, Any]:
    """Measure what an edit to one ``down_proj`` matrix is.

    Parameters
    ----------
    base : ndarray, shape (d, n)
        The matrix before the edit.
    edited : ndarray, shape (d, n)
        The matrix after the edit.

    Returns
    -------
    dict
        ``columns`` : list of int
            Indices of the changed columns (neurons).
        ``rel_change`` : float
            Frobenius norm of the change over that of ``base``.
        ``rank1_share`` : float or None
            Share of the change's squared norm in its top singular direction.
        ``multiplier`` : float or None
            Least-squares ``m`` in the model ``w' = w - m * u (u . w)``, with ``u`` the top
            direction of the change. None when the changed columns have no loading on ``u``.
        ``direction`` : list of float or None
            The unit vector ``u``.

        An unchanged matrix gives no columns, a ``rel_change`` of 0 and None for the rest.

    Notes
    -----
    Read ``multiplier`` together with ``rank1_share``: the single-direction model describes the
    edit only when ``rank1_share`` is close to 1.
    """
    delta = edited - base
    cols = np.flatnonzero(np.abs(delta).sum(0) > 0)
    if len(cols) == 0:
        return {"columns": [], "rel_change": 0.0, "rank1_share": None, "multiplier": None, "direction": None}
    U, s, _ = np.linalg.svd(delta[:, cols], full_matrices=False)
    u = U[:, 0]
    before, after = u @ base[:, cols], u @ edited[:, cols]
    denom = float(before @ before)
    return {
        "columns": [int(c) for c in cols],
        "rel_change": float(np.linalg.norm(delta) / np.linalg.norm(base)),
        "rank1_share": float(s[0] ** 2 / (s ** 2).sum()),
        "multiplier": float(-((after - before) @ before) / denom) if denom > 0 else None,
        "direction": [float(x) for x in u],
    }


def random_matched_edit(edit: dict[int, tuple[Array, Array]], seed: int) -> dict[int, Array]:
    """Make a control edit: the published edit's neurons and sizes, along a random direction.

    Parameters
    ----------
    edit : dict of int to (ndarray, ndarray)
        ``{layer: (base, edited)}`` as returned by :func:`load_published_edit`.
    seed : int
        Seed of the one generator used across all layers, in dict order.

    Returns
    -------
    dict of int to ndarray
        ``{layer: edited array}``. Each changed column moves by the same L2 norm as in ``edit``,
        but along one random unit direction per layer, with a random sign per column.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for layer, (base, edited) in edit.items():
        delta = edited - base
        cols = np.flatnonzero(np.abs(delta).sum(0) > 0)
        r = rng.standard_normal(size=int(base.shape[0])).astype(np.float32)
        r /= np.linalg.norm(r)
        norms = np.linalg.norm(delta[:, cols], axis=0)
        signs = rng.choice([-1.0, 1.0], size=len(cols)).astype(np.float32)
        new = base.copy()
        new[:, cols] = base[:, cols] + np.outer(r, norms * signs)
        out[layer] = new
    return out


def synthetic_edit(down: Array, seed: Seed, rel_change: float, kind: str,
                   n_cols: int = 64) -> tuple[Array, npt.NDArray[np.intp]]:
    """Make an edit of the published kind, for testing detectors.

    One direction is pushed out of the ``n_cols`` columns that load on it most, and the change is
    scaled to ``rel_change`` of the matrix's Frobenius norm.

    Parameters
    ----------
    down : ndarray, shape (d, n)
        A clean ``down_proj`` matrix. It is not modified.
    seed : int or tuple of int
        Seed for the random generator.
    rel_change : float
        Size of the edit, as a share of the Frobenius norm of ``down``.
    kind : {"svd", "random"}
        ``"svd"`` uses a direction the layer already uses (one of the top 8 left singular
        vectors of a random sample of columns); ``"random"`` uses a random unit direction.
    n_cols : int, default 64
        Number of columns to edit.

    Returns
    -------
    edited : ndarray, shape (d, n)
        The edited matrix.
    columns : ndarray of int
        Sorted indices of the edited columns.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"svd"`` or ``"random"``.
    """
    rng = np.random.default_rng(seed)
    if kind == "svd":
        sample = rng.choice(down.shape[1], min(1024, down.shape[1]), replace=False)
        U, _, _ = np.linalg.svd(down[:, sample], full_matrices=False)
        u = U[:, rng.integers(0, 8)]
    elif kind == "random":
        u = rng.standard_normal(size=int(down.shape[0])).astype(np.float32)
        u /= np.linalg.norm(u)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    load = u @ down
    cols = np.argsort(-np.abs(load))[:n_cols]
    delta = -np.outer(u, load[cols])
    scale = rel_change * np.linalg.norm(down) / np.linalg.norm(delta)
    edited = down.copy()
    edited[:, cols] += scale * delta
    return edited, np.sort(cols)


def apply_edits(mlps: Mlps, edits: dict[int, Array]) -> Mlps:
    """Return a checkpoint's MLP weights with some ``down_proj`` matrices replaced.

    Parameters
    ----------
    mlps : Mlps
        Clean weights. Not modified.
    edits : dict of int to ndarray
        ``{layer: edited down_proj}``.

    Returns
    -------
    Mlps
        A shallow copy of ``mlps`` with the ``down_proj`` of each layer in ``edits`` replaced.
    """
    out = dict(mlps)
    for layer, array in edits.items():
        out[(layer, "down_proj")] = array
    return out
