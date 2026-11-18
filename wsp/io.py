"""Result files: JSON written atomically, and a cache keyed on its inputs."""
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

RESULTS = Path(__file__).resolve().parents[1] / "results"

T = TypeVar("T")


def save_json(path: str | os.PathLike[str], obj: T) -> T:
    """Write an object as JSON, atomically.

    The text is written to a temporary file beside the target and then renamed, so a reader
    never sees a partial file.

    Parameters
    ----------
    path : str or path-like
        Where to write. Parent folders are created.
    obj : JSON-serialisable
        The object to write.

    Returns
    -------
    JSON-serialisable
        ``obj``, unchanged.

    Raises
    ------
    ValueError
        If ``obj`` contains NaN or infinity, which are not valid JSON. Nothing is written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False)   # NaN is not JSON: fail loudly
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return obj


def load_json(path: str | os.PathLike[str]) -> Any:
    """Read a JSON file.

    Parameters
    ----------
    path : str or path-like
        The file to read.

    Returns
    -------
    object
        The decoded contents.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cached(path: str | os.PathLike[str], key: Any, fn: Callable[[], T], force: bool = False) -> T:
    """Return a computed value, reusing a stored one only when its key matches.

    Parameters
    ----------
    path : str or path-like
        Cache file. It stores ``{"key": key, "value": value}``.
    key : JSON-serialisable
        Everything the value depends on.
    fn : callable
        Computes the value. Called with no arguments, and only when needed.
    force : bool, default False
        Recompute even when a stored value with a matching key exists.

    Returns
    -------
    JSON-serialisable
        The stored value when the file exists, its key equals ``key`` and ``force`` is false;
        otherwise the result of ``fn()``, which is then stored.
    """
    path = Path(path)
    if path.exists() and not force:
        stored = load_json(path)
        if stored.get("key") == key:
            return stored["value"]
    value = fn()
    save_json(path, {"key": key, "value": value})
    return value
