"""Provenance: how data/passages.jsonl was built. Needs network; the notebook never runs this.

    uv run python scripts/build_passages.py [output path]

Swedish: UD Swedish-Talbanken test set (CC BY-SA 4.0), pinned commit below. Consecutive sentences
of one document are joined into passages of about 256 tokens, so loss is measured on running
prose. Three documents are held for development; the rest are test.
English: 32 GSM8K test questions (MIT), rows 200 to 231, which no released loop transcript uses.
"""
import json
import random
import sys
import urllib.request
from typing import Any

from transformers import AutoTokenizer

from wsp import weights

TALBANKEN = ("https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/"
             "0185ae1fb4728ae501bf2a329b958e6c6acb9c8d/sv_talbanken-ud-test.conllu")
GSM8K = ("https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main"
         "&split=test&offset=200&length=100")
PASSAGE_TOKENS = 256
PER_DOC = 8
N_ENGLISH = 32
SEED = 42
DEV_DOCS = 3


def fetch(url: str) -> str:
    """Download a text resource.

    Parameters
    ----------
    url : str
        Address to fetch.

    Returns
    -------
    str
        The response body, decoded as UTF-8.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8")


def swedish_docs(conllu: str) -> dict[str, list[str]]:
    """Split a CoNLL-U treebank into documents of sentences.

    Parameters
    ----------
    conllu : str
        Contents of the CoNLL-U file.

    Returns
    -------
    dict of str to list of str
        ``{document id: sentences in order}``, from the ``# newdoc id`` and ``# text`` lines.
    """
    docs: dict[str, list[str]] = {}
    cur = ""
    for line in conllu.splitlines():
        if line.startswith("# newdoc id ="):
            cur = line.split("=", 1)[1].strip()
            docs[cur] = []
        elif line.startswith("# text ="):
            docs[cur].append(line.split("=", 1)[1].strip())
    return docs


def passages(sentences: list[str], tok: Any) -> list[str]:
    """Join consecutive sentences into passages of about :data:`PASSAGE_TOKENS` tokens.

    Parameters
    ----------
    sentences : list of str
        Sentences of one document, in order.
    tok : transformers tokenizer
        Tokenizer used to count tokens.

    Returns
    -------
    list of str
        Passages, in order. A final passage shorter than half the target length is dropped.
    """
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for s in sentences:
        k = len(tok(" " + s, add_special_tokens=False).input_ids)
        if buf and n + k > PASSAGE_TOKENS:
            out.append(" ".join(buf))
            buf, n = [], 0
        buf.append(s)
        n += k
    if n >= PASSAGE_TOKENS // 2:          # drop a short tail
        out.append(" ".join(buf))
    return out


def main(out_path: str) -> None:
    """Build the passage file.

    Parameters
    ----------
    out_path : str
        Where to write the JSON-lines file: one passage per line with ``id``, ``lang``, ``doc``,
        ``split`` and ``text``.
    """
    tok = AutoTokenizer.from_pretrained(weights.base_dir())
    rng = random.Random(SEED)
    docs = swedish_docs(fetch(TALBANKEN))
    ids = sorted(docs)
    rng.shuffle(ids)
    dev = set(ids[:DEV_DOCS])
    rows = []
    for d in sorted(docs):
        ps = passages(docs[d], tok)
        rng.shuffle(ps)
        for i, p in enumerate(ps[:PER_DOC]):
            rows.append({"id": f"sv_{d}_{i}", "lang": "sv", "doc": d,
                         "split": "dev" if d in dev else "test", "text": p})
    for r in json.loads(fetch(GSM8K))["rows"][:N_ENGLISH]:
        rows.append({"id": f"en_gsm8k_test_{r['row_idx']}", "lang": "en", "doc": f"gsm8k_{r['row_idx']}",
                     "split": "test", "text": r["row"]["question"]})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} passages to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(weights.ROOT / "data" / "passages.jsonl"))
