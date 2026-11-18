"""Run the slow loss measurement outside the notebook, so the notebook finds it cached.

    uv run python scripts/run_loss.py [dev|test] [--force]

About 2 seconds per passage per condition on CPU: roughly 20 minutes for the test split.
"""
import sys

from wsp import loss

split = next((a for a in sys.argv[1:] if a in ("dev", "test")), "test")
rows = loss.run_loss(split, force="--force" in sys.argv)
for lang in ("sv", "en"):
    if any(r["lang"] == lang for r in rows):
        for condition in ("published", "random0", "random1", "random2"):
            c = loss.paired_change(rows, condition, lang)
            print(f"{split} {lang} {condition:9s} change {c['mean']:+.4f} "
                  f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  worse {c['worse']}/{c['n']}")
