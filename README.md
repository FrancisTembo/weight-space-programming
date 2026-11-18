# Can a published weight edit be found from the weights alone?

A small experiment on the 50-pass weight edit from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), applied to `Qwen/Qwen3.5-0.8B`: without the original model, can one tell that a layer was edited, and which neurons?

- **Research note:** [`RESEARCH_NOTE.md`](RESEARCH_NOTE.md)
- **Report (PDF), with the method written out:** [`report.pdf`](report.pdf)
- **Experiment, with outputs:** [`notebooks/weight_edit_detection.ipynb`](notebooks/weight_edit_detection.ipynb)
- **AI usage:** [`AI_USAGE.md`](AI_USAGE.md)

## Reproduce

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), about 6 GB of RAM and no GPU.

```bash
uv sync

# One download, about 1.75 GB: the base model at the pinned revision
uv run hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17

uv run jupyter nbconvert --to notebook --execute --inplace notebooks/weight_edit_detection.ipynb
```

The notebook takes about 4 minutes. Its slowest step, the Swedish and English loss, is read from `results/loss_dev.json` and `results/loss_test.json`. Those files are reused only when the model revision, passages, edits and scoring code are unchanged. To recompute them (about 25 minutes):

```bash
uv run python scripts/run_loss.py dev --force
uv run python scripts/run_loss.py test --force
```

Nothing is downloaded while the notebook runs. Results in this repository were produced on an AMD Ryzen 5 5600H (6 cores, 12 threads), CPU only, float32, with PyTorch using 6 threads.

## Layout

| Path | What it is |
|---|---|
| `notebooks/weight_edit_detection.ipynb` | The experiment. It defines no functions; it calls `wsp` and shows the results. |
| `wsp/weights.py` | Reads MLP weights, rebuilds the published edit (hash-checked), makes random and synthetic edits |
| `wsp/detect.py` | The two detectors, threshold calibration, the sweep, and the decision rules |
| `wsp/loss.py` | Next-token loss on the bundled passages, with a cache keyed on its inputs and code |
| `wsp/plots.py`, `wsp/io.py` | Tables, the two figures, result files |
| `results/` | Every experiment number quoted in the note, as JSON, plus the two figures |
| `data/` | The published patch, the Swedish and English passages, and [`LICENSES.md`](data/LICENSES.md) |
| `scripts/build_passages.py` | How `data/passages.jsonl` was built (needs network; not run by the notebook) |
| `scripts/run_loss.py` | Runs the loss step on its own |
| `report/report.tex` | Source of `report.pdf`. Build with `pdflatex report.tex`, run twice from the `report/` folder, then copy `report/report.pdf` to the repository root |

## Fixed in advance

Model revision `2fc06364715b967f1860aea9cf38778875588b17`; patch from `RikiyaT/lm_understand-logs` at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Development layers `(2, 6, 10, 13, 17, 21)`, held-out test layers `(0, 3, 7, 11, 14, 18, 22)`. Seeds: synthetic edits `(100, 101)`, random matched edits `(0, 1, 2)`, passages `42`, bootstrap `0`. Detector settings: top 16 singular vectors, 64 neurons.
