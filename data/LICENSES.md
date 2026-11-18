# Data and weights: sources and licences

| Item | Source | Licence |
|---|---|---|
| `passages.jsonl`, Swedish rows (`lang: sv`) | UD Swedish-Talbanken test set, commit `0185ae1f`, <https://github.com/UniversalDependencies/UD_Swedish-Talbanken>. Sentences joined into passages by `scripts/build_passages.py`. | CC BY-SA 4.0. These rows are a derivative and are shared under the same licence. |
| `passages.jsonl`, English rows (`lang: en`) | GSM8K test questions, rows 200 to 231, <https://huggingface.co/datasets/openai/gsm8k> | MIT |
| `edit_p50c/edit.npz`, `edit_p50c/manifest.json` | The 50-pass edit from "Sparks of Weight-Space Programming", <https://github.com/RikiyaT/lm_understand-logs>, commit `4d09ba6d`, `exp8_math/runs/p50c/edit/` | CC BY 4.0 (Takehi, Teoh, Zhang, Khattab) |
| Base model weights (not in this repository) | `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17` | Apache-2.0 |

Talbanken is based on the Professional Prose section of Talbanken, originally annotated at Lund
University; the UD version is distributed by the Universal Dependencies project.

## Licence texts and notices

- CC BY-SA 4.0: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>
- CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/legalcode>
- Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0>

The GSM8K questions are distributed under the MIT License, whose notice must accompany copies:

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
