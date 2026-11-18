# Can a published weight edit be found from the weights alone?

SE.LLMA candidate assignment on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). The numbers from my experiment are documented in `results/`, and the companion notebook is `notebooks/weight_edit_detection.ipynb`. The figures in section 1 come from the logs the article's authors released. A longer version with the full method is in `report.pdf`.

## 1. Critical read of the article

### 1.1 Strongest evidence

The most convincing results are the tiny-network ones that use no model feedback, such as writing a game-playing network in one reply. I did not check those. I did check that the published Qwen3.5-0.8B edit rebuilds from its patch to the exact released weights.

### 1.2 Apparent limitations

The article's "50 forward passes" were a long feedback loop. The run took 158 turns and 20.3 hours. One pass ran up to hundreds of practice problems on the original and edited model and returned the accuracy difference, which climbed from +6 to +40 points. This is optimisation through repeated feedback, not reading weights and knowing the answer.

### 1.3 What the article does and does not justify

I think the article justifies this much: a frontier model with feedback might be able to find a small edit that fixes one particular failure. It does not show that this is better than standard methods, because it ran no baseline with steering or fine-tuning. It also does not show that it brings new capabilities.

## 2. Question and hypothesis

### 2.1 The question

If someone publishes weights containing a malicious or restrictive edit, can it be found from the weights alone, without a fingerprint or the original model weights?

### 2.2 Why it matters for Swedish models

From my understanding, SE.LLMA builds on third-party open model weights. If weight inspection catches such edits, it is a cheap safeguard. If not, every adopted checkpoint needs behavioural checks in Swedish.

### 2.3 Two claims and how they are judged

I tested two claims, and fixed the rules for judging them before the held-out run.

|  | A. Screening: is there an edit? | B. Localisation: where, given a suspect layer? |
| :---: | :---- | :---- |
| Rule | A layer alarms if its statistic exceeds the highest among six clean development layers. | The detector names 64 neurons |
| Supported | Recall ≥ 0.80, at most 1 false alarm on 18 other clean layers | Mean precision ≥ 0.80 |
| Challenged | Recall < 0.50, or more than 1 false alarm | Mean precision < 0.50 |

I had expected claim A to fail, because exploration had revealed two pristine layers scoring above the published edit. I also threw out a previous, more favourable estimate, because it used the true edited neurons to choose what to score.

## 3. Experiment

### 3.1 The edit

An edit of the published kind changes 64 columns of a layer's MLP output matrix $W$, which has one column $w_j$ per neuron. Each changed column loses a multiple $m$ of its component along one unit direction $u$:

$$w_j' = w_j - m\,u\,(u^\top w_j), \qquad j \in S,\quad |S| = 64.$$

In the published edit, $m = 6$ in both layers.

### 3.2 Two detectors

There are two detectors, and each reads the weights of a single checkpoint.

The spectral one uses the fact that a change confined to a few neurons appears as a singular vector of the layer's output matrix that sits on those neurons. With $v_1, \dots, v_{16}$ the right singular vectors for the 16 largest singular values, its statistic is

$$T = n \max_{k \le 16} \sum_{j=1}^{n} v_{kj}^{4}, \qquad n = 3584.$$

$T$ is 1 for a vector spread evenly over all neurons and $n/64 = 56$ for one spread evenly over 64 of them. The detector names the 64 neurons with the largest absolute entries in the vector that gives the maximum.

The balance one, which is the alternative method, uses the fact that a neuron's output weight norm normally tracks its input norms. It scores how far the 64 most unusual neurons sit from that relation.

### 3.3 Held-out test and controls

The held-out test is 84 new synthetic edits of the published kind, with new seeds, at 3%, 5% and 9% of the layer's weight norm. They sit in seven layers where no edit had been tried before, although I had already seen the clean statistics of every layer. I judge the claims at 9%. The published edit is 8.1% and 10.6%.

The controls are the unedited model (for false alarms), three random edits of the same size on the published edit's neurons, and the alternative detector under the same rules. Prompting and decoding controls do not apply, because the detectors never run the model.

### 3.4 Swedish measurement

For the Swedish measurement I use next-token loss on 84 held-out Swedish passages (UD Talbanken) and 32 English questions (GSM8K).

## 4. Results

| At 9% edit size, 28 held-out edits | Spectral | Balance |
| ----- | :---: | :---: |
| A. Recall at the threshold | 0.00 | 0.11 |
| A. False alarms on 18 clean layers | 0 | 1 |
| **A. Verdict** | **challenged** | **challenged** |
| B. Mean precision at 64 | 0.77 | 0.18 |
| **B. Verdict** | **inconclusive** | **challenged** |
| Median rank of the edited layer, of 24 | 4 | 18 |
| Mean precision at 5% and 3% edit size | 0.11, 0.07 | 0.02, 0.01 |

- On the published edit (seen in exploration, reproduced here), the spectral detector names 59 of the 64 edited neurons in each layer and recovers the direction (cosine 0.94 and 0.95). Even so, the two layers rank 4th and 3rd of 24 and do not alarm, and the two layers above them are untouched.
- Random edits on the same neurons are found equally well (precision 0.97 to 1.00). So the detector responds to the shape and size of a change and is blind to its meaning.
- Localisation is weaker when the edit follows a direction the layer already uses. Precision is 0.58 there, against 0.95 for edits along a random direction.
- On the held-out Swedish passages, the published edit raises loss by 0.052 nats per token (95% interval 0.046 to 0.059), and it does so on all 84. Random edits of the same size cost 0.009 to 0.032. On the English questions the published edit (+0.007) is within the range of the random edits.

## 5. Interpretation and limits

Often the weights alone will tell you where such an edit falls. They do not signal that a checkpoint has been edited: the pattern this detector looks for is already present in clean layers, so there is no threshold on its statistic that can tell them apart. An edit half the size is rarely found (1 of 28 localised), and one aligned with a direction already used by the layer is found unreliably.

This is one model, two detectors and one published edit in two layers. The synthetic edits all come from one family, and the threshold needs known-clean weights to calibrate. I did not test the article's other two edits (one spread over 30 matrices, one in an attention head, which these detectors do not look at), or whether ordinary fine-tuning raises alarms. The localisation result missed its bar (0.77 against 0.80), so I report it as inconclusive. The Swedish prose and English maths passages are not matched, so the loss result does not show the edit is specific to Swedish. Random edits also cost more on the Swedish set.

## 6. Recommendation and next experiment

- Do not rely on weight inspection to vet third-party weights. Verify the hash against a trusted source, and run a Swedish behavioural check on every checkpoint or edit adopted.
- Treat edits found by English feedback as untested in Swedish. The one published example has a measurable Swedish cost that its English evaluation never saw.
- I would keep watching the article's approach without building on it yet. Its language-model result is a fix for one particular failure, found by a 20-hour feedback loop, with no comparison to standard steering.

The most informative next experiment would be to build an edit that intentionally lowers Swedish quality but not English, on matched parallel text, at several sizes. If Swedish can be damaged by an edit smaller than these detectors can localise (they localised 1 of 28 at 5%), they offer no defence and a stronger weight statistic would be needed. This will give more indication that the model owner can intentionally edit the weights to hinder development for a specific area of research. It would be less evident than a restriction that can be done during training, where the model is trained to deny work along a subject matter.
