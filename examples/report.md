# LLM Response Evaluation Report

Dataset: 40 prompts, 120 candidate responses from 3 models (atlas-v2, borealis-mini, cirrus-large), 360 human annotations by 3 annotators (ann-01, ann-02, ann-03), 24 gold-labelled responses. Scores are 1-5 on six rubric dimensions (higher is better; for `hallucination`, higher means fewer fabrications).

## 1. Model comparison

Mean consensus score per dimension (mean of annotators per response, then mean over 40 prompts) with 95% bootstrap CIs (2000 resamples over prompts). `overall` is the rubric-weighted mean of the six dimensions.

| Dimension | atlas-v2 | borealis-mini | cirrus-large |
|---|---|---|---|
| factual_accuracy | **4.47 [4.17, 4.73]** | 3.68 [3.35, 4.01] | 4.45 [4.17, 4.70] |
| relevance | 4.69 [4.53, 4.83] | 4.24 [3.98, 4.49] | **4.83 [4.68, 4.95]** |
| completeness | **4.47 [4.21, 4.71]** | 3.18 [2.88, 3.49] | 3.74 [3.44, 4.06] |
| instruction_following | 4.40 [4.13, 4.63] | 4.58 [4.37, 4.77] | **4.59 [4.39, 4.78]** |
| hallucination | 4.48 [4.10, 4.81] | 3.72 [3.25, 4.16] | **4.56 [4.27, 4.80]** |
| style | 3.94 [3.75, 4.12] | 4.32 [4.09, 4.52] | **4.82 [4.71, 4.91]** |
| **overall** | 4.45 [4.32, 4.57] | 3.92 [3.73, 4.09] | **4.47 [4.34, 4.60]** |

Ranking by overall score: cirrus-large > atlas-v2 > borealis-mini.

### Win rates (overall score)

Share of prompts where the row model beats the column model; differences smaller than 0.25 points count as ties (half a win).

|  | atlas-v2 | borealis-mini | cirrus-large |
|---|---|---|---|
| atlas-v2 | - | 0.80 | 0.49 |
| borealis-mini | 0.20 | - | 0.23 |
| cirrus-large | 0.51 | 0.78 | - |

Per-dimension differences whose paired 95% bootstrap CI excludes zero:

- `completeness`: atlas-v2 > borealis-mini by 1.29 (CI [0.85, 1.70])
- `style`: cirrus-large > atlas-v2 by 0.88 (CI [0.68, 1.07])
- `hallucination`: cirrus-large > borealis-mini by 0.84 (CI [0.33, 1.37])
- `factual_accuracy`: atlas-v2 > borealis-mini by 0.79 (CI [0.36, 1.21])
- `factual_accuracy`: cirrus-large > borealis-mini by 0.77 (CI [0.33, 1.19])
- `hallucination`: atlas-v2 > borealis-mini by 0.77 (CI [0.20, 1.35])
- `completeness`: atlas-v2 > cirrus-large by 0.73 (CI [0.37, 1.08])
- `relevance`: cirrus-large > borealis-mini by 0.59 (CI [0.32, 0.88])
- `completeness`: cirrus-large > borealis-mini by 0.56 (CI [0.07, 1.05])
- `style`: cirrus-large > borealis-mini by 0.50 (CI [0.29, 0.72])
- `relevance`: atlas-v2 > borealis-mini by 0.45 (CI [0.14, 0.77])
- `style`: borealis-mini > atlas-v2 by 0.38 (CI [0.10, 0.63])

### Overall score by prompt category

| Category | atlas-v2 | borealis-mini | cirrus-large |
|---|---|---|---|
| explanation | 4.43 | 3.76 | 4.74 |
| factual_qa | 4.53 | 4.17 | 4.44 |
| instruction_following | 4.39 | 3.61 | 4.56 |
| open_ended | 4.46 | 3.88 | 4.12 |
| summarization | 4.41 | 4.03 | 4.44 |

## 2. Inter-annotator agreement

Krippendorff's alpha uses the ordinal distance metric; Fleiss' kappa treats scores as nominal categories; Cohen's kappa is averaged over annotator pairs (quadratic weights). Rule of thumb for alpha: >= 0.80 reliable, 0.67-0.80 tentative, < 0.67 weak. `alpha (bias-corr.)` is interval alpha after subtracting each annotator's estimated offset; the gap to the raw alpha is the agreement lost to systematic leniency or harshness rather than to genuinely ambiguous items.

| Dimension | alpha (ordinal) | reading | alpha (bias-corr.) | Fleiss kappa | Cohen kappa (quad.) | exact agree | within 1 |
|---|---|---|---|---|---|---|---|
| factual_accuracy | 0.688 | tentative | 0.817 | 0.320 | 0.742 | 0.59 | 0.93 |
| relevance | 0.558 | weak | 0.596 | 0.261 | 0.524 | 0.69 | 0.94 |
| completeness | 0.694 | tentative | 0.744 | 0.263 | 0.662 | 0.47 | 0.84 |
| instruction_following | 0.552 | weak | 0.565 | 0.270 | 0.507 | 0.66 | 0.90 |
| hallucination | 0.780 | tentative | 0.890 | 0.377 | 0.860 | 0.71 | 0.94 |
| style | 0.388 | weak | 0.542 | 0.192 | 0.435 | 0.52 | 0.91 |

### Pairwise annotator agreement (averaged over dimensions)

| Pair | exact | within 1 | mean diff (a-b) | kappa | kappa (quad.) |
|---|---|---|---|---|---|
| ann-01 vs ann-02 | 0.61 | 0.92 | -0.45 | 0.266 | 0.578 |
| ann-01 vs ann-03 | 0.68 | 0.95 | 0.10 | 0.448 | 0.776 |
| ann-02 vs ann-03 | 0.53 | 0.87 | 0.55 | 0.166 | 0.511 |

## 3. Annotator calibration

`bias` comes from an additive model (score = item quality + annotator offset), centred on the median annotator. `spread` is the ratio of the annotator's score std to the consensus std (below 1 means a compressed scale). `corr` is the correlation with the leave-one-out consensus. Gold columns compare against expert-adjudicated labels.

| Annotator | mean | bias | spread | corr | gold bias | gold MAE | gold exact | flags |
|---|---|---|---|---|---|---|---|---|
| ann-01 | 4.17 | -0.00 | 1.32 | 0.82 | -0.01 | 0.05 | 0.95 | ok |
| ann-02 | 4.62 | 0.44 | 0.62 | 0.76 | 0.39 | 0.39 | 0.65 | lenient, compressed_range |
| ann-03 | 4.07 | -0.11 | 1.40 | 0.80 | -0.17 | 0.24 | 0.78 | ok |

- **ann-02**: Scores +0.44 above peers on average; review anchor descriptions for scores 2-3 and re-score gold items; uses a narrow part of the scale; encourage using the full 1-5 range.
  Dimension flags: factual_accuracy: lenient; completeness: lenient; instruction_following: lenient, noisy; style: lenient

## 4. Disagreement workflow

37 of 120 responses were flagged after removing each annotator's bias: 18 ambiguous, 16 dimension_dispute, 3 outlier_annotator. A dimension is disputed when the bias-corrected scores span at least 1.75 points or their std is at least 0.8; an item is `ambiguous` when 2+ dimensions are disputed or one dimension spans 2.75+ points. All flagged items go to `adjudication_queue.jsonl`, most contentious first.

Synthetic sanity check: the generator planted 18 ambiguous responses. Items labelled `ambiguous`: precision 0.78, recall 0.78. Any flag: recall 1.00.

Top of the adjudication queue:

| # | Response | Label | Priority | Disputed dims (scores by annotator) |
|---|---|---|---|---|
| 1 | r-p035-borealis | ambiguous | 10.45 | relevance 4/4/2; completeness 1/2/4; instruction_following 5/5/3 |
| 2 | r-p021-borealis | ambiguous | 9.91 | factual_accuracy 3/3/1; relevance 3/5/5; instruction_following 3/2/4 |
| 3 | r-p037-atlas | ambiguous | 9.31 | completeness 5/5/3; instruction_following 3/5/5; style 4/5/2 |
| 4 | r-p021-cirrus | ambiguous | 8.95 | completeness 3/3/4; instruction_following 3/4/5; style 3/5/5 |
| 5 | r-p035-atlas | ambiguous | 8.37 | completeness 3/1/3; instruction_following 5/3/5 |
| 6 | r-p021-atlas | ambiguous | 8.29 | instruction_following 2/4/1; style 1/4/1 |
| 7 | r-p037-borealis | ambiguous | 7.72 | factual_accuracy 1/4/2; style 5/5/3 |
| 8 | r-p037-cirrus | ambiguous | 7.31 | completeness 5/5/3; style 2/5/3 |
| 9 | r-p033-atlas | ambiguous | 7.14 | factual_accuracy 2/4/1; hallucination 4/3/2 |
| 10 | r-p025-borealis | ambiguous | 6.96 | instruction_following 1/5/3 |

Example task `r-p035-borealis` (borealis-mini). Prompt: "What's the best programming language?"

- `relevance`:
  - ann-01 gave 4: Stays on topic.
  - ann-02 gave 4: Stays on topic.
  - ann-03 gave 2: The prompt is open to interpretation and this reading feels only partly on target.
- `completeness`:
  - ann-01 gave 1: Missing: There is no single best language; it depends on what you...; Consider the ecosystem, available libraries, and what you...
  - ann-02 gave 2: Missing: There is no single best language; it depends on what you...; Consider the ecosystem, available libraries, and what you...
  - ann-03 gave 4: Covers the main points; a minor detail is missing.
- `instruction_following`:
  - ann-01 gave 5: Follows the requested format and length.
  - ann-02 gave 5: Follows the requested format and length.
  - ann-03 gave 3: Hard to say what the request expects; the answer only partially fits.

## 5. Automated annotator vs human consensus (mock backend)

The mock judge is a deterministic heuristic (fact overlap, false-claim matching, constraint checks). On this synthetic data it reads the same signals the generator planted, so its agreement is an upper bound, not evidence of real judge quality.

| Dimension | alpha vs median | kappa (quad.) | MAE | exact | bias (judge-human) |
|---|---|---|---|---|---|
| factual_accuracy | 0.929 | 0.961 | 0.09 | 0.92 | -0.07 |
| relevance | 0.621 | 0.642 | 0.31 | 0.79 | -0.16 |
| completeness | 0.902 | 0.902 | 0.26 | 0.77 | 0.06 |
| instruction_following | 0.861 | 0.901 | 0.12 | 0.89 | 0.05 |
| hallucination | 0.923 | 0.938 | 0.17 | 0.85 | 0.03 |
| style | 0.647 | 0.586 | 0.51 | 0.57 | -0.24 |

## 6. Per-model feedback

### cirrus-large (rank 1, overall 4.47 [4.34, 4.60])

Responses: 40; likely hallucinations: 4; instruction violations: 3; flagged for adjudication: 10. Win rate vs atlas-v2: 0.51, borealis-mini: 0.78.

**Strengths**

- `style` (4.82, +0.69 vs others): Concise and cleanly formatted.
- `hallucination` (4.56, +0.46 vs others): Rarely introduces unsupported content.
- `factual_accuracy` (4.45, +0.37 vs others): Claims are reliably correct.

**Weaknesses**

- `completeness` (3.74, -0.09 vs others): Responses omit reference key points; prompt the model to cover all parts of multi-part questions.
  - Example `r-p002-cirrus` (score 2.0): "Distance is not the cause; Earth is actually closest to the Sun in early January."
    Annotator note: Missing: Seasons are caused by the tilt of Earth's axis, about 23....; The hemisphere tilted toward the Sun gets more direct sun...

### atlas-v2 (rank 2, overall 4.45 [4.32, 4.57])

Responses: 40; likely hallucinations: 5; instruction violations: 4; flagged for adjudication: 10. Win rate vs borealis-mini: 0.80, cirrus-large: 0.49.

**Strengths**

- `completeness` (4.47, +1.01 vs others): Covers the reference key points well.
- `factual_accuracy` (4.47, +0.41 vs others): Claims are reliably correct.
- `hallucination` (4.48, +0.35 vs others): Rarely introduces unsupported content.

**Weaknesses**

- `style` (3.94, -0.62 vs others): Cut conversational boilerplate (greetings, sign-offs) and match the requested layout.
  - Example `r-p021-atlas` (score 2.0): "Absolutely, here you go! - Recursion is when you solve a problem by solving a smaller version of the same problem. - It is like nesting dolls: you open one doll to find a smaller one inside, again and again. - Every r..."
    Annotator note: Style issues: unnecessary pleasantries, wrong layout.
- `instruction_following` (4.40, -0.18 vs others): Explicit format and length constraints are often ignored; add constraint checks or few-shot examples.
  - Example `r-p021-atlas` (score 2.33): "Absolutely, here you go! - Recursion is when you solve a problem by solving a smaller version of the same problem. - It is like nesting dolls: you open one doll to find a smaller one inside, again and again. - Every r..."
    Annotator note: Violates instructions: over word limit (75 > 60); expected a paragraph, got bullets

### borealis-mini (rank 3, overall 3.92 [3.73, 4.09])

Responses: 40; likely hallucinations: 14; instruction violations: 3; flagged for adjudication: 17. Win rate vs atlas-v2: 0.20, cirrus-large: 0.23.

**Strengths**

- No dimension stands out above the other models.

**Weaknesses**

- `completeness` (3.18, -0.93 vs others): Responses omit reference key points; prompt the model to cover all parts of multi-part questions.
  - Example `r-p009-borealis` (score 2.0): "It is a universal physical constant, written as c. On a related note, the history of scientific discovery is full of surprising stories."
    Annotator note: Missing: The speed of light in a vacuum is exactly 299,792,458 met...; No object with mass can reach the speed of light.
- `hallucination` (3.72, -0.80 vs others): The model asserts unsupported content; penalize unsupported specifics and encourage hedging.
  - Example `r-p015-borealis` (score 1.33): "- Northwind's third-quarter revenue rose 9 percent to $412 million, mostly from online sales. - The CEO announced she will step down at the end of the year."
    Annotator note: Asserts unsupported content: "The CEO announced she will step down at the end of the year."
- `factual_accuracy` (3.68, -0.78 vs others): Audit the incorrect claims below; consider retrieval grounding or answer verification for factual prompts.
  - Example `r-p031-borealis` (score 2.0): "This produces a linear history without merge commits. Rebase permanently deletes the branch you rebase onto. Many textbooks cover this topic in much greater depth across several chapters."
    Annotator note: Incorrect claim: "Rebase permanently deletes the branch you rebase onto."
