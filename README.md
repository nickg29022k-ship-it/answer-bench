# LLM Response Evaluation Benchmark

A small, reusable benchmark plus tooling for evaluating LLM-generated answers with human
annotations. It contains:

- JSONL datasets of prompts, candidate responses from several models, reference labels and
  multi-annotator human scores, all validated with pydantic.
- A YAML rubric with six dimensions (factual accuracy, relevance, completeness, instruction
  following, hallucination, style), each with anchored 1-5 score descriptions.
- Inter-annotator agreement metrics (Krippendorff's alpha, Cohen's and Fleiss' kappa, pairwise
  agreement), implemented from scratch with numpy and checked against published examples.
- A disagreement workflow that flags ambiguous items, writes an adjudication queue, and calibrates
  annotators (bias, scale compression, agreement with gold labels).
- Model comparison with bootstrap confidence intervals and pairwise win rates.
- Structured per-model feedback as JSON and a Markdown report.
- An optional automated annotator (LLM-as-judge) behind a pluggable backend: a deterministic
  mock by default, or Ollama, Anthropic or OpenAI over plain HTTP.

The checked-in dataset is synthetic (40 prompts, 3 models, 3 annotators, 360 annotations). It was
generated so that the interesting cases exist: some prompts are deliberately ambiguous, one
annotator is systematically lenient, and the three models have different failure profiles.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run respbench validate                 # schema + cross-reference checks on data/
uv run respbench run-all                  # full pipeline -> examples/ (mock judge)
uv run pytest                             # test suite
```

Individual steps:

```bash
uv run respbench generate-data --out data --seed 7     # regenerate the synthetic benchmark
uv run respbench agreement                             # agreement table per dimension
uv run respbench calibrate                             # annotator bias / spread / flags
uv run respbench disagreement --out out                # flagged items + adjudication_queue.jsonl
uv run respbench compare --n-boot 2000                 # model means, CIs, win rates
uv run respbench annotate --backend mock --out out/auto_annotations.jsonl
uv run respbench run-all --data data --out examples --backend mock   # or --backend none
```

Every command takes `--data/-d` (default `data/`). `respbench <command> --help` lists the options,
for example the disagreement thresholds or `--bias-correct` for `compare`.

## Repository layout

```
data/
  rubric.yaml              six dimensions, weights, anchored score descriptions
  prompts.jsonl            40 prompts with machine-checkable constraints
  responses.jsonl          120 candidate responses (3 models x 40 prompts)
  references.jsonl         reference answer, key facts, known false claims per prompt
  annotations.jsonl        360 human annotations (3 annotators x 120 responses)
  gold.jsonl               expert scores for 24 responses (used for calibration)
  generation_truth.jsonl   latent truth from the generator (only used for the sanity check)
examples/                  committed output of `respbench run-all` on the mock backend
src/llm_response_benchmark/
  schemas.py               pydantic record models and JSONL I/O
  dataset.py               loading, cross-file validation, score matrices
  rubric.py                rubric loader / validator
  agreement.py             Krippendorff's alpha, Cohen's kappa, Fleiss' kappa, pairwise stats
  disagreement.py          bias model, ambiguity flags, adjudication queue, calibration
  compare.py               bootstrap CIs and win rates
  feedback.py              per-model strengths / weaknesses / examples
  report.py                Markdown report
  llm.py                   backend abstraction (mock, ollama, anthropic, openai)
  auto_annotator.py        LLM judge prompt, output parsing, judge-vs-human agreement
  synth.py, seeds.py       synthetic data generator and its hand-written seed content
  textutils.py             tokenizing, word counts, constraint checker
  pipeline.py, cli.py      end-to-end orchestration and the typer CLI
tests/                     pytest suite
```

## Data model

All files are JSONL, one record per line, validated by the models in `schemas.py`
(unknown fields are rejected).

| Record | Key fields |
|---|---|
| `Prompt` | `prompt_id`, `category`, `text`, `constraints` (`max_words`, `format` = any/paragraph/bullets, `num_bullets`) |
| `CandidateResponse` | `response_id`, `prompt_id`, `model`, `text` |
| `ReferenceLabel` | `prompt_id`, `reference_answer`, `key_facts[]`, `known_false_claims[]` |
| `Annotation` | `annotator_id`, `response_id`, `scores` (all six dimensions, int 1-5), `rationales` (per dimension), `source` = human/auto |
| `GoldLabel` | `response_id`, `scores`, `adjudicator` |

`Dataset.validate()` (and `respbench validate`) also checks for duplicate ids, dangling
references, prompts without a reference label, and responses with fewer than two annotations.

## Pipeline

```
prompts / responses / references / annotations / gold
        |
        +--> agreement.py     per-dimension alpha, kappas, pairwise agreement
        +--> disagreement.py  annotator bias model --> bias-corrected spreads
        |                       --> flagged items --> adjudication_queue.jsonl
        |                       --> calibration report (flags + recommendations)
        +--> compare.py       consensus score per response --> bootstrap CIs, win rates
        +--> feedback.py      strengths / weaknesses / example items per model
        +--> auto_annotator   (optional) LLM judge --> judge-vs-human agreement
        v
   report.md + JSON files
```

## Metric definitions

Scores form a matrix with one row per annotator and one column per response; missing ratings are NaN.

**Krippendorff's alpha.** `alpha = 1 - D_o / D_e`. Build the coincidence matrix `o[c,k]`: for
every response with `m >= 2` ratings, each ordered pair of ratings from different annotators adds
`1 / (m - 1)`. With marginals `n_c` and total `n`, `D_o = sum o[c,k] * d(c,k) / n` and
`D_e = sum n_c * n_k * d(c,k) / (n (n - 1))`. Distances: nominal `d = [c != k]`, interval
`d = (c - k)^2`, ordinal `d = (sum_{g=c..k} n_g - (n_c + n_k) / 2)^2`. Ordinal is the headline
number because rubric scores are ordered but not necessarily equally spaced. Missing data is
handled naturally. The implementation reproduces the worked example in Krippendorff (2011):
0.743 nominal, 0.815 ordinal, 0.849 interval.

**Bias-corrected alpha.** Interval alpha after subtracting each annotator's estimated offset
(see annotator bias below). The gap between this and raw alpha is agreement lost to systematic
leniency or harshness rather than to genuinely hard items.

**Cohen's kappa** (pairs of annotators). `kappa = 1 - sum(w * O) / sum(w * E)`, with `O` the
observed joint proportions, `E` the outer product of the two marginals, and `w` the disagreement
weights: `[i != j]` (unweighted), `|i - j| / (K - 1)` (linear) or `((i - j) / (K - 1))^2`
(quadratic). Unweighted is the familiar `(p_o - p_e) / (1 - p_e)`. Reported per pair and averaged.

**Fleiss' kappa** (all annotators, nominal). With `n` ratings per item and `n_ij` raters putting
item `i` in category `j`: `P_i = (sum_j n_ij^2 - n) / (n (n - 1))`, `p_j = sum_i n_ij / (N n)`,
`kappa = (mean(P_i) - sum p_j^2) / (1 - sum p_j^2)`. Items with missing ratings are dropped.

**Pairwise agreement.** For each annotator pair: exact-match rate, within-one-point rate, mean
absolute difference and mean signed difference.

**Annotator bias.** An additive model `score[a, i] = quality[i] + bias[a]` fitted by alternating
least squares per dimension, then centred so the median annotator has bias 0. Median centring
matters: with a plain leave-one-out comparison, one lenient annotator makes everyone else look harsh.

**Calibration stats.** Per annotator and dimension: `bias` (above), `loo_bias` and `mae` against
the mean of the other annotators, `spread_ratio = std(own scores) / std(leave-one-out consensus)`
and the correlation with that consensus. If gold labels exist: `gold_bias`, `gold_mae`,
`gold_exact`. Flags: `lenient` (bias >= 0.3), `harsh` (bias <= -0.3), `compressed_range`
(spread ratio < 0.75), `noisy` (correlation < 0.6). Thresholds live in `DisagreementConfig`.

**Ambiguity flags.** After removing annotator bias, a (response, dimension) pair is *disputed* if
the score range is >= 1.75 or the std is >= 0.8. A response is labelled `ambiguous` if two or more
dimensions are disputed or one dimension spans >= 2.75 points; `outlier_annotator` if a single
disputed dimension is explained by one annotator who is >= 2 points from the other two (who agree
within 1); otherwise `dimension_dispute`. Priority = sum of disputed ranges + 2 x mean std across
dimensions (+2 for `ambiguous`). The adjudication queue is sorted by priority, and each task
carries every annotator's score and rationale for the disputed dimensions, plus
`status`/`resolution` fields for the adjudicator to fill in.

**Consensus and model comparison.** The consensus score of a response on a dimension is the
mean over annotators (optionally bias-corrected). `overall` is the rubric-weighted mean of the six
dimensions. 95% CIs use a percentile bootstrap over prompts (2000 resamples). The same resampled
prompt indices are used for every model, so differences between models are paired; a difference
is reported as significant when its paired CI excludes zero.

**Win rate.** For models A and B on a dimension, `(wins + 0.5 * ties) / n_prompts`, where a
difference smaller than 0.25 points counts as a tie.

## Sample output

From `examples/report.md` (mock backend, checked-in data), abridged:

```
| Dimension | atlas-v2 | borealis-mini | cirrus-large |
|---|---|---|---|
| factual_accuracy | 4.47 [4.17, 4.73] | 3.68 [3.35, 4.01] | 4.45 [4.17, 4.70] |
| completeness | 4.47 [4.21, 4.71] | 3.18 [2.88, 3.49] | 3.74 [3.44, 4.06] |
| hallucination | 4.48 [4.10, 4.81] | 3.72 [3.25, 4.16] | 4.56 [4.27, 4.80] |
| style | 3.94 [3.75, 4.12] | 4.32 [4.09, 4.52] | 4.82 [4.71, 4.91] |
| overall | 4.45 [4.32, 4.57] | 3.92 [3.73, 4.09] | 4.47 [4.34, 4.60] |

| Dimension | alpha (ordinal) | reading | alpha (bias-corr.) | Fleiss kappa |
|---|---|---|---|---|
| factual_accuracy | 0.688 | tentative | 0.817 | 0.320 |
| relevance | 0.558 | weak | 0.596 | 0.261 |
| hallucination | 0.780 | tentative | 0.890 | 0.377 |
| style | 0.388 | weak | 0.542 | 0.192 |

| Annotator | mean | bias | spread | corr | gold bias | gold MAE | flags |
|---|---|---|---|---|---|---|---|
| ann-01 | 4.17 | -0.00 | 1.32 | 0.82 | -0.01 | 0.05 | ok |
| ann-02 | 4.62 | 0.44 | 0.62 | 0.76 | 0.39 | 0.39 | lenient, compressed_range |
| ann-03 | 4.07 | -0.11 | 1.40 | 0.80 | -0.17 | 0.24 | ok |

37 of 120 responses were flagged after removing each annotator's bias:
18 ambiguous, 16 dimension_dispute, 3 outlier_annotator.
Synthetic sanity check: the generator planted 18 ambiguous responses.
Items labelled `ambiguous`: precision 0.78, recall 0.78. Any flag: recall 1.00.
```

Other files in `examples/`: `comparison.json`, `agreement.json`, `calibration.json`,
`flagged_items.json`, `adjudication_queue.jsonl`, `feedback/<model>.json`,
`auto_annotations.jsonl`, `judge_agreement.json`.

## Switching LLM backends

The automated annotator only needs `get_backend(name).complete(prompt, system=None, temperature=0.0)`.

| Backend | Selection | Configuration |
|---|---|---|
| `mock` (default) | `--backend mock` | none; deterministic, no network |
| `ollama` | `--backend ollama` | `OLLAMA_MODEL` (default `llama3.1:8b`), `OLLAMA_HOST` (default `http://localhost:11434`) |
| `anthropic` | `--backend anthropic` | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` (default `claude-haiku-4-5-20251001`) |
| `openai` | `--backend openai` | `OPENAI_API_KEY`, optional `OPENAI_MODEL` (default `gpt-4o-mini`), `OPENAI_BASE_URL` |

```bash
export ANTHROPIC_API_KEY=...
uv run respbench annotate --backend anthropic --limit 20
uv run respbench run-all --backend ollama --out out/ollama_run   # needs `ollama serve` running
```

Network backends use `httpx` directly (no vendor SDKs). A missing key or an unreachable server
raises a `BackendError` that says what to fix, and the CLI exits with code 2. The judge prompt
contains the rubric anchors, the prompt and its constraints, the reference key facts and known
false claims, and the response, and asks for JSON. The parser accepts JSON wrapped in prose or
code fences and clips scores into 1-5.

The mock backend scores responses heuristically: token overlap with key facts (completeness),
matching known false claims (accuracy, hallucination), the same constraint checker used elsewhere
(instruction following), and phrase lists for padding (style). A hash of the input adds
deterministic jitter to about 15% of judgements, so it behaves like a slightly imperfect judge
while always giving the same output for the same input.

## How the synthetic data is built

`synth.py` turns 40 hand-written seeds (`seeds.py`: prompt, three key facts, one plausible false
claim, constraints) into responses by sampling, per model profile, which facts to include,
whether to insert the false claim, add filler or an off-topic sentence, wrap the answer in chatty
pleasantries, or break the requested format:

- `atlas-v2`: high recall, chatty, often ignores format instructions.
- `borealis-mini`: misses facts and hallucinates most often.
- `cirrus-large`: concise and obedient, sometimes incomplete.

Latent rubric scores follow from those planted attributes. Each annotator then scores
`latent + bias + noise`, rounded and clipped to 1-5: `ann-01` is well calibrated, `ann-02` is
lenient (a positive offset that grows as quality drops, which also compresses their scale), and
`ann-03` is noisier. Six prompts are marked ambiguous (underspecified or subjective, such as "Tell
me about Mercury." or "What's the best programming language?"); their annotations get extra noise
on subjective dimensions plus one deliberate split decision per response. Generation is seeded,
and a test checks that the checked-in files match a fresh run.

## Limitations

- The dataset is synthetic. Responses are assembled from reference sentences, so they read more
  uniformly than real model output, and the annotator rationales are templated.
- The mock judge's high agreement with humans is partly by construction: it reads the same signals
  the generator planted. It shows the plumbing works, not that a heuristic judge is good.
- The network backends are written against the public HTTP APIs, but the test suite only exercises
  their failure paths (missing key, unreachable server), not live calls.
- With three annotators per item, spread-based ambiguity detection is noisy, and the default
  thresholds were tuned on this synthetic data.
- Scores cluster near the top of the scale, which depresses chance-corrected statistics (Fleiss'
  kappa especially) even when raw agreement is high.
