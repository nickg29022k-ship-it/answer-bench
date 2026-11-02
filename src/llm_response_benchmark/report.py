"""Markdown report rendering."""

from __future__ import annotations

import math

import numpy as np

from .agreement import interpret_alpha
from .compare import win_matrix
from .schemas import DIMENSIONS


def _f(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_report(ctx: dict) -> str:
    ds_info = ctx["dataset"]
    comp = ctx["comparison"]
    agree = ctx["agreement"]
    calib = ctx["calibration"]
    flagged = ctx["flagged"]
    queue = ctx["queue"]
    feedback = ctx["feedback"]
    models = comp["models"]
    out: list[str] = []
    w = out.append

    w("# LLM Response Evaluation Report\n")
    w(
        f"Dataset: {ds_info['prompts']} prompts, {ds_info['responses']} candidate responses from "
        f"{len(models)} models ({', '.join(models)}), {ds_info['annotations']} human annotations by "
        f"{len(ds_info['annotators'])} annotators ({', '.join(ds_info['annotators'])}), "
        f"{ds_info['gold']} gold-labelled responses. Scores are 1-5 on six rubric dimensions "
        "(higher is better; for `hallucination`, higher means fewer fabrications).\n"
    )

    # ---------------------------------------------------------------- leaderboard
    w("## 1. Model comparison\n")
    w(
        f"Mean consensus score per dimension (mean of annotators per response, then mean over "
        f"{comp['n_prompts']} prompts) with 95% bootstrap CIs ({comp['n_boot']} resamples over prompts). "
        "`overall` is the rubric-weighted mean of the six dimensions.\n"
    )
    header = ["Dimension"] + models
    rows = []
    for d in list(DIMENSIONS) + ["overall"]:
        row = [f"**{d}**" if d == "overall" else d]
        best = max(models, key=lambda m: comp["dimensions"][d][m]["mean"])
        for m in models:
            s = comp["dimensions"][d][m]
            cell = f"{s['mean']:.2f} [{s['ci_low']:.2f}, {s['ci_high']:.2f}]"
            row.append(f"**{cell}**" if m == best else cell)
        rows.append(row)
    w(_table(header, rows) + "\n")
    w(f"Ranking by overall score: {' > '.join(comp['ranking'])}.\n")

    w("### Win rates (overall score)\n")
    w(
        f"Share of prompts where the row model beats the column model; differences smaller than "
        f"{comp['tie_margin']} points count as ties (half a win).\n"
    )
    wm = win_matrix(comp, "overall")
    w(_table(["", *models], [[a] + ["-" if a == b else _f(wm[a][b]) for b in models] for a in models]) + "\n")

    sig = [x for x in comp["win_rates"] if x["significant"] and x["dimension"] != "overall"]
    if sig:
        w("Per-dimension differences whose paired 95% bootstrap CI excludes zero:\n")
        for x in sorted(sig, key=lambda x: -abs(x["mean_diff"])):
            better, worse = (x["model_a"], x["model_b"]) if x["mean_diff"] > 0 else (x["model_b"], x["model_a"])
            w(
                f"- `{x['dimension']}`: {better} > {worse} by {abs(x['mean_diff']):.2f} "
                f"(CI [{_f(min(abs(x['diff_ci_low']), abs(x['diff_ci_high'])))}, {_f(max(abs(x['diff_ci_low']), abs(x['diff_ci_high'])))}])"
            )
        w("")

    w("### Overall score by prompt category\n")
    cats = comp["by_category"]
    w(_table(["Category", *models], [[c] + [_f(cats[c][m]) for m in models] for c in cats]) + "\n")

    # ---------------------------------------------------------------- agreement
    w("## 2. Inter-annotator agreement\n")
    w(
        "Krippendorff's alpha uses the ordinal distance metric; Fleiss' kappa treats scores as nominal "
        "categories; Cohen's kappa is averaged over annotator pairs (quadratic weights). Rule of thumb for "
        "alpha: >= 0.80 reliable, 0.67-0.80 tentative, < 0.67 weak. `alpha (bias-corr.)` is interval alpha after "
        "subtracting each annotator's estimated offset; the gap to the raw alpha is the agreement lost to "
        "systematic leniency or harshness rather than to genuinely ambiguous items.\n"
    )
    rows = []
    for d in DIMENSIONS:
        a = agree["dimensions"][d]
        pairs = a["pairwise"]
        rows.append([
            d,
            _f(a["krippendorff_alpha_ordinal"], 3),
            interpret_alpha(a["krippendorff_alpha_ordinal"]),
            _f(a["krippendorff_alpha_interval_bias_corrected"], 3),
            _f(a["fleiss_kappa"], 3),
            _f(a["mean_cohen_kappa_quadratic"], 3),
            _f(np.mean([p["exact"] for p in pairs])),
            _f(np.mean([p["within_1"] for p in pairs])),
        ])
    w(_table(["Dimension", "alpha (ordinal)", "reading", "alpha (bias-corr.)", "Fleiss kappa", "Cohen kappa (quad.)", "exact agree", "within 1"], rows) + "\n")

    w("### Pairwise annotator agreement (averaged over dimensions)\n")
    pair_rows: dict[tuple[str, str], list[dict]] = {}
    for d in DIMENSIONS:
        for p in agree["dimensions"][d]["pairwise"]:
            pair_rows.setdefault((p["rater_a"], p["rater_b"]), []).append(p)
    rows = []
    for (a, b), ps in pair_rows.items():
        rows.append([
            f"{a} vs {b}",
            _f(np.mean([p["exact"] for p in ps])),
            _f(np.mean([p["within_1"] for p in ps])),
            _f(np.mean([p["mean_signed_diff"] for p in ps])),
            _f(np.mean([p["cohen_kappa"] for p in ps]), 3),
            _f(np.mean([p["cohen_kappa_quadratic"] for p in ps]), 3),
        ])
    w(_table(["Pair", "exact", "within 1", "mean diff (a-b)", "kappa", "kappa (quad.)"], rows) + "\n")

    # ---------------------------------------------------------------- calibration
    w("## 3. Annotator calibration\n")
    w(
        "`bias` comes from an additive model (score = item quality + annotator offset), centred on the median "
        "annotator. `spread` is the ratio of the annotator's score std to the consensus std (below 1 means a "
        "compressed scale). `corr` is the correlation with the leave-one-out consensus. Gold columns compare "
        "against expert-adjudicated labels.\n"
    )
    rows = []
    for a, c in calib["annotators"].items():
        o = c["overall"]
        rows.append([
            a, _f(o["mean_score"]), _f(o["bias"], 2), _f(o["spread_ratio"]), _f(o["corr"]),
            _f(o.get("gold_bias")), _f(o.get("gold_mae")), _f(o.get("gold_exact")),
            ", ".join(c["flags"]) or "ok",
        ])
    w(_table(["Annotator", "mean", "bias", "spread", "corr", "gold bias", "gold MAE", "gold exact", "flags"], rows) + "\n")
    for a, c in calib["annotators"].items():
        if c["flags"]:
            w(f"- **{a}**: {c['recommendation']}")
            if c["dimension_flags"]:
                w("  Dimension flags: " + "; ".join(f"{d}: {', '.join(f)}" for d, f in c["dimension_flags"].items()))
    w("")

    # ---------------------------------------------------------------- disagreement
    w("## 4. Disagreement workflow\n")
    labels: dict[str, int] = {}
    for f in flagged:
        labels[f["label"]] = labels.get(f["label"], 0) + 1
    w(
        f"{len(flagged)} of {ds_info['responses']} responses were flagged after removing each annotator's bias: "
        + ", ".join(f"{v} {k}" for k, v in sorted(labels.items()))
        + ". A dimension is disputed when the bias-corrected scores span at least "
        f"{ctx['config']['range_threshold']} points or their std is at least {ctx['config']['std_threshold']}; "
        f"an item is `ambiguous` when {ctx['config']['min_flagged_dims']}+ dimensions are disputed or one "
        f"dimension spans {ctx['config']['severe_range']}+ points. All flagged items go to "
        "`adjudication_queue.jsonl`, most contentious first.\n"
    )
    if ctx.get("ambiguity_recovery"):
        r = ctx["ambiguity_recovery"]
        w(
            f"Synthetic sanity check: the generator planted {r['planted']} ambiguous responses. "
            f"Items labelled `ambiguous`: precision {_f(r['precision'])}, recall {_f(r['recall'])}. "
            f"Any flag: recall {_f(r['recall_any_flag'])}.\n"
        )
    w("Top of the adjudication queue:\n")
    rows = []
    for q in queue[:10]:
        disputes = "; ".join(
            f"{d['dimension']} {'/'.join(str(v) for v in d['scores'].values())}" for d in q["disputed"]
        )
        rows.append([q["queue_rank"], q["response_id"], q["label"], _f(q["priority"]), disputes])
    w(_table(["#", "Response", "Label", "Priority", "Disputed dims (scores by annotator)"], rows) + "\n")
    if queue:
        q = queue[0]
        w(f"Example task `{q['response_id']}` ({q['model']}). Prompt: \"{q['prompt']}\"\n")
        for d in q["disputed"][:3]:
            w(f"- `{d['dimension']}`:")
            for a, s in d["scores"].items():
                w(f"  - {a} gave {s}: {d['rationales'].get(a, '')}")
        w("")

    # ---------------------------------------------------------------- judge
    if ctx.get("judge"):
        j = ctx["judge"]
        w(f"## 5. Automated annotator vs human consensus ({j['backend']} backend)\n")
        if j["backend"] == "mock":
            w(
                "The mock judge is a deterministic heuristic (fact overlap, false-claim matching, constraint checks). "
                "On this synthetic data it reads the same signals the generator planted, so its agreement is an upper "
                "bound, not evidence of real judge quality.\n"
            )
        rows = [
            [d, _f(v["alpha_ordinal_vs_median"], 3), _f(v["kappa_quadratic"], 3), _f(v["mae"]), _f(v["exact"]), _f(v["mean_bias"])]
            for d, v in j["dimensions"].items()
        ]
        w(_table(["Dimension", "alpha vs median", "kappa (quad.)", "MAE", "exact", "bias (judge-human)"], rows) + "\n")

    # ---------------------------------------------------------------- feedback
    w(f"## {6 if ctx.get('judge') else 5}. Per-model feedback\n")
    for m in sorted(feedback, key=lambda k: feedback[k]["rank"]):
        fb = feedback[m]
        o = fb["overall"]
        w(f"### {m} (rank {fb['rank']}, overall {o['mean']:.2f} [{o['ci_low']:.2f}, {o['ci_high']:.2f}])\n")
        c = fb["counts"]
        w(
            f"Responses: {c['responses']}; likely hallucinations: {c['likely_hallucinations']}; "
            f"instruction violations: {c['instruction_violations']}; flagged for adjudication: {c['flagged_for_adjudication']}. "
            "Win rate vs " + ", ".join(f"{k}: {v:.2f}" for k, v in fb["win_rate_vs"].items()) + ".\n"
        )
        w("**Strengths**\n")
        if fb["strengths"]:
            for s in fb["strengths"]:
                w(f"- `{s['dimension']}` ({s['mean']:.2f}, {s['delta_vs_others']:+.2f} vs others): {s['summary']}")
        else:
            w("- No dimension stands out above the other models.")
        w("\n**Weaknesses**\n")
        if fb["weaknesses"]:
            for s in fb["weaknesses"]:
                w(f"- `{s['dimension']}` ({s['mean']:.2f}, {s['delta_vs_others']:+.2f} vs others): {s['advice']}")
                for ex in s["examples"][:1]:
                    rats = [r for r in ex["annotator_rationales"].values() if r]
                    w(f"  - Example `{ex['response_id']}` (score {ex['consensus_score']}): \"{ex['response_excerpt']}\"")
                    if rats:
                        w(f"    Annotator note: {rats[0]}")
        else:
            w("- No dimension falls clearly below the other models.")
        w("")
    return "\n".join(out).rstrip() + "\n"
