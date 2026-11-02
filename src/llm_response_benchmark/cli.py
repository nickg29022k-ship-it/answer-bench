"""Command-line interface (`respbench`)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .agreement import agreement_report, interpret_alpha
from .dataset import Dataset
from .disagreement import DisagreementConfig, adjudication_queue, calibrate, flag_items
from .llm import BackendError, get_backend
from .schemas import DIMENSIONS, write_json, write_jsonl

app = typer.Typer(add_completion=False, help="LLM response evaluation benchmark.", no_args_is_help=True)
console = Console()

DATA = typer.Option(Path("data"), "--data", "-d", help="Benchmark data directory.")


def _load(data: Path) -> Dataset:
    try:
        return Dataset.load(data)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Failed to load {data}: {exc}[/red]")
        raise typer.Exit(1)


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


@app.command("generate-data")
def generate_data(
    out: Path = typer.Option(Path("data"), "--out", "-o"),
    seed: int = typer.Option(7, help="RNG seed; the same seed always yields the same files."),
):
    """Generate the synthetic benchmark (prompts, responses, references, annotations, gold)."""
    from .synth import generate

    counts = generate(out, seed=seed)
    for k, v in counts.items():
        console.print(f"wrote {v:4d} records to {out / (k + '.jsonl')}")


@app.command()
def validate(data: Path = DATA, rubric: Optional[Path] = typer.Option(None, help="Rubric YAML (default: <data>/rubric.yaml).")):
    """Schema-validate every JSONL file and check cross-file references."""
    from .rubric import load_rubric

    ds = _load(data)
    try:
        load_rubric(rubric or data / "rubric.yaml")
    except Exception as exc:
        console.print(f"[red]Rubric invalid: {exc}[/red]")
        raise typer.Exit(1)
    problems = ds.validate()
    console.print(
        f"{len(ds.prompts)} prompts, {len(ds.responses)} responses, {len(ds.references)} references, "
        f"{len(ds.annotations)} annotations, {len(ds.gold)} gold labels"
    )
    if problems:
        for p in problems:
            console.print(f"[red]- {p}[/red]")
        raise typer.Exit(1)
    console.print("[green]OK[/green]")


@app.command()
def agreement(data: Path = DATA, out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write JSON here.")):
    """Inter-annotator agreement per dimension (Krippendorff's alpha, Fleiss' and Cohen's kappa)."""
    ds = _load(data)
    rep = agreement_report(ds)
    t = Table(title=f"Agreement ({len(rep['raters'])} annotators)")
    for c in ["dimension", "alpha ord", "reading", "alpha int", "fleiss k", "cohen k", "cohen k quad"]:
        t.add_column(c)
    for d, v in rep["dimensions"].items():
        a = v["krippendorff_alpha_ordinal"]
        t.add_row(d, _fmt(a), interpret_alpha(a), _fmt(v["krippendorff_alpha_interval"]), _fmt(v["fleiss_kappa"]),
                  _fmt(v["mean_cohen_kappa"]), _fmt(v["mean_cohen_kappa_quadratic"]))
    console.print(t)
    if out:
        write_json(out, rep)
        console.print(f"wrote {out}")


def _cfg(range_threshold: float, std_threshold: float, min_dims: int, no_bias_correct: bool) -> DisagreementConfig:
    return DisagreementConfig(range_threshold=range_threshold, std_threshold=std_threshold,
                              min_flagged_dims=min_dims, bias_correct=not no_bias_correct)


@app.command()
def disagreement(
    data: Path = DATA,
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    range_threshold: float = typer.Option(1.75, help="Score range that marks a dimension as disputed."),
    std_threshold: float = typer.Option(0.8, help="Score std that marks a dimension as disputed."),
    min_dims: int = typer.Option(2, help="Disputed dimensions needed to call an item ambiguous."),
    no_bias_correct: bool = typer.Option(False, help="Use raw scores instead of removing annotator bias."),
):
    """Flag ambiguous items and write the adjudication queue."""
    ds = _load(data)
    cfg = _cfg(range_threshold, std_threshold, min_dims, no_bias_correct)
    flagged = flag_items(ds, cfg)
    queue = adjudication_queue(ds, cfg)
    write_json(out / "flagged_items.json", flagged)
    n = write_jsonl(out / "adjudication_queue.jsonl", queue)
    t = Table(title=f"Flagged items ({len(flagged)} of {len(ds.responses)})")
    for c in ["#", "response", "model", "label", "priority", "disputed dimensions"]:
        t.add_column(c)
    for q in queue[:15]:
        t.add_row(str(q["queue_rank"]), q["response_id"], q["model"], q["label"], f"{q['priority']:.2f}",
                  ", ".join(d["dimension"] for d in q["disputed"]))
    console.print(t)
    console.print(f"wrote {n} tasks to {out / 'adjudication_queue.jsonl'}")


@app.command("calibrate")
def calibrate_cmd(data: Path = DATA, out: Optional[Path] = typer.Option(None, "--out", "-o")):
    """Per-annotator bias, spread and gold agreement, with lenient/harsh flags."""
    ds = _load(data)
    rep = calibrate(ds)
    t = Table(title="Annotator calibration")
    for c in ["annotator", "mean", "bias", "spread", "corr", "gold MAE", "flags"]:
        t.add_column(c)
    for a, v in rep["annotators"].items():
        o = v["overall"]
        t.add_row(a, _fmt(o["mean_score"], 2), f"{o['bias']:+.2f}", _fmt(o["spread_ratio"], 2), _fmt(o["corr"], 2),
                  _fmt(o.get("gold_mae"), 2), ", ".join(v["flags"]) or "ok")
    console.print(t)
    for a, v in rep["annotators"].items():
        if v["flags"]:
            console.print(f"[yellow]{a}[/yellow]: {v['recommendation']}")
    if out:
        write_json(out, rep)
        console.print(f"wrote {out}")


@app.command()
def compare(
    data: Path = DATA,
    n_boot: int = typer.Option(2000, help="Bootstrap resamples."),
    seed: int = 0,
    bias_correct: bool = typer.Option(False, help="Remove annotator bias before averaging."),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
):
    """Per-dimension model means with bootstrap CIs and pairwise win rates."""
    from .compare import compare_models, win_matrix
    from .rubric import load_rubric

    ds = _load(data)
    rubric_path = data / "rubric.yaml"
    weights = load_rubric(rubric_path).weights if rubric_path.exists() else None
    comp = compare_models(ds, n_boot=n_boot, seed=seed, bias_correct=bias_correct, weights=weights)
    t = Table(title=f"Model comparison ({comp['n_prompts']} prompts, 95% bootstrap CI)")
    t.add_column("dimension")
    for m in comp["models"]:
        t.add_column(m)
    for d in list(DIMENSIONS) + ["overall"]:
        t.add_row(d, *[f"{s['mean']:.2f} [{s['ci_low']:.2f}, {s['ci_high']:.2f}]" for s in (comp["dimensions"][d][m] for m in comp["models"])])
    console.print(t)
    wm = win_matrix(comp)
    w = Table(title="Overall win rate (row beats column)")
    w.add_column("")
    for m in comp["models"]:
        w.add_column(m)
    for a in comp["models"]:
        w.add_row(a, *["-" if a == b else f"{wm[a][b]:.2f}" for b in comp["models"]])
    console.print(w)
    if out:
        write_json(out, comp)
        console.print(f"wrote {out}")


@app.command()
def annotate(
    data: Path = DATA,
    backend: str = typer.Option("mock", "--backend", "-b", help="mock | ollama | anthropic | openai"),
    out: Path = typer.Option(Path("out/auto_annotations.jsonl"), "--out", "-o"),
    limit: Optional[int] = typer.Option(None, help="Only judge the first N responses."),
):
    """Score responses with an LLM judge and compare it to the human consensus."""
    from .auto_annotator import annotate as run_annotate
    from .auto_annotator import judge_vs_humans
    from .rubric import load_rubric

    ds = _load(data)
    try:
        be = get_backend(backend)
        anns, errors = run_annotate(ds, be, load_rubric(data / "rubric.yaml"), limit=limit)
    except BackendError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    write_jsonl(out, anns)
    console.print(f"wrote {len(anns)} annotations to {out} ({len(errors)} parse errors)")
    for e in errors[:5]:
        console.print(f"[yellow]{e}[/yellow]")
    rep = judge_vs_humans(ds, anns)
    t = Table(title=f"{be.name} judge vs human median ({rep['n']} responses)")
    for c in ["dimension", "alpha", "kappa quad", "MAE", "exact"]:
        t.add_column(c)
    for d, v in rep["dimensions"].items():
        t.add_row(d, _fmt(v["alpha_ordinal_vs_median"]), _fmt(v["kappa_quadratic"]), _fmt(v["mae"], 2), _fmt(v["exact"], 2))
    console.print(t)


@app.command("run-all")
def run_all(
    data: Path = DATA,
    out: Path = typer.Option(Path("examples"), "--out", "-o"),
    backend: str = typer.Option("mock", "--backend", "-b", help="Judge backend, or 'none' to skip the automated annotator."),
    n_boot: int = 2000,
    seed: int = 0,
):
    """Run the whole pipeline and write JSON outputs plus report.md."""
    from .pipeline import run_pipeline

    try:
        summary = run_pipeline(data, out, backend=None if backend == "none" else backend, n_boot=n_boot, seed=seed)
    except BackendError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"ranking: {' > '.join(summary['ranking'])}")
    console.print(f"flagged {summary['flagged']} items; adjudication queue has {summary['queue']} tasks")
    console.print(f"outputs written to {summary['out_dir']}/ (report.md, *.json, adjudication_queue.jsonl, feedback/)")


if __name__ == "__main__":
    app()
