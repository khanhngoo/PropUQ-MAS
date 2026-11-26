#!/usr/bin/env python3
"""Fill the Table 1 report from PropUQ-MAS metrics files.

Reads the UQ outputs written by run.py for the three generation runs of each
(model, dataset, topology) cell and writes:

  * a Markdown report: paper vs ours per model (MSP, Verb, NormMSP-Answer),
    the NormMSP step-by-step ablation, and per-run diagnostics;
  * a long-format CSV with every metrics row, for later analysis.

Runs (see experiment_plans/2026-11-26_propuq_setup_comparison.md):
  R1  Verb                              -> outputs/Verb/
  R2  NormMSP                           -> outputs/NormMSP/
  R3  NormMSP --no_alpha_elicitation    -> outputs/NormMSP_noalpha/

Setups:
  S1 original code      Verb: R1 Normalized          MSP: R3 Alpha1-Normalized
  S2 paper rule         Verb: R1 Prop                MSP: R2 Prop
  S3 alpha + normalized Verb: R1 Normalized          MSP: R2 Normalized

Missing runs are shown as "–", so the script can be run at any point.
Standard library only; does not need a GPU, torch or vLLM.

Usage (from the repository root):
  python analysis/make_table1.py
  python analysis/make_table1.py --models Qwen/Qwen3-8B --tasks medqa
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "google/gemma-3-12b-it"]
MODEL_LABELS = {"Qwen/Qwen3-4B": "Qwen3-4B", "Qwen/Qwen3-8B": "Qwen3-8B", "google/gemma-3-12b-it": "Gemma-3-12B"}
TASKS = ["gsm8k", "mbppplus", "medqa"]
TASK_LABELS = {"gsm8k": "GSM8K", "mbppplus": "MBPP+", "medqa": "MedQA"}
TOPOLOGIES = ["sequential", "hierarchical", "decentralized"]
METRICS = ["auroc", "prr"]

RUNS = {
    "R1": {"folder": "Verb", "desc": "Verb"},
    "R2": {"folder": "NormMSP", "desc": "NormMSP"},
    "R3": {"folder": "NormMSP_noalpha", "desc": "NormMSP no-α̂"},
}

LOCAL = "Local-Uncertainty-Final"
PROP = "Prop-Uncertainty-Final"
PROP_NORM = "Prop-Uncertainty-Final-Normalized"
ALPHA1 = "Prop-Uncertainty-Final-Alpha1"
ALPHA1_NORM = "Prop-Uncertainty-Final-Alpha1-Normalized"

# Paper Table 1, checked against the PDF text layer. Per row, 12 values:
# for Qwen3-4B, Qwen3-8B, Gemma-3-12B in turn: MSP, MSP+PropUQ, Verb, Verb+PropUQ.
_PAPER_TABLE1 = {
    ("sequential", "gsm8k", "auroc"): "0.578 0.617 0.466 0.501 0.605 0.625 0.505 0.680 0.471 0.517 0.581 0.651",
    ("sequential", "gsm8k", "prr"): "0.160 0.238 -0.073 -0.068 0.224 0.257 -0.028 0.340 0.005 0.102 0.156 0.250",
    ("sequential", "mbppplus", "auroc"): "0.679 0.722 0.577 0.607 0.583 0.648 0.562 0.577 0.544 0.583 0.684 0.742",
    ("sequential", "mbppplus", "prr"): "0.340 0.378 0.090 0.173 0.155 0.282 0.064 0.120 0.079 0.128 0.277 0.465",
    ("sequential", "medqa", "auroc"): "0.649 0.742 0.428 0.607 0.754 0.842 0.600 0.791 0.635 0.668 0.627 0.674",
    ("sequential", "medqa", "prr"): "0.254 0.410 -0.167 0.236 0.496 0.624 0.217 0.482 0.232 0.240 0.237 0.322",
    ("hierarchical", "gsm8k", "auroc"): "0.564 0.541 0.535 0.597 0.664 0.717 0.516 0.577 0.535 0.498 0.549 0.606",
    ("hierarchical", "gsm8k", "prr"): "0.100 0.094 0.082 0.222 0.321 0.446 -0.001 0.153 0.063 0.060 0.080 0.204",
    ("hierarchical", "mbppplus", "auroc"): "0.688 0.718 0.632 0.661 0.619 0.719 0.580 0.630 0.630 0.674 0.682 0.730",
    ("hierarchical", "mbppplus", "prr"): "0.335 0.432 0.201 0.289 0.172 0.401 0.081 0.245 0.204 0.207 0.382 0.423",
    ("hierarchical", "medqa", "auroc"): "0.628 0.760 0.621 0.705 0.658 0.711 0.604 0.722 0.567 0.604 0.660 0.669",
    ("hierarchical", "medqa", "prr"): "0.238 0.417 0.216 0.296 0.277 0.349 0.223 0.362 0.115 0.184 0.358 0.351",
    ("decentralized", "gsm8k", "auroc"): "0.657 0.704 0.544 0.574 0.728 0.768 0.486 0.563 0.563 0.567 0.634 0.665",
    ("decentralized", "gsm8k", "prr"): "0.311 0.428 0.128 0.185 0.445 0.534 -0.004 0.181 0.128 0.162 0.271 0.294",
    ("decentralized", "mbppplus", "auroc"): "0.695 0.710 0.588 0.604 0.701 0.726 0.511 0.583 0.557 0.645 0.670 0.720",
    ("decentralized", "mbppplus", "prr"): "0.382 0.426 0.140 0.210 0.332 0.367 0.014 0.117 0.096 0.262 0.342 0.393",
    ("decentralized", "medqa", "auroc"): "0.749 0.824 0.693 0.731 0.812 0.821 0.718 0.854 0.599 0.636 0.576 0.611",
    ("decentralized", "medqa", "prr"): "0.416 0.563 0.355 0.337 0.539 0.568 0.429 0.626 0.155 0.208 0.160 0.200",
}
_PAPER_MODEL_ORDER = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "google/gemma-3-12b-it"]


def paper_value(model: str, topo: str, task: str, metric: str, column: str) -> Optional[float]:
    """column in {"msp", "msp_prop", "verb", "verb_prop"}."""
    row = _PAPER_TABLE1.get((topo, task, metric))
    if row is None or model not in _PAPER_MODEL_ORDER:
        return None
    values = [float(v) for v in row.split()]
    offset = 4 * _PAPER_MODEL_ORDER.index(model)
    return values[offset + ["msp", "msp_prop", "verb", "verb_prop"].index(column)]


# ---------------------------------------------------------------- loading


def _stem_glob(task: str, topo: str, seed: int, prompt: str) -> str:
    # run.py adds "_role" only when role prompts are actually used, which is
    # the 4-agent sequential/hierarchical case.
    role = "_role" if prompt == "role" and topo in ("sequential", "hierarchical") else ""
    return f"mas_{task}_{topo}_nodes4{role}_seed{seed}_n*"


def _pick_latest(paths: List[Path]) -> Optional[Path]:
    # Without --force_uq, run.py writes _2, _3, ... copies; take the newest.
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def find_run_files(outputs: Path, run: str, model: str, task: str, topo: str, seed: int, prompt: str):
    uq_dir = outputs / RUNS[run]["folder"] / Path(*model.split("/")) / "uq"
    stem = _stem_glob(task, topo, seed, prompt)
    metrics = _pick_latest(sorted(uq_dir.glob(f"uq_metrics_{stem}.json")))
    preds = _pick_latest(sorted(uq_dir.glob(f"uq_preds_{stem}.jsonl")))
    return metrics, preds


class Cell:
    """The three runs of one (model, task, topology) cell."""

    def __init__(self, outputs: Path, model: str, task: str, topo: str, seed: int, prompt: str):
        self.model, self.task, self.topo = model, task, topo
        self.metrics: Dict[str, dict] = {}
        self.files: Dict[str, Tuple[Optional[Path], Optional[Path]]] = {}
        for run in RUNS:
            m_path, p_path = find_run_files(outputs, run, model, task, topo, seed, prompt)
            self.files[run] = (m_path, p_path)
            if m_path is not None:
                self.metrics[run] = json.loads(m_path.read_text(encoding="utf-8"))

    def value(self, run: str, method: str, row: str, metric: str) -> Optional[float]:
        payload = self.metrics.get(run)
        if payload is None:
            return None
        by_method = payload.get("metrics", {}).get("metrics_by_uq_method", {})
        if method == "NormMSP" and method not in by_method:
            method = "MSP"  # files written before the NormMSP rename
        return (by_method.get(method, {}).get(row) or {}).get(metric)

    def meta(self, run: str, key: str):
        payload = self.metrics.get(run)
        return None if payload is None else payload.get("metadata", {}).get(key)

    def n_scored(self, run: str, method: str) -> Optional[int]:
        payload = self.metrics.get(run)
        if payload is None:
            return None
        rows = payload.get("metrics", {}).get("metrics_by_uq_method", {}).get(method, {})
        return (rows.get(LOCAL) or {}).get("n_with_uncertainty")


def _norm_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def preds_diagnostics(preds_path: Optional[Path]) -> Dict[str, Optional[float]]:
    """alpha report rate over edges, and answer-truncation rate over agent outputs."""
    out = {"alpha_rate": None, "trunc_rate": None}
    if preds_path is None:
        return out
    edges = reported = outputs = truncated = 0
    with preds_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            for agent in pred.get("agents", []):
                incoming = agent.get("incoming_agents") or []
                alpha = {_norm_label(k): v for k, v in (agent.get("alpha") or {}).items()}
                for name in incoming:
                    edges += 1
                    if alpha.get(_norm_label(name)) is not None:
                        reported += 1
                stats = agent.get("logits_uq_stats")
                if isinstance(stats, dict) and "answer_truncated" in stats:
                    outputs += 1
                    truncated += bool(stats["answer_truncated"])
    out["alpha_rate"] = reported / edges if edges else None
    out["trunc_rate"] = truncated / outputs if outputs else None
    return out


# ---------------------------------------------------------------- formatting


def fmt(x: Optional[float]) -> str:
    return "–" if x is None else f"{x:.3f}"


def fmt_pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def fmt_delta(a: Optional[float], b: Optional[float]) -> str:
    return "–" if a is None or b is None else f"{b - a:+.3f}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def table1_rows(cells: Dict[tuple, Cell], model: str, tasks, topos, columns) -> List[List[str]]:
    """columns: list of callables (cell, topo, task, metric) -> str."""
    rows = []
    for topo in topos:
        first_topo = True
        for task in tasks:
            first_task = True
            for metric in METRICS:
                cell = cells[(model, task, topo)]
                rows.append(
                    [topo.capitalize() if first_topo else "", TASK_LABELS[task] if first_task else "", metric.upper()]
                    + [col(cell, topo, task, metric) for col in columns]
                )
                first_topo = first_task = False
    return rows


def build_report(cells, models, tasks, topos, seed) -> str:
    out = [
        "# Table 1 Recreation: Results",
        "",
        f"Generated by `analysis/make_table1.py` (seed {seed}). \"–\" = run not found yet.",
        "",
        "Setups: **S1** original code · **S2** paper rule (raw α̂) · **S3** α̂ elicited + normalized.",
        "Runs: **R1** Verb · **R2** NormMSP · **R3** NormMSP without the α̂ request.",
    ]

    def v(run, method, row):
        return lambda c, t, d, m: fmt(c.value(run, method, row, m))

    def p(column):
        return lambda c, t, d, m: fmt(paper_value(c.model, t, d, m, column))

    for model in models:
        label = MODEL_LABELS.get(model, model)
        out += ["", f"## {label}", "", "### MSP (NormMSP)", ""]
        out.append(md_table(
            ["Topology", "Dataset", "Metric", "Paper MSP", "Paper +PropUQ", "Ours local (R3)", "S1",
             "Ours local (R2)", "S2", "S3"],
            table1_rows(cells, model, tasks, topos, [
                p("msp"), p("msp_prop"),
                v("R3", "NormMSP", LOCAL), v("R3", "NormMSP", ALPHA1_NORM),
                v("R2", "NormMSP", LOCAL), v("R2", "NormMSP", PROP), v("R2", "NormMSP", PROP_NORM),
            ]),
        ))
        out += ["", "### Verb", ""]
        out.append(md_table(
            ["Topology", "Dataset", "Metric", "Paper Verb", "Paper +PropUQ", "Ours local (R1)", "S1 = S3", "S2"],
            table1_rows(cells, model, tasks, topos, [
                p("verb"), p("verb_prop"),
                v("R1", "Verb", LOCAL), v("R1", "Verb", PROP_NORM), v("R1", "Verb", PROP),
            ]),
        ))
        out += ["", "### NormMSP-Answer (new, no paper column)", ""]
        out.append(md_table(
            ["Topology", "Dataset", "Metric", "Local (R3)", "S1", "Local (R2)", "S2", "S3"],
            table1_rows(cells, model, tasks, topos, [
                v("R3", "NormMSP-Answer", LOCAL), v("R3", "NormMSP-Answer", ALPHA1_NORM),
                v("R2", "NormMSP-Answer", LOCAL), v("R2", "NormMSP-Answer", PROP),
                v("R2", "NormMSP-Answer", PROP_NORM),
            ]),
        ))

    # Step-by-step ablation for NormMSP, only for cells that have R2 or R3.
    out += ["", "## NormMSP step-by-step", "",
            "Each step changes one thing. A: asking for α̂ (prompt). B: normalization with α = 1. "
            "C: α̂ itself (paper Table 4). D: normalization with α̂ (S3 vs S2).", ""]
    steps = [
        ("Local", "R2", LOCAL, None),
        ("Alpha1-Normalized (S1)", "R3", ALPHA1_NORM, None),
        ("Alpha1-Normalized", "R2", ALPHA1_NORM, "A"),
        ("Alpha1", "R2", ALPHA1, "B"),
        ("Prop-Uncertainty-Final (S2)", "R2", PROP, "C"),
        ("Prop-Uncertainty-Final-Normalized (S3)", "R2", PROP_NORM, "D vs S2"),
    ]
    rows = []
    for model in models:
        for task in tasks:
            for topo in topos:
                cell = cells[(model, task, topo)]
                if not ({"R2", "R3"} & set(cell.metrics)):
                    continue
                name = f"{MODEL_LABELS.get(model, model)} / {TASK_LABELS[task]} / {topo}"
                prev = None
                for idx, (label, run, row, step) in enumerate(steps):
                    auroc = cell.value(run, "NormMSP", row, "auroc")
                    prr = cell.value(run, "NormMSP", row, "prr")
                    if step == "D vs S2":
                        ref = cell.value("R2", "NormMSP", PROP, "auroc")
                        delta = f"D: {fmt_delta(ref, auroc)}"
                    elif step:
                        delta = f"{step}: {fmt_delta(prev, auroc)}"
                    else:
                        delta = "–"
                    rows.append([name if idx == 0 else "", label, run, fmt(auroc), fmt(prr), delta])
                    prev = auroc
    out.append(md_table(["Model / Dataset / Topology", "Row", "Run", "AUROC", "PRR", "Δ AUROC (step)"], rows)
               if rows else "_No NormMSP runs found yet._")

    # Diagnostics per run.
    out += ["", "## Diagnostics", ""]
    rows = []
    for model in models:
        for task in tasks:
            for topo in topos:
                cell = cells[(model, task, topo)]
                for run in RUNS:
                    if run not in cell.metrics:
                        continue
                    method = "Verb" if run == "R1" else "NormMSP"
                    diag = preds_diagnostics(cell.files[run][1])
                    rows.append([
                        MODEL_LABELS.get(model, model), TASK_LABELS[task], topo, f"{run} {RUNS[run]['desc']}",
                        fmt(cell.meta(run, "accuracy")), fmt(cell.meta(run, "time_per_sample_sec")),
                        str(cell.n_scored(run, method) if cell.n_scored(run, method) is not None else "–"),
                        "–" if run == "R3" else fmt_pct(diag["alpha_rate"]),
                        "–" if run == "R1" else fmt_pct(diag["trunc_rate"]),
                    ])
    out.append(md_table(["Model", "Dataset", "Topology", "Run", "Accuracy", "Time/sample (s)", "n scored",
                         "α̂ report rate", "Truncation rate"], rows) if rows else "_No runs found yet._")

    # Which files were used.
    out += ["", "## Files used", ""]
    used = []
    for (model, task, topo), cell in cells.items():
        for run, (m_path, _) in cell.files.items():
            if m_path is not None:
                used.append(f"- {run} · {MODEL_LABELS.get(model, model)} · {task} · {topo}: `{m_path}`")
    out += used or ["_None._"]
    return "\n".join(out) + "\n"


def write_csv(cells, path: Path) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "task", "topology", "run", "uq_method", "row", "auroc", "prr", "ece", "brier",
                         "n_with_uncertainty", "accuracy", "metrics_file"])
        for (model, task, topo), cell in cells.items():
            for run, payload in cell.metrics.items():
                by_method = payload.get("metrics", {}).get("metrics_by_uq_method", {})
                for method, rows in by_method.items():
                    for row, vals in rows.items():
                        writer.writerow([model, task, topo, run, method, row, vals.get("auroc"), vals.get("prr"),
                                         vals.get("ece"), vals.get("brier"), vals.get("n_with_uncertainty"), cell.meta(run, "accuracy"),
                                         cell.files[run][0]])
                        count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs", help="run.py output root")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--topologies", nargs="+", default=TOPOLOGIES, choices=TOPOLOGIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mas_prompt", default="role", choices=["role", "norole"])
    parser.add_argument("--out_md", type=Path, default=REPO_ROOT / "outputs" / "table1_report.md")
    parser.add_argument("--out_csv", type=Path, default=REPO_ROOT / "outputs" / "table1_long.csv")
    args = parser.parse_args(argv)

    cells = {
        (model, task, topo): Cell(args.outputs, model, task, topo, args.seed, args.mas_prompt)
        for model in args.models for task in args.tasks for topo in args.topologies
    }
    found = sum(len(c.metrics) for c in cells.values())
    expected = 3 * len(cells)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(build_report(cells, args.models, args.tasks, args.topologies, args.seed), encoding="utf-8")
    n_rows = write_csv(cells, args.out_csv)
    print(f"Runs found: {found}/{expected}")
    print(f"Report: {args.out_md}")
    print(f"CSV ({n_rows} rows): {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
