#!/usr/bin/env python3
"""Compare quantized checkpoints against a full-precision base model.

For each generation run (R1 Verb, R2 NormMSP, R3 NormMSP no-alpha), task and
topology, it loads the base model's uq_preds and each variant's, and reports:

  * accuracy, and answer flips vs the base, paired by question
    (correct -> wrong and wrong -> correct, counted separately);
  * UQ quality: AUROC, PRR, ECE, Brier for the local row and the main
    propagation row (Prop for R1/R2 = setup 2, Alpha1-Normalized for R3 =
    setup 1), plus the propagation gain (prop minus local);
  * diagnostics: alpha report rate, answer truncation rate, missing local-UQ
    rate (e.g. no <local_uncertainty> tag), mean local uncertainty per agent.

Metrics are recomputed from uq_preds, so files written before ECE/Brier were
added still get them. Every propagation row goes to the CSV.

Standard library only; does not need a GPU, torch or vLLM.

Usage (from the repository root):
  python analysis/compare_precision.py --base Qwen/Qwen3-8B \\
      --variants Qwen/Qwen3-8B-FP8 Qwen/Qwen3-8B-AWQ
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import make_table1 as t1  # noqa: E402
from methods.uncertainty_quantification import (  # noqa: E402
    _agent_local_uncertainty_for_method,
    build_uncertainty_metrics_summary,
)

LOCAL, PROP, PROP_NORM, ALPHA1, ALPHA1_NORM = t1.LOCAL, t1.PROP, t1.PROP_NORM, t1.ALPHA1, t1.ALPHA1_NORM

# (uq_method, main propagation row) shown in the report for each run.
REPORT_VIEWS = {
    "R1": [("Verb", PROP)],
    "R2": [("NormMSP", PROP), ("NormMSP-Answer", PROP)],
    "R3": [("NormMSP", ALPHA1_NORM)],
}
UQ_METRICS = ["auroc", "prr", "ece", "brier"]


def short(model: str) -> str:
    return model.split("/")[-1]


def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RunData:
    """One model's outputs for one (run, task, topology)."""

    def __init__(self, outputs: Path, run: str, model: str, task: str, topo: str, seed: int, prompt: str):
        self.model = model
        _, self.preds_path = t1.find_run_files(outputs, run, model, task, topo, seed, prompt)
        self.preds = load_jsonl(self.preds_path) if self.preds_path else None
        self.metrics = build_uncertainty_metrics_summary(self.preds)["metrics_by_uq_method"] if self.preds else {}
        self.diag = t1.preds_diagnostics(self.preds_path)

    @property
    def ok(self) -> bool:
        return bool(self.preds)

    @property
    def accuracy(self) -> Optional[float]:
        if not self.preds:
            return None
        return sum(bool(p.get("correct")) for p in self.preds) / len(self.preds)

    def value(self, method: str, row: str, metric: str) -> Optional[float]:
        if method == "NormMSP" and method not in self.metrics:
            method = "MSP"  # files written before the NormMSP rename
        return (self.metrics.get(method, {}).get(row) or {}).get(metric)

    def correctness_by_key(self) -> Dict:
        keys = [p.get("question") for p in self.preds]
        if all(keys) and len(set(keys)) == len(keys):
            return {k: bool(p.get("correct")) for k, p in zip(keys, self.preds)}
        return {i: bool(p.get("correct")) for i, p in enumerate(self.preds)}  # fall back to position

    def local_stats(self, method: str) -> Dict:
        """Missing local-UQ rate over agent outputs, and mean local uncertainty per agent position."""
        outputs = missing = 0
        sums: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for pred in self.preds or []:
            for pos, agent in enumerate(pred.get("agents", [])):
                outputs += 1
                u = _agent_local_uncertainty_for_method(agent, method)
                if u is None:
                    missing += 1
                    continue
                idx = agent.get("index", pos)
                sums[idx] = sums.get(idx, 0.0) + u
                counts[idx] = counts.get(idx, 0) + 1
        return {
            "missing_rate": missing / outputs if outputs else None,
            "mean_by_agent": {i: sums[i] / counts[i] for i in sorted(sums)},
        }


def flips(base: RunData, other: RunData) -> Dict[str, Optional[float]]:
    a, b = base.correctness_by_key(), other.correctness_by_key()
    shared = [k for k in a if k in b]
    if not shared:
        return {"n_paired": 0, "c2w": None, "w2c": None, "flip": None}
    c2w = sum(a[k] and not b[k] for k in shared) / len(shared)
    w2c = sum(not a[k] and b[k] for k in shared) / len(shared)
    return {"n_paired": len(shared), "c2w": c2w, "w2c": w2c, "flip": c2w + w2c}


def fmt(x: Optional[float]) -> str:
    return t1.fmt(x)


def fmt_pct(x: Optional[float]) -> str:
    return t1.fmt_pct(x)


def fmt_gain(prop: Optional[float], local: Optional[float]) -> str:
    return t1.fmt_delta(local, prop)


def build_report(data, views_by_run, base, variants, tasks, topos, seed) -> str:
    models = [base, *variants]
    out = [
        f"# Precision comparison: {short(base)} vs {', '.join(short(v) for v in variants)}",
        "",
        f"Seed {seed}. Flips are paired by question against `{short(base)}`. "
        "c→w = correct in base, wrong in variant; w→c = the reverse. "
        "Gain = propagation row minus local row (positive is better for AUROC/PRR; for ECE/Brier lower is better). "
        "ECE uses 15 equal-width bins on confidence = 1 − uncertainty; compare it across precisions, "
        "not across UQ methods. Missing runs show as “–”.",
        "",
    ]

    for run, views in views_by_run.items():
        desc = t1.RUNS[run]["desc"]
        out += [f"## {run} · {desc}", "", "### Accuracy and flips", ""]
        rows = []
        for task in tasks:
            for topo in topos:
                b = data[(run, base, task, topo)]
                for model in models:
                    d = data[(run, model, task, topo)]
                    if model == base or not (b.ok and d.ok):
                        f = {"n_paired": None, "c2w": None, "w2c": None, "flip": None}
                    else:
                        f = flips(b, d)
                    rows.append([
                        t1.TASK_LABELS.get(task, task), topo, short(model), fmt_pct(d.accuracy),
                        "" if model == base else t1.fmt_delta(b.accuracy, d.accuracy),
                        "" if model == base else fmt_pct(f["c2w"]),
                        "" if model == base else fmt_pct(f["w2c"]),
                        "" if model == base else str(f["n_paired"] or "–"),
                    ])
        out += [t1.md_table(["Task", "Topology", "Model", "Acc", "ΔAcc", "c→w", "w→c", "Paired n"], rows), ""]

        for method, prop_row in views:
            prop_name = "Prop" if prop_row == PROP else "PropA1Norm"
            out += [f"### UQ · {method} · local vs {prop_name}", ""]
            rows = []
            for task in tasks:
                for topo in topos:
                    for model in models:
                        d = data[(run, model, task, topo)]
                        loc = {m: d.value(method, LOCAL, m) for m in UQ_METRICS}
                        prp = {m: d.value(method, prop_row, m) for m in UQ_METRICS}
                        rows.append([
                            t1.TASK_LABELS.get(task, task), topo, short(model),
                            fmt(loc["auroc"]), fmt(prp["auroc"]), fmt_gain(prp["auroc"], loc["auroc"]),
                            fmt(loc["prr"]), fmt(prp["prr"]), fmt_gain(prp["prr"], loc["prr"]),
                            fmt(loc["ece"]), fmt(prp["ece"]),
                            fmt(loc["brier"]), fmt(prp["brier"]),
                        ])
            header = ["Task", "Topology", "Model",
                      "AUROC loc", f"AUROC {prop_name}", "gain",
                      "PRR loc", f"PRR {prop_name}", "gain",
                      "ECE loc", f"ECE {prop_name}", "Brier loc", f"Brier {prop_name}"]
            out += [t1.md_table(header, rows), ""]

        out += ["### Diagnostics", ""]
        rows = []
        for task in tasks:
            for topo in topos:
                for model in models:
                    d = data[(run, model, task, topo)]
                    method = views[0][0]
                    ls = d.local_stats(method) if d.ok else {"missing_rate": None, "mean_by_agent": {}}
                    per_agent = " / ".join(f"{u:.2f}" for u in ls["mean_by_agent"].values()) or "–"
                    rows.append([
                        t1.TASK_LABELS.get(task, task), topo, short(model),
                        fmt_pct(d.diag["alpha_rate"]), fmt_pct(d.diag["trunc_rate"]),
                        fmt_pct(ls["missing_rate"]), per_agent,
                    ])
        out += [t1.md_table(["Task", "Topology", "Model", "α̂ reported", "Truncated",
                             f"{views[0][0]} missing", "Mean local u per agent"], rows), ""]

    out += ["## Files used", ""]
    for (run, model, task, topo), d in sorted(data.items(), key=lambda kv: kv[0]):
        if d.preds_path:
            out.append(f"- {run} · {short(model)} · {task} · {topo}: `{d.preds_path}`")
    out.append("")
    return "\n".join(out)


def write_csv(data, base, path: Path) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "model", "task", "topology", "uq_method", "row", *UQ_METRICS,
                         "n_with_uncertainty", "accuracy", "c2w_vs_base", "w2c_vs_base", "n_paired", "preds_file"])
        for (run, model, task, topo), d in data.items():
            if not d.ok:
                continue
            b = data.get((run, base, task, topo))
            f = flips(b, d) if model != base and b is not None and b.ok else {}
            for method, rows in d.metrics.items():
                for row, vals in rows.items():
                    writer.writerow([run, model, task, topo, method, row, *(vals.get(m) for m in UQ_METRICS),
                                     vals.get("n_with_uncertainty"), d.accuracy,
                                     f.get("c2w"), f.get("w2c"), f.get("n_paired"), d.preds_path])
                    count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=t1.REPO_ROOT / "outputs", help="run.py output root")
    parser.add_argument("--base", default="Qwen/Qwen3-8B", help="full-precision model (reference for flips)")
    parser.add_argument("--variants", nargs="+", default=["Qwen/Qwen3-8B-FP8", "Qwen/Qwen3-8B-AWQ"])
    parser.add_argument("--runs", nargs="+", default=list(REPORT_VIEWS), choices=list(REPORT_VIEWS))
    parser.add_argument("--tasks", nargs="+", default=t1.TASKS, choices=t1.TASKS)
    parser.add_argument("--topologies", nargs="+", default=t1.TOPOLOGIES, choices=t1.TOPOLOGIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mas_prompt", default="role", choices=["role", "norole"])
    parser.add_argument("--out_md", type=Path, default=t1.REPO_ROOT / "outputs" / "precision_report.md")
    parser.add_argument("--out_csv", type=Path, default=t1.REPO_ROOT / "outputs" / "precision_long.csv")
    args = parser.parse_args(argv)

    views_by_run = {run: REPORT_VIEWS[run] for run in args.runs}

    models = [args.base, *args.variants]
    data = {
        (run, model, task, topo): RunData(args.outputs, run, model, task, topo, args.seed, args.mas_prompt)
        for run in views_by_run for model in models for task in args.tasks for topo in args.topologies
    }
    found = sum(d.ok for d in data.values())

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        build_report(data, views_by_run, args.base, args.variants, args.tasks, args.topologies, args.seed), encoding="utf-8"
    )
    n_rows = write_csv(data, args.base, args.out_csv)
    print(f"Runs found: {found}/{len(data)}")
    print(f"Report: {args.out_md}")
    print(f"CSV ({n_rows} rows): {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
