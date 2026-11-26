# PropUQ-MAS: Recreating Table 1 Under Three Setups

- **Date:** 2026-11-26
- **Status:** planned (not run)
- **Code:** this repository (changes vs upstream are logged in [`DECISIONS.md`](../DECISIONS.md))

## 1. Purpose

Recreate the paper's **Table 1** (final-output uncertainty, 3 models × 3 datasets × 3 topologies, MSP and Verb, each with and without PropUQ) under three ways of computing PropUQ:

1. **Original code setup**: what the released repository computes. This should reproduce the paper's numbers.
2. **Predicted optimal setup**: the paper's described method (raw adoption scores α̂ in Eq. 2, α̂ elicited for every local-UQ mode).
3. **Normalized setup**: α̂ elicited for every mode, combined with the release's α̂ normalization.

Questions to answer:

- Do we reproduce Table 1 with the original code?
- Does the paper rule (setup 2) beat what was released?
- Does the answer-only MSP estimator help?

Work is staged: **Phase 1** is a cheap first look, **Phase 2** completes Table 1, and **Phase 3 (optional)** adds UProp.

## 2. Background: what differs between the setups

PropUQ combines, at each agent v, a local uncertainty u_v with inherited risk from each parent p:

r_v = 1 − (1 − u_v) × Π_p (1 − α_pv × r_p)

| Part | Original code | Paper (as described) |
|---|---|---|
| α̂ in MSP mode | not asked; α = 1 | asked (Algorithm 1, step 5) |
| α̂ use | divided by the sum of α̂ at the node (single parent → always 1) | used as reported |
| MSP formula | 1 − exp(mean token log-prob), all tokens (= `NormMSP`) | cites sequence MSP, 1 − ∏p; no formula or token range given |

Local UQ estimators:

- **Verb**: the agent writes `<local_uncertainty score=…/>`.
- **NormMSP**: 1 − exp(mean token log-prob) over all generated tokens (thinking + answer). This is the code's "MSP" and the "MSP" column of Table 1.
- **NormMSP-Answer** (new): the same formula over tokens after the last `</think>`. If thinking never closes, it is set to 1.0. It comes from the same run as NormMSP. For Gemma (no thinking) it equals NormMSP.

Both NormMSP variants exclude the tokens of the appended `<alpha/>` / `<local_uncertainty/>` tags.

## 3. Configuration

| Setting | Value |
|---|---|
| Models | `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `google/gemma-3-12b-it` (gated on Hugging Face: request access and `huggingface-cli login` first) |
| Datasets | GSM8K (test, 1,319) and MBPP+ (test, 378) download automatically from Hugging Face on first use (the server needs internet access); MedQA (300) is included in the repository as `data/medqa.json` |
| Topologies | sequential, hierarchical, decentralized |
| Agents | 4 |
| Prompt | `--mas_prompt role` (sequential/hierarchical use role prompts; decentralized automatically uses the generic graph prompt). Role prompts match the paper: Table 8's "sequential, role" numbers equal Table 1's Qwen3-8B MedQA numbers. |
| Decoding | temperature 0.6, top-p 0.95, repetition penalty 1.05 |
| Max new tokens | 8192 for GSM8K and MedQA; 32768 for MBPP+ (paper settings; the `run.py` default of 4096 must be overridden) |
| Seed | 42 (single run) |
| Backend | vLLM, `--generate_bs 16` (repo default) |

## 4. Generation runs

There are 3 runs per (model, dataset, topology) cell. Every setup in Section 5 is read from these runs; nothing needs regenerating.

| Run | Mode | Asks for α̂? | Output folder |
|---|---|---|---|
| R1 | `Verb` | yes (as in the original code) | `outputs/Verb/` |
| R2 | `NormMSP` | yes | `outputs/NormMSP/` |
| R3 | `NormMSP --no_alpha_elicitation` | no (prompts byte-identical to the original MSP runs) | `outputs/NormMSP_noalpha/` |

### Phases

| Phase | Scope | Runs |
|---|---|---|
| 1: first look | Qwen3-8B × MedQA × 3 topologies | 9 |
| 2: full Table 1 | 3 models × 3 datasets × 3 topologies (includes Phase 1) | 81 |
| 3: UProp (optional) | see Section 9 | 10 per configuration |

Before Phase 1, run a smoke test: add `--max_samples 10` and check that the files appear and the metrics JSON has all rows. GSM8K is about two-thirds of all questions in Phase 2. If budget is tight, run MedQA and MBPP+ for every model first.

### Commands (run from the repository root)

```bash
MODELS="Qwen/Qwen3-8B"            # Phase 2: "Qwen/Qwen3-4B Qwen/Qwen3-8B google/gemma-3-12b-it"
TASKS="medqa"                     # Phase 2: "medqa mbppplus gsm8k"
TOPOS="sequential hierarchical decentralized"

for model in $MODELS; do
  for task in $TASKS; do
    if [ "$task" = "mbppplus" ]; then max_tokens=32768; else max_tokens=8192; fi
    for topo in $TOPOS; do
      common=(--method mas --model_name "$model" --task "$task"
              --mas_topology "$topo" --mas_node_num 4 --mas_prompt role
              --max_new_tokens "$max_tokens" --generate_bs 16 --seed 42
              --use_vllm --force_uq)
      python -u run.py "${common[@]}" --local_uncertainty_mode Verb
      python -u run.py "${common[@]}" --local_uncertainty_mode NormMSP
      python -u run.py "${common[@]}" --local_uncertainty_mode NormMSP --no_alpha_elicitation
    done
  done
done
```

**MBPP+ safety:** scoring executes the model-generated Python on the machine (`run_with_timeout`). Run it inside a container or VM, not on a machine holding credentials.

Result files, one per run:

```text
outputs/<folder>/<org>/<model>/raw/raw_preds_mas_<task>_<topo>_nodes4[_role]_seed42_n-1.jsonl
outputs/<folder>/<org>/<model>/uq/uq_metrics_mas_<task>_<topo>_nodes4[_role]_seed42_n<size>.json
```

- `_role` appears only for sequential and hierarchical.
- The raw file ends in `_n-1`, and the UQ files in `_n<dataset size>`.
- Metric values are under `metrics.metrics_by_uq_method.<method>.<row>` with `auroc`, `prr`, `ece`, `brier` and `n_with_uncertainty`. Accuracy and time are under `metadata`.

## 5. Setups (how results are read)

Each cell gives the run and metrics row.

| Setup | Verb | NormMSP ("MSP") | NormMSP-Answer |
|---|---|---|---|
| **Local baseline for setup 1** | R1 · `Local-Uncertainty-Final` | R3 · `Local-Uncertainty-Final` | R3 · `Local-Uncertainty-Final` |
| **1. Original code** | R1 · `Prop-Uncertainty-Final-Normalized` | R3 · `Prop-Uncertainty-Final-Alpha1-Normalized` | R3 · `Prop-Uncertainty-Final-Alpha1-Normalized` |
| **Local baseline for setups 2–3** | R1 · `Local-Uncertainty-Final` | R2 · `Local-Uncertainty-Final` | R2 · `Local-Uncertainty-Final` |
| **2. Predicted optimal (paper rule)** | R1 · `Prop-Uncertainty-Final` | R2 · `Prop-Uncertainty-Final` | R2 · `Prop-Uncertainty-Final` |
| **3. α̂ elicited + normalized** | R1 · `Prop-Uncertainty-Final-Normalized` | R2 · `Prop-Uncertainty-Final-Normalized` | R2 · `Prop-Uncertainty-Final-Normalized` |

Notes:

- **The MSP baseline differs by setup.** The original MSP baseline came from runs without the α̂ request (R3). Setups 2–3 use R2, whose agents also wrote α̂.
- **Verb is identical in setups 1 and 3.**
- **Setup 2 vs the paper.** Setup 2 follows the paper's described method but uses the code's NormMSP formula, not the cited sequence MSP.
- **NormMSP-Answer in setup 1** is a new estimator applied to the original setup, not something the original code computed.

## 6. Extra comparisons (free, from the same runs)

These separate what changes between setups 1 and 2 for NormMSP. Each step changes one thing:

| Step | Compare | Isolates |
|---|---|---|
| A | R3 `Alpha1-Normalized` vs R2 `Alpha1-Normalized` | Prompt effect of asking for α̂ (same math, different text) |
| B | R2 `Alpha1-Normalized` vs R2 `Alpha1` | Normalization, with α = 1 |
| C | R2 `Alpha1` vs R2 `Prop-Uncertainty-Final` | α̂ itself (the paper's Table 4 ablation) |
| D | R2 `Prop-Uncertainty-Final-Normalized` vs R2 `Prop-Uncertainty-Final` | Normalization with α̂ (setup 3 vs 2) |

Also compare R2 vs R3 accuracy, which is the prompt effect on answers.

## 7. Expectations

| # | Expectation | Source |
|---|---|---|
| E0 | Setup 1 roughly reproduces Table 1. Exact match is unlikely (sampling, hardware, library versions); look for the same sign and similar size of the PropUQ gains. | reproduction goal |
| E1 | Step A ≈ 0: asking for α̂ barely changes AUROC/PRR or accuracy | user's bet |
| E2 | On sequential, setups 1 and 3 (and Alpha1) are nearly identical: one parent per node means normalization gives 1. They differ only where α̂ = 0 or is missing. | follows from the math |
| E3 | Setup 2 differs most from setups 1 and 3 on hierarchical and decentralized (multi-parent nodes) | follows from the math |
| E4 | Setup 2 ≥ setup 1 on AUROC/PRR | hypothesis ("predicted optimal") |
| E5 | NormMSP-Answer vs NormMSP: unknown; answer tokens may track answer confidence better than thinking text. Identical for Gemma. | open question |

## 8. Report template

The tables in 8.2–8.4 are filled automatically. From the repository root, after any number of runs:

```bash
python analysis/make_table1.py
```

This writes:

- `outputs/table1_report.md`: paper vs ours per model, the step-by-step table, diagnostics, and the files used.
- `outputs/table1_long.csv`: every metrics row, for later analysis.

Runs not done yet show as "–". Use `--models`, `--tasks` and `--topologies` to narrow the scope (e.g. Phase 1: `--models Qwen/Qwen3-8B --tasks medqa`). The script needs only the standard library.

### 8.1 Paper reference: Table 1 (transcribed)

Each cell is local → +PropUQ.

**Qwen3-4B**

| Topology | Dataset | Metric | MSP → +PropUQ | Verb → +PropUQ |
|---|---|---|---|---|
| Sequential | GSM8K | AUROC | 0.578 → 0.617 | 0.466 → 0.501 |
| | | PRR | 0.160 → 0.238 | -0.073 → -0.068 |
| | MBPP+ | AUROC | 0.679 → 0.722 | 0.577 → 0.607 |
| | | PRR | 0.340 → 0.378 | 0.090 → 0.173 |
| | MedQA | AUROC | 0.649 → 0.742 | 0.428 → 0.607 |
| | | PRR | 0.254 → 0.410 | -0.167 → 0.236 |
| Hierarchical | GSM8K | AUROC | 0.564 → 0.541 | 0.535 → 0.597 |
| | | PRR | 0.100 → 0.094 | 0.082 → 0.222 |
| | MBPP+ | AUROC | 0.688 → 0.718 | 0.632 → 0.661 |
| | | PRR | 0.335 → 0.432 | 0.201 → 0.289 |
| | MedQA | AUROC | 0.628 → 0.760 | 0.621 → 0.705 |
| | | PRR | 0.238 → 0.417 | 0.216 → 0.296 |
| Decentralized | GSM8K | AUROC | 0.657 → 0.704 | 0.544 → 0.574 |
| | | PRR | 0.311 → 0.428 | 0.128 → 0.185 |
| | MBPP+ | AUROC | 0.695 → 0.710 | 0.588 → 0.604 |
| | | PRR | 0.382 → 0.426 | 0.140 → 0.210 |
| | MedQA | AUROC | 0.749 → 0.824 | 0.693 → 0.731 |
| | | PRR | 0.416 → 0.563 | 0.355 → 0.337 |

**Qwen3-8B**

| Topology | Dataset | Metric | MSP → +PropUQ | Verb → +PropUQ |
|---|---|---|---|---|
| Sequential | GSM8K | AUROC | 0.605 → 0.625 | 0.505 → 0.680 |
| | | PRR | 0.224 → 0.257 | -0.028 → 0.340 |
| | MBPP+ | AUROC | 0.583 → 0.648 | 0.562 → 0.577 |
| | | PRR | 0.155 → 0.282 | 0.064 → 0.120 |
| | MedQA | AUROC | 0.754 → 0.842 | 0.600 → 0.791 |
| | | PRR | 0.496 → 0.624 | 0.217 → 0.482 |
| Hierarchical | GSM8K | AUROC | 0.664 → 0.717 | 0.516 → 0.577 |
| | | PRR | 0.321 → 0.446 | -0.001 → 0.153 |
| | MBPP+ | AUROC | 0.619 → 0.719 | 0.580 → 0.630 |
| | | PRR | 0.172 → 0.401 | 0.081 → 0.245 |
| | MedQA | AUROC | 0.658 → 0.711 | 0.604 → 0.722 |
| | | PRR | 0.277 → 0.349 | 0.223 → 0.362 |
| Decentralized | GSM8K | AUROC | 0.728 → 0.768 | 0.486 → 0.563 |
| | | PRR | 0.445 → 0.534 | -0.004 → 0.181 |
| | MBPP+ | AUROC | 0.701 → 0.726 | 0.511 → 0.583 |
| | | PRR | 0.332 → 0.367 | 0.014 → 0.117 |
| | MedQA | AUROC | 0.812 → 0.821 | 0.718 → 0.854 |
| | | PRR | 0.539 → 0.568 | 0.429 → 0.626 |

**Gemma-3-12B**

| Topology | Dataset | Metric | MSP → +PropUQ | Verb → +PropUQ |
|---|---|---|---|---|
| Sequential | GSM8K | AUROC | 0.471 → 0.517 | 0.581 → 0.651 |
| | | PRR | 0.005 → 0.102 | 0.156 → 0.250 |
| | MBPP+ | AUROC | 0.544 → 0.583 | 0.684 → 0.742 |
| | | PRR | 0.079 → 0.128 | 0.277 → 0.465 |
| | MedQA | AUROC | 0.635 → 0.668 | 0.627 → 0.674 |
| | | PRR | 0.232 → 0.240 | 0.237 → 0.322 |
| Hierarchical | GSM8K | AUROC | 0.535 → 0.498 | 0.549 → 0.606 |
| | | PRR | 0.063 → 0.060 | 0.080 → 0.204 |
| | MBPP+ | AUROC | 0.630 → 0.674 | 0.682 → 0.730 |
| | | PRR | 0.204 → 0.207 | 0.382 → 0.423 |
| | MedQA | AUROC | 0.567 → 0.604 | 0.660 → 0.669 |
| | | PRR | 0.115 → 0.184 | 0.358 → 0.351 |
| Decentralized | GSM8K | AUROC | 0.563 → 0.567 | 0.634 → 0.665 |
| | | PRR | 0.128 → 0.162 | 0.271 → 0.294 |
| | MBPP+ | AUROC | 0.557 → 0.645 | 0.670 → 0.720 |
| | | PRR | 0.096 → 0.262 | 0.342 → 0.393 |
| | MedQA | AUROC | 0.599 → 0.636 | 0.576 → 0.611 |
| | | PRR | 0.155 → 0.208 | 0.160 → 0.200 |

Transcribed from the paper's Table 1. All 216 numbers were checked programmatically against the PDF's text layer (0 mismatches). Qwen3-8B MedQA sequential and hierarchical also agree with Table 8.

### 8.2 Our results: one set of tables per model

Fill one block per model. Rows follow Table 1 order (3 topologies × 3 datasets × AUROC/PRR = 18 rows). The paper columns repeat Section 8.1 for side-by-side reading.

**`<model>` · MSP (NormMSP)**

| Topology | Dataset | Metric | Paper MSP | Paper +PropUQ | Ours local (R3) | S1 | Ours local (R2) | S2 | S3 |
|---|---|---|---|---|---|---|---|---|---|
| Sequential | GSM8K | AUROC | | | | | | | |
| | | PRR | | | | | | | |
| | MBPP+ | AUROC | | | | | | | |
| | | PRR | | | | | | | |
| | MedQA | AUROC | | | | | | | |
| | | PRR | | | | | | | |
| Hierarchical | … | | | | | | | | |
| Decentralized | … | | | | | | | | |

**`<model>` · Verb**

| Topology | Dataset | Metric | Paper Verb | Paper +PropUQ | Ours local (R1) | S1 = S3 | S2 |
|---|---|---|---|---|---|---|---|
| (same 18 rows) | | | | | | | |

**`<model>` · NormMSP-Answer (new, no paper column)**

| Topology | Dataset | Metric | Local (R3) | S1 | Local (R2) | S2 | S3 |
|---|---|---|---|---|---|---|---|
| (same 18 rows) | | | | | | | |

### 8.3 NormMSP step-by-step (Phase 1 cells; extend as useful)

| Model / Dataset / Topology | Row | Run | AUROC | PRR | Δ AUROC vs previous |
|---|---|---|---|---|---|
| | Local | R2 | | | – |
| | Alpha1-Normalized (setup 1) | R3 | | | – |
| | Alpha1-Normalized | R2 | | | (step A) |
| | Alpha1 | R2 | | | (step B) |
| | Prop-Uncertainty-Final (setup 2) | R2 | | | (step C) |
| | Prop-Uncertainty-Final-Normalized (setup 3) | R2 | | | vs setup 2: (step D) |

### 8.4 Diagnostics (per run)

| Model | Dataset | Topology | Run | Accuracy | Time/sample (s) | n scored | α̂ report rate | Truncation rate |
|---|---|---|---|---|---|---|---|---|
| | | | R1 Verb | | | | | – |
| | | | R2 NormMSP | | | | | |
| | | | R3 NormMSP no-α̂ | | | | – | |

Where to find each value:

- **Accuracy:** metrics JSON `metadata.accuracy`.
- **Time per sample:** metrics JSON `metadata.time_per_sample_sec`.
- **n scored:** `n_with_uncertainty` for each row.
- **α̂ report rate:** share of edges with a parsable α̂, from `agents[*].alpha` vs `agents[*].incoming_agents` in `uq_preds`. Missing α̂ means the parent is skipped, which weakens propagation.
- **Truncation rate:** `agents[*].logits_uq_stats.answer_truncated` (a truncated output gets NormMSP-Answer = 1.0).

### 8.5 Observations

- E0–E5 held or not, with numbers:
- Cells where setup 1 does not reproduce the paper (sign flip or large gap):
- Surprises:
- Next step:

## 9. Phase 3 (optional): UProp baseline

- **What:** UProp (Duan et al., 2025) splits each step's uncertainty into its own part and the part inherited from earlier steps, estimated from sampled trajectories. It only applies to the **sequential** topology.
- **Existing scripts:**
  - `sampling_baselines/uprop/scripts/uprop_generate_medqa.sh`: 10 seeds (42–51), Qwen3-8B, MedQA, now `NormMSP --no_alpha_elicitation`.
  - Then `uprop_score_medqa.sh`.
- **Fairness issue (decide before running):** the script defaults to `MAX_NEW_TOKENS=512` and `TEMPERATURE=0.8`, versus 8192 and 0.6 for PropUQ. With Qwen3 thinking, many 512-token trajectories are likely cut off before an answer, which would understate UProp. Options:
  - (a) run as shipped, to reproduce the paper's Table 8 UProp numbers;
  - (b) also run with 8192 / 0.6 for a fair comparison.
- **Scope:** MedQA only (scripts are MedQA-specific). Each configuration costs 10 full generation runs.

## 10. Known limitations and notes

- **Single seed.** Results come from one run (seed 42). The paper's α̂ ablation gaps are about 0.01 AUROC, which is roughly run-to-run noise, so treat differences under ~0.01–0.02 as inconclusive. Follow-up: 3 seeds (`--seed 42/43/44`), or a paired bootstrap over questions.
- **Exact reproduction is not expected.** vLLM sampling varies with hardware, library versions and batch composition. Compare direction and size of gains, not third decimals.
- **MSP formula.** All MSP numbers use NormMSP (the code's formula), not the cited sequence MSP.
- **Inherited code behaviour, unchanged:**
  - PRR is computed only up to 50% rejection (`PRR_MAX_REJECTION_RATE = 0.5`); the paper doesn't mention this cap.
  - A missing local uncertainty means no local term (equivalent to u = 0).
- **Decentralized** uses the generic graph prompt (no role prompts exist for it).
- **Not in scope:** Qwen3-14B (appendix Table 5), intermediate-agent results (Table 2), scale analysis (Figure 3).
