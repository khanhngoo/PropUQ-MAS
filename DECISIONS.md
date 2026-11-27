# Decisions and Changes Log

This log records every change made to this repository relative to the upstream release, and the reasoning behind it.

- **Upstream:** <https://github.com/yaokunliu/PropUQ-MAS>, commit `de5e594`
- **Paper:** *PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems* (EMNLP 2026)
- **Started:** 2026-11-26
- **Goal:** reproduce the paper's Table 1, and compare the released code with the method as the paper describes it, before using PropUQ for topology research.

Notation: u_v is an agent's local uncertainty, α̂ the self-reported adoption score on an edge, and r_v the propagated uncertainty from Eq. 2:

r_v = 1 − (1 − u_v) × Π_p (1 − α_pv × r_p)

---

## 1. Findings: upstream code vs paper

These are discrepancies found while reading the code and the paper. They motivated the changes in Section 2.

| # | Finding | Evidence | Impact |
|---|---|---|---|
| F1 | The code divides α̂ by the sum of α̂ at each node, unlike Eq. 2. A single parent's weight therefore always becomes 1, and only the ratios between parents survive. The README claims single-parent nodes are unaffected, which is false. | `_alpha_lookup` in `methods/uncertainty_quantification.py`. Toy check: chain with u = 0.1, parent r = 0.5 gives 0.55 for both α̂ = 0.3 and α̂ = 0.9; the paper rule gives 0.235 and 0.505. | α̂ has almost no effect in the released code. On sequential topologies the paper's Table 4 "with vs without α̂" difference should not be reproducible with this code. |
| F2 | The paper's own case study (Figure 7) uses raw α̂. | Critic: u = 0.75, parent r = 0.15, α̂ = 0.6 → 1 − 0.25 × (1 − 0.09) = **0.773**, as printed. The normalized rule would give 0.788. | The paper's reported method is the raw rule. |
| F3 | In MSP mode, agents are never asked for α̂, and every α is set to 1 (then normalized). | `prompts.py`: `use_alpha = allow_alpha and use_verb`; `methods/mas.py`: `_default_alpha_scores`. | "MSP + PropUQ" in the released code is the paper's "w/o α̂" ablation (Table 4), not the full method. |
| F4 | "MSP" is computed as 1 − exp(mean token log-prob) over all generated tokens, including the `<think>` block. The paper cites MSP (Vashurin et al., 2025), i.e. 1 − P(whole output), and gives no formula or token range. | `models.py`: `_uq_stats_from_raw_logits`, `_uq_stats_from_cumulative_logprob`. | Mislabelled: it equals 1 − 1/perplexity. Sequence MSP would saturate near 1 for long reasoning outputs. |
| F5 | Part of Table 8's runtime column is estimated, not measured. | `sampling_baselines/comparison.md`: hierarchical MSP time "scaled", Verb times taken from MATU, PropUQ values "transcribed". | Runtime comparisons with MATU/UProp are approximate. |
| F6 | The UProp baseline generates with 512 max new tokens and temperature 0.8, versus 8192 and 0.6 for PropUQ. | `sampling_baselines/uprop/scripts/uprop_generate_medqa.sh`. | With Qwen3 thinking, many UProp trajectories are likely cut off before an answer, which would understate UProp. **Not changed**; to revisit before any baseline comparison. |
| F7 | PRR is computed only up to 50% rejection. | `PRR_MAX_REJECTION_RATE = 0.5`. The paper (Appendix C.3) defines PRR over the full curve. | Reported PRR may not equal full-curve PRR. **Not changed.** |
| F8 | A missing local uncertainty means no local term (equivalent to u = 0). | `_compute_prop_uncertainty`. | Optimistic default. **Not changed.** |

---

## 2. Changes

### C1. Paper rule and release rule side by side

**Files:** `methods/uncertainty_quantification.py`

The UQ replay now computes four propagation variants from the same generations and reports all of them in the metrics JSON:

| Metrics row | Fields | α on each edge | Meaning |
|---|---|---|---|
| `Prop-Uncertainty-Final` | `prop_uncertainty*` | α̂ as reported | Paper rule (Eq. 2). **Now the main row.** |
| `Prop-Uncertainty-Final-Normalized` | `prop_norm_uncertainty*` | α̂ ÷ sum of α̂ at the node | Upstream rule (the old `Prop-Uncertainty-Final`) |
| `Prop-Uncertainty-Final-Alpha1` | `prop_alpha1_uncertainty*` | 1 | Paper's "w/o α̂" ablation (Table 4) |
| `Prop-Uncertainty-Final-Alpha1-Normalized` | `prop_alpha1_norm_uncertainty*` | 1/k at a node with k parents | Upstream MSP-mode behaviour |

- Implemented as the `PROP_SOURCES` table. `_alpha_lookup` gains `normalize=`, and `_compute_prop_uncertainty` gains `normalize_alpha=`.
- **Breaking change:** the meaning of `Prop-Uncertainty-Final` / `prop_uncertainty*` changed from the normalized rule to the paper rule. The old meaning is now the `Normalized` row.

### C2. MSP renamed to NormMSP, plus an answer-only variant

**Files:** `token_uq.py` (new), `models.py`, `methods/mas.py`, `run.py`

- The CLI mode `MSP` is now `NormMSP`. Passing `MSP` is rejected by argparse.
- One `NormMSP` run records two local estimators from the same outputs:
  - **`NormMSP`:** 1 − exp(mean token log-prob), all generated tokens. Same formula as upstream "MSP".
  - **`NormMSP-Answer`:** the same formula, only tokens after the last `</think>`.
- Both are computed by `token_uq.token_prob_uncertainties` from per-token log-probs, for both the vLLM and Hugging Face backends. The old helpers `_uq_stats_from_raw_logits` and `_uq_stats_from_cumulative_logprob` were removed.
- `uq_stats` now holds `NormMSP`, `NormMSP-Answer`, `token_count_all`, `token_count_answer`, `answer_truncated` and `has_think_block`. It no longer holds `msp` or `mean_token_probability`.

### C3. α̂ elicited in every local-UQ mode

**Files:** `prompts.py`, `methods/mas.py`

- The prompt asks for α̂ whenever the agent has incoming messages, in both Verb and NormMSP modes. Previously only Verb mode asked.
- `MASMethod._alpha` always parses the `<alpha/>` tags. The silent default of α = 1 (`_default_alpha_scores`) was removed.

### C4. `--no_alpha_elicitation` switch

**Files:** `run.py`, `prompts.py`

- **What it does:** removes the α̂ request from all prompts. With `NormMSP`, the prompts are byte-identical to upstream MSP prompts (verified for all 9 role and generic prompt variants).
- **Outputs:** runs with the switch write to `outputs/<mode>_noalpha/`. Raw-pred cache names don't encode the prompt, so without a separate folder they would collide with normal runs.
- **Metadata:** metrics JSON records `alpha_elicitation`.

### C5. Separate output folders

**Files:** `run.py`

The mode names in folder paths changed, so new runs never reuse upstream caches generated with different prompts:

- `outputs/NormMSP/` instead of `outputs/MSP/`
- `outputs/NormMSP_noalpha/` for runs with the switch

### C6. UProp baseline repointed

**Files:** `sampling_baselines/uprop/scripts/uprop_generate_medqa.sh`, `uprop_score_medqa.sh`, `code/uprop_reproduction.py`, `README.md`

- The generation script now uses `--local_uncertainty_mode NormMSP --no_alpha_elicitation`, which matches upstream MSP prompts.
- Scoring reads from `outputs/NormMSP_noalpha/`.
- Without this change the script failed after the rename (C2). Its 512-token / temperature 0.8 settings (F6) are unchanged.

### C7. Table 1 report script

**Files:** `analysis/make_table1.py` (new)

- **Reads:** the metrics and preds files of the three runs per (model, dataset, topology) cell.
- **Writes:** `outputs/table1_report.md` (paper vs ours per model, a step-by-step ablation, diagnostics) and `outputs/table1_long.csv`.
- **Paper values:** Table 1 is embedded, checked against the PDF text layer (216 numbers, 0 mismatches).
- **Dependencies:** standard library only.

### C8. Documentation

**Files:** `README.md`, `sampling_baselines/uprop/README.md`, `docs/experiment_plan_2026-11-26.md`, this file

- The README describes the four propagation rows, NormMSP / NormMSP-Answer, the switch, and the output folders.
- The incorrect claim that single-parent nodes are unaffected by normalization was replaced.
- The experiment plan lives in `docs/`, and the README links to it and to this file.

### C9. Calibration metrics: ECE and Brier

**Files:** `methods/uncertainty_quantification.py`, `analysis/make_table1.py`

- Every metrics row now also has `ece` (15 equal-width bins) and `brier`, computed on confidence = 1 − uncertainty, same examples as AUROC/PRR.
- Printed in the run summary and written to `table1_long.csv`. AUROC/PRR values are unchanged.
- Why: AUROC/PRR only check ranking. Quantization studies report calibration, and double counting at reconvergent nodes (diamond/decentralized) should show up in calibration before ranking.

### C10. Precision comparison script

**Files:** `analysis/compare_precision.py` (new), `docs/experiment_plan_2026-11-27_quantization.md` (new)

- Compares quantized checkpoints (default `Qwen/Qwen3-8B-FP8`, `Qwen/Qwen3-8B-AWQ`) against the BF16 base on the same configs.
- Reports accuracy, answer flips vs base (correct→wrong and wrong→correct, paired by question), AUROC/PRR/ECE/Brier for local and the main propagation row with the gain, and diagnostics (α̂ report rate, truncation, missing local UQ, mean local uncertainty per agent).
- Recomputes metrics from `uq_preds`, so older files also get ECE/Brier. Writes `outputs/precision_report.md` and `outputs/precision_long.csv`. Standard library only.

---

## 3. Decisions

| # | Decision | Choice | Reason | Alternative considered | Decided by |
|---|---|---|---|---|---|
| D1 | Which α̂ rule is the main row | Paper rule (raw α̂); keep the normalized rule as a row | Matches Eq. 2 and Figure 7 (F1, F2) | Keep normalized as main | user |
| D2 | Keep α = 1 | Two rows: raw (paper ablation) and normalized (exact upstream MSP rule) | They differ at multi-parent nodes (toy star: 0.712 vs 0.347) | Only one α = 1 row | user asked; second row by Claude |
| D3 | Rename MSP | `NormMSP` | It is 1 − 1/perplexity, not sequence MSP (F4) | Keep the name `MSP` | user asked; name by Claude |
| D4 | Answer-only estimator | `NormMSP-Answer`, computed in the same run | Exact comparison on identical outputs; thinking text dominates the token count | Separate run | user |
| D5 | α̂ in MSP mode | Ask for it | α̂ is an edge input, independent of how u_v is estimated | Keep α = 1 | user |
| D6 | Exact upstream MSP reproduction | Generation-time switch `--no_alpha_elicitation` | Alpha1 rows are chosen after generation; the α̂ request still changes the generated text | Accept a near-match | user |
| D7 | α̂ tag tokens in NormMSP | Excluded from both variants | Otherwise asking for α̂ would shift the local estimate | Include them | Claude |
| D8 | Plain-text `alpha: 0.7` lines in NormMSP | Not excluded; only XML tags are | The plain-text pattern also matches real answers (e.g. `alpha = 0.5` in MBPP code) | Exclude them like upstream's text-stripping regex | Claude |
| D9 | Thinking never closes (truncated) | `NormMSP-Answer` = 1.0 | No answer means maximal uncertainty; dropping these would hide easy errors | Drop from scoring | Claude |
| D10 | Output without think tags (Gemma, thinking disabled) | Whole output counts as the answer | For Gemma both NormMSP variants are identical | – | Claude |
| D11 | Missing α̂ for a parent | Parent skipped (α = 0) in the reported-α̂ rows; Alpha1 rows ignore reports | Same as upstream for Verb; the Alpha1 rows give the conservative bound | Default to 1 | inherited, kept |
| D12 | Examples scored | Only examples with values for local + all four propagation rows | All rows of a metrics summary use the same examples | Per-row filtering | Claude |
| D13 | vLLM log-probs | Mean of per-token log-probs, not `cumulative_logprob` | Needed for tag exclusion and the answer split; same value otherwise | – | Claude |
| D14 | Old caches with an `MSP` key | Still scored (`KNOWN_UQ_METHODS` includes `MSP`; the report script maps it to NormMSP) | Lets older runs be rescored | Drop support | Claude |
| D15 | Sequence MSP (1 − ∏p) variant | Not added | Saturates near 1 for long outputs; not needed for now | Add `SeqMSP-Answer` | user |
| D16 | UProp / PRR / missing-u behaviour (F6–F8) | Unchanged, documented | Out of scope for the first reproduction | Fix now | user (UProp deferred) |
| D17 | Calibration metrics | ECE (15 equal-width bins) + Brier on 1 − u | Standard in quantization-UQ work; Brier needs no binning | Adaptive-bin ECE only | user asked; binning by Claude |
| D18 | How to quantize | Official pre-quantized checkpoints (`-FP8` = 8-bit, `-AWQ` = 4-bit), loaded by vLLM as is | No calibration-data choices of our own; FP8 is native on H100; "Q8/Q4" GGUF is poorly supported in vLLM | Quantize ourselves (AutoAWQ/GPTQ), or vLLM on-the-fly `quantization="fp8"` | user |
| D20 | Generation batch size | `--generate_bs` 64 for MedQA/GSM8K, 32 for MBPP+, fixed per task for BF16 and quantized runs | 16 leaves most of an 80 GB H100 idle; fixing it per task keeps flip counts free of batching effects | 16 everywhere | user |
| D19 | Flip pairing | By question text; positional fallback if questions repeat | Robust to reordering between runs | Positional only | Claude |

---

## 4. Experiment setups this enables

Three generation runs per (model, dataset, topology) cell:

| Run | Mode | Output folder |
|---|---|---|
| R1 | Verb | `outputs/Verb/` |
| R2 | NormMSP | `outputs/NormMSP/` |
| R3 | NormMSP + `--no_alpha_elicitation` | `outputs/NormMSP_noalpha/` |

| Setup | Verb | NormMSP |
|---|---|---|
| S1: original code | R1 · Normalized | R3 · Alpha1-Normalized |
| S2: paper rule | R1 · Prop-Uncertainty-Final | R2 · Prop-Uncertainty-Final |
| S3: α̂ + normalized | R1 · Normalized | R2 · Normalized |

The full experiment plan (phases, commands, expectations, report template) is in [`docs/experiment_plan_2026-11-26.md`](docs/experiment_plan_2026-11-26.md).

---

## 5. Verification

**Done (CPU only):**

- **Propagation rules:** 5 toy cases (chain/star × α̂ values, missing α̂) for all four rows match hand-calculated values.
- **Token-level estimates:** NormMSP / NormMSP-Answer checked with the real Qwen3-8B tokenizer (`</think>` = token 151668): thinking + answer + tag, truncated output, no think tags, and code containing `alpha = 0.5`. All exact.
- **Prompts:**
  - NormMSP asks for α̂ but not local uncertainty.
  - `--no_alpha_elicitation` removes α̂ for Verb and NormMSP across sequential, hierarchical and generic prompts.
  - With the switch, NormMSP prompts are byte-identical to upstream MSP prompts (9/9).
- **Output folders:** switched and normal runs go to different folders.
- **Report script:** end-to-end test on fake outputs written with `run.py`'s own path builders and UQ writers. Values match the underlying JSON.
- **ECE/Brier:** hand-checked toys (labels [1,0], conf [0.9,0.2] → ECE 0.15, Brier 0.025; conf 1.0 falls in the last bin).
- **Precision script:** fake base/FP8/AWQ outputs with planted flips; flip rates, gaps for missing runs, and CSV rows check out.

**Not done:** no real generation run (no GPU available yet). The first GPU run should check the α̂ report rate in NormMSP mode and the NormMSP-Answer truncation rate; both appear in the report's diagnostics.
