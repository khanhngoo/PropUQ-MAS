# PropUQ-MAS Under Quantization

**Date:** 2026-11-27
**Depends on:** [`experiment_plan_2026-11-26.md`](experiment_plan_2026-11-26.md) (the BF16 runs are the reference)

## 1. Purpose

Single-agent studies find that quantization slightly hurts UQ (for Qwen3 FP8/AWQ, AUROC and ECE degrade by up to about 5%; arXiv 2505.23854), and that 4-bit lowers confidence mostly where the full model was already unsure (Proskurina et al., NAACL Findings 2024). Nobody has checked this in a multi-agent system. Questions:

1. **Q1.** How much do accuracy and local UQ (Verb, NormMSP) degrade at 8-bit and 4-bit inside a MAS?
2. **Q2.** Does the propagation gain (prop minus local) shrink, stay, or grow under quantization?
3. **Q3.** Does quantization break the tag format (`<alpha>`, `<local_uncertainty>`) that PropUQ depends on?
4. **Q4.** Does the damage depend on topology, i.e. does it compound along a chain?

## 2. Setup

| Item | Value |
|---|---|
| Precisions | BF16 `Qwen/Qwen3-8B` (reference, from the main plan), 8-bit `Qwen/Qwen3-8B-FP8`, 4-bit `Qwen/Qwen3-8B-AWQ` |
| How | Official pre-quantized checkpoints. vLLM reads the quantization from the checkpoint config; only `--model_name` changes. |
| Datasets | MedQA (300), GSM8K (1,319), MBPP+ (378) |
| Topologies | sequential, hierarchical, decentralized |
| Runs per config | R1 Verb, R2 NormMSP (setup 2 and 3 rows come from these). R3 `--no_alpha_elicitation` is optional. |
| Everything else | Same as the main plan: role prompts, 8192 tokens (32768 for MBPP+), seed 42, `--use_vllm --force_uq` |

Count per quantized model: 3 datasets × 3 topologies = 9 configs, × 2 runs = 18 runs. Both precisions: 36 runs.

Suggested order: MedQA first (cheapest), then GSM8K, then MBPP+ (longest, and it runs generated code: use a container).

## 3. Commands

```bash
MODELS="Qwen/Qwen3-8B-FP8 Qwen/Qwen3-8B-AWQ"
TASKS="medqa gsm8k mbppplus"
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
    done
  done
done
```

Then (no GPU needed):

```bash
python analysis/compare_precision.py --base Qwen/Qwen3-8B \
    --variants Qwen/Qwen3-8B-FP8 Qwen/Qwen3-8B-AWQ --runs R1 R2
```

This writes `outputs/precision_report.md` and `outputs/precision_long.csv`.

## 4. Metrics

| Metric | Answers |
|---|---|
| Accuracy, ΔAcc vs BF16 | Q1 |
| Flips vs BF16: c→w and w→c separately, paired by question | Q1; flat accuracy can hide many changed answers |
| AUROC, PRR (local and Prop) | Q1, Q2 |
| ECE, Brier (local and Prop) | Q1, Q2; ranking can stay while calibration drifts |
| Gain = Prop − Local for AUROC and PRR | Q2 |
| α̂ report rate, truncation rate, missing local-UQ rate | Q3 |
| Mean local uncertainty per agent position | Q4 (does uncertainty rise down the chain?) |

ECE treats 1 − u as P(correct). NormMSP is not a calibrated probability, so compare ECE across precisions within one method, not across methods.

## 5. Expectations

- **E1.** FP8 ≈ BF16: |ΔAcc| < 1 point, AUROC within noise (~0.01–0.02 at one seed).
- **E2.** AWQ loses a few points of accuracy, with more c→w than w→c flips.
- **E3.** NormMSP moves more than Verb under quantization, since it reads token probabilities directly.
- **E4.** If quantization raises upstream errors, propagation has more to catch, so the gain may grow; if α̂ reporting breaks, the gain shrinks. Q3 tells which.

## 6. Report template

Copy the tables from `outputs/precision_report.md`, then fill in:

- Q1 (degradation by precision):
- Q2 (gain stays / shrinks / grows):
- Q3 (format failures):
- Q4 (topology dependence):
- Surprises:
- Next step:

## 7. Limitations and follow-ups

- **One seed.** Flip rates include sampling noise. A BF16-vs-BF16 rerun with another seed gives the noise floor for flips; without it, small flip rates can't be attributed to quantization.
- **One model.** Qwen3-4B and 14B also have official FP8/AWQ checkpoints; Gemma-3 has no official AWQ.
- **Mixed precision (later).** Quantize only some agents (e.g. 4-bit experts, BF16 aggregator). This needs two vLLM engines in one run and is closer to the topology/risk search question.
