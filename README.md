# PropUQ-MAS

[![EMNLP 2026](https://img.shields.io/badge/EMNLP_2026-Main_Conference-7b2cbf)](https://2026.emnlp.org/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Accepted to the EMNLP 2026 Main Conference.**

Official implementation of **PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems**.

> **This fork** adds the paper's propagation rule alongside the released one, NormMSP variants, and Table 1 tooling. See [`DECISIONS.md`](DECISIONS.md) for every change, [`docs/experiment_plan_2026-11-26.md`](docs/experiment_plan_2026-11-26.md) for the reproduction plan, and [`docs/experiment_plan_2026-11-27_quantization.md`](docs/experiment_plan_2026-11-27_quantization.md) for the quantization study.

## Highlights

- Propagation-aware and interaction-aware uncertainty for LLM multi-agent systems.
- Compatible with different MAS topologies, model families, and local UQ estimators.
- Training-free, online-computable, and linear in the execution-graph size.

## Installation

We recommend Python 3.10 on Linux with an NVIDIA GPU and a CUDA-compatible PyTorch installation.

```bash
git clone https://github.com/yaokunliu/PropUQ-MAS.git
cd PropUQ-MAS

conda create -n mas python=3.10 -y
conda activate mas
pip install -r requirements.txt
```

The default launcher uses [vLLM](https://docs.vllm.ai/). Set `USE_VLLM=0` to use the Hugging Face Transformers backend instead.

Graphviz is optional and is needed only for SVG graph export:

```bash
conda install -c conda-forge graphviz -y
```

Authenticate first if a selected Hugging Face model requires access approval:

```bash
huggingface-cli login
```

## Data

| Task | Source |
|---|---|
| GSM8K | Downloaded automatically from [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) |
| MBPP-Plus | Downloaded automatically from [`evalplus/mbppplus`](https://huggingface.co/datasets/evalplus/mbppplus) |
| MedQA | Loaded from `data/medqa.json` |

The MedQA JSON records must contain `query`, `options`, and `answer` fields. Please follow the licenses and terms of the original datasets.

## Quick Start

Run a 10-example smoke test:

```bash
MAX_SAMPLES=10 bash run/run_experiment.sh
```

Run the full default experiment:

```bash
bash run/run_experiment.sh
```

The default configuration is `Qwen/Qwen3-8B + MedQA + hierarchical + role + Verb + 4 agents`.

For a custom experiment, call `run.py` directly:

```bash
python run.py \
  --method mas \
  --model_name Qwen/Qwen3-8B \
  --task medqa \
  --mas_topology hierarchical \
  --mas_node_num 4 \
  --mas_prompt role \
  --local_uncertainty_mode Verb \
  --max_samples 10 \
  --seed 42 \
  --use_vllm
```

Use `--force_generation` to ignore cached predictions and `--force_uq` to overwrite matching UQ outputs. Run `python run.py --help` for all options.

## Supported Configurations

| Option | Values |
|---|---|
| Dataset | `gsm8k`, `medqa`, `mbppplus` |
| Model | `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `google/gemma-3-12b-it` |
| MAS topology | `sequential`, `hierarchical`, `decentralized` |
| Local UQ | `Verb`, `NormMSP` (formerly `MSP`) |
| Prompt | `norole`, `role` |

`NormMSP` is length-normalized MSP, `1 - exp(mean token log-prob)` (equivalently `1 - 1/perplexity`), not sequence-level `1 - P(output)`. A single `NormMSP` run reports two local estimators from the same generations:

- `NormMSP`: all generated tokens, including the `<think>` block.
- `NormMSP-Answer`: only tokens after the last `</think>`. If thinking never closes (truncated output), it is set to `1.0`; if the output has no think tags, the whole output counts as the answer.

Both exclude the tokens of the appended `<alpha .../>` / `<local_uncertainty .../>` tags. Adoption scores (alpha) are now elicited in every mode, including `NormMSP`; earlier releases set alpha to 1 in MSP mode and did not ask for it. To reproduce those runs exactly, pass `--no_alpha_elicitation` (prompts are then identical to the original MSP prompts) and read the `Alpha1-Normalized` metrics row; such runs are written to `outputs/<mode>_noalpha/`. Without the switch, outputs go to `outputs/NormMSP/`, so older `outputs/MSP/` caches are not reused. The paper's "w/o alpha" ablation on the same generations is the `Alpha1` row (see Reproducibility Notes).

Role-specialized prompts are available for four-agent sequential and hierarchical systems. Other topology/node-count combinations automatically use the generic graph prompt.

The paper uses temperature `0.6`, top-p `0.95`, repetition penalty `1.05`, and a maximum generation length of 8,192 tokens for reasoning or 32,768 for code generation. Main experiments use four agents on NVIDIA GH200 120 GB GPUs.

## Outputs

Generated files are written under `outputs/`:

```text
outputs/<uq_mode>/<model>/raw/raw_preds_*.jsonl
outputs/<uq_mode>/<model>/uq/uq_preds_*.jsonl
outputs/<uq_mode>/<model>/uq/uq_metrics_*.json
outputs/mas_graphs/<graph_config>/
```

The metrics JSON reports accuracy, runtime, AUROC, Prediction Rejection Ratio (PRR), ECE (15 bins) and Brier score for local and propagation-aware uncertainty.

Generate MAS graph artifacts without loading a model:

```bash
bash run/prepare_mas_graphs.sh
```

Set `RENDER_SVG=0` if Graphviz is unavailable.

## Sampling-Based Baselines

The repository includes MedQA reproductions for:

- [MATU](sampling_baselines/matu/README.md)
- [UProp](sampling_baselines/uprop/README.md)

See [`sampling_baselines/comparison.md`](sampling_baselines/comparison.md) for consolidated results and runtime information.

## Reproducibility Notes

- The default seed is `42`; stochastic generation may still vary across hardware and library versions.
- Reduce `GENERATE_BS` if GPU memory is limited.
- Raw-prediction caches do not encode every decoding option; use `--force_generation` after changing generation settings.
- Post-hoc replay computes four propagation variants side by side from the same generations:

  | Metrics row | Fields | Edge weight α | Meaning |
  |---|---|---|---|
  | `Prop-Uncertainty-Final` | `prop_uncertainty*` | self-reported, used as is | Paper rule (Eq. 2) |
  | `Prop-Uncertainty-Final-Normalized` | `prop_norm_uncertainty*` | self-reported, divided by the node's sum | Original release rule; a single parent's weight is always 1 |
  | `Prop-Uncertainty-Final-Alpha1` | `prop_alpha1_uncertainty*` | 1 on every edge | Paper's "w/o α̂" ablation (Table 4) |
  | `Prop-Uncertainty-Final-Alpha1-Normalized` | `prop_alpha1_norm_uncertainty*` | 1 on every edge, divided by the node's sum (1/k) | Original release rule in MSP mode |

  With self-reported scores, a parent with a missing or zero score is skipped; the α = 1 variants ignore the reports. Re-running UQ on cached raw predictions (without `--force_generation`) is enough to get all four.
- Generated predictions and metrics are ignored by Git, so archive the relevant `outputs/` directory separately.

## Citation

The available manuscript is anonymized. Replace the provisional author field below with the camera-ready author list or official ACL Anthology BibTeX when available.

```bibtex
@inproceedings{propuqmas2026,
  title     = {{PropUQ-MAS}: Propagation-Aware Uncertainty Quantification for {LLM} Multi-Agent Systems},
  author    = {{PropUQ-MAS Authors}},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
}
```

## License

This project is released under the [MIT License](LICENSE). Models and datasets are subject to their respective licenses and terms of use.
