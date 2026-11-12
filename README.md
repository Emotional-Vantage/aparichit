# Aparichit

**A machine unlearning toolkit and benchmark for language models.**

*Aparichit* means "unfamiliar" or "unknown". That is the goal of this project: take a language model that has learned something, and make it behave as if that data had always been unfamiliar to it.


---

## Table of contents

1. [The problem](#1-the-problem)
2. [What this project is (and is not)](#2-what-this-project-is-and-is-not)
3. [Dataset and models](#3-dataset-and-models)
4. [Unlearning methods compared](#4-unlearning-methods-compared)
5. [How results are evaluated](#5-how-results-are-evaluated)
6. [Project file structure](#6-project-file-structure)
7. [Getting started](#7-getting-started)
8. [Results & Benchmark Leaderboard](#8-results--benchmark-leaderboard)
9. [Scope tiers](#9-scope-tiers)
10. [Limitations](#10-limitations)
11. [References](#11-references)

---

## 1. The problem

Language models do not only learn, they can **memorize**. Specific facts and phrasing from the training data can be recovered from a trained model. If a person asks for their data to be removed, or a copyright owner wants their content taken out, deleting the record from a database does not remove its influence from the model's weights.

The naive fix is to retrain from scratch without the offending data, which is very expensive for large models. **Approximate unlearning** tries to remove the influence of specific data cheaply, without full retraining.

**Research question:**

> *How close can approximate unlearning methods bring a fine-tuned language model to a model that was never trained on the forgotten data, and what does each method cost in retained knowledge, general ability, and output quality?*

Success is always measured **relative to a retain-only reference model** (a model trained without the forget data), not by simply making the model fail on the forgotten data.

## 2. What this project is (and is not)

| It **is** | It **is not** |
|-----------|---------------|
| A controlled empirical comparison of existing approximate unlearning methods on one benchmark, with a shared evaluation protocol | A new unlearning algorithm |
| A reusable evaluation and experiment toolkit | A reproduction of a single paper |
| An analysis of the forgetting-vs-utility trade-off, including how each method fails | A proof or certificate that data is erased |
| A Trustworthy AI study of *evaluating* deletion behaviour | A system that provides legal (GDPR etc.) compliance |


## 3. Dataset and models

**Dataset: TOFU** (Task of Fictitious Unlearning, Maini et al., 2024). 200 synthetic author profiles with 20 QA pairs each (4,000 QA pairs total). The authors are fictional, ensuring any knowledge the model has after fine-tuning originates solely from training.

**Strict Split Architecture (85% Retain / 10% Forget / 5% Unseen Holdout):**
This project strictly enforces the **85 / 10 / 5** split across all algorithms and benchmarks, with no exceptions:

| Split Partition | Proportion | Size | Role in This Project |
|-----------------|:----------:|:----:|----------------------|
| **Forget Split** (`forget10`) | **10%** | 20 Authors (400 QA pairs) | Targeted private facts to be permanently unlearned. |
| **Retain Split** (from `retain90`) | **85%** | 170 Authors (3,400 QA pairs) | Core domain knowledge that must remain preserved. |
| **Unseen Control** (from `retain90`) | **5%** | 10 Authors (200 QA pairs) | **Strictly excluded from training.** Serves as the ground-truth reference distribution for Kolmogorov-Smirnov Forget Quality and Membership Inference Attacks (MIA). |

**Models: GPT-2 Small (~124M parameters)**, chosen for feasibility and reproducibility.

| Model | Trained on | Role |
|-------|-----------|------|
| `M_base` | Pre-trained GPT-2 only | Lower bound: never saw TOFU |
| `M_full` | Fine-tuned on forget (10%) + retain (85%) authors | The memorized model we unlearn |
| `M_retain` | Fine-tuned on retain (85%) authors only | **Gold standard**: what "never saw the forget set" looks like |

## 4. Unlearning methods compared

Four primary unlearning algorithm families are evaluated:

| Method Name | Key Concept | Source | Role & Implementation |
|:---|:---|:---|:---|
| **GD (Gradient Difference)** | Pushes forget loss UP (ascent) while regularizing retain loss DOWN (descent). Includes pure Gradient Ascent (GA) as an unregularized ablation mode. | TOFU (Maini et al., 2024) | Baseline optimization; enforces active ascent thresholding (`gradient_difference.py`). |
| **Fisher Forgetting** | Injects selective noise into parameter weights, scaled inversely to parameter importance (diagonal Fisher information matrix) on retain facts. | Adapted from Golatkar et al., 2020 | Selective parameter perturbation without retain corruption (`fisher.py`). |
| **Task Vector Negation** | Constructs an unlearning direction from a forget-fine-tuned branch and subtracts it from $M_{\text{full}}$ using TIES magnitude trimming and sign consensus. | Adapted from Ilharco et al., 2023; Yadav et al., 2023 | Instantaneous weight-space arithmetic requiring zero backpropagation at unlearn time (`task_vectors.py`). |
| **Generic-Label Fine-Tuning** | Replaces private entities with safe, domain-appropriate generic descriptions via reinforced logit subtraction ($M_{\text{reinforced}}$). | Adapted from Eldan & Russinovich (Microsoft Research, 2023) | Smooth knowledge overwrite eliminating catastrophic forgetting (`generic_label_ft.py`). |


## 5. How results are evaluated

Every unlearned model is compared against `M_retain`.

| Layer | Metrics | Origin |
|-------|---------|--------|
| **Primary** | Probability, ROUGE-L, Truth Ratio, Model Utility, Forget Quality (KS test) | TOFU |
| **Supplementary** | Loss-threshold Membership Inference Attack (AUC) with held-out non-member control | Ours |
| **Supplementary** | Perplexity on forget set, retain set, and general English | Ours |
| **Supplementary** | LLM-as-a-judge with 4 labels: `MEMORIZED`, `GIBBERISH`, `GENERIC/INVENTED`, `REFUSAL` (about 50 responses manually checked) | Ours |
| **Cost** | Wall-clock time and memory per method | Ours |


## 6. Project file structure

```
Aparichit/
│
├── unlearn_ai/                       # Enterprise Python Library & Unlearning Suite
│   ├── __init__.py                   #   Package exports and metadata
│   ├── cli.py                        #   Unified CLI entrypoint (`aparichit`)
│   ├── config.py                     #   Checkpoint paths and session management
│   ├── data.py                       #   Enforces strict 85/10/5 split & offline TOFU caching
│   ├── utils.py                      #   Seeding, I/O, checkpoint validation
│   │
│   ├── algorithms/                   #   Core Unlearning Algorithm Implementations
│   │   ├── base.py                   #     Abstract base class (`BaseAlgorithm`)
│   │   ├── gradient_difference.py    #     Method 1: GD (Gradient Difference) & GA ablation
│   │   ├── fisher.py                 #     Method 2: Fisher Information Masking
│   │   ├── task_vectors.py           #     Method 3: Task Vector Negation + TIES Merging
│   │   └── generic_label_ft.py       #     Method 4: Generic-Label Fine-Tuning (Tiers 1 & 2)
│   │
│   ├── evaluation/                   #   Standardized Evaluation & UI Display
│   │   ├── evaluator.py              #     Per-sample loss, perplexity, ROUGE evaluation
│   │   ├── metrics.py                #     ROUGE-L, KS-test p-value, truth ratio
│   │   └── display.py                #     Rich terminal scorecards & sentence audit panels
│   │
│   ├── auditing/                     #   Privacy & Security Auditing
│   │   └── privacy_audit.py          #     Membership Inference Attacks (MIA ROC-AUC)
│   │
│   ├── benchmark/                    #   Comparative Multi-Checkpoint Benchmarking
│   │   └── compare_all.py            #     Generates leaderboard and benchmark JSON
│   │
│   └── visualization/                #   Publication-Grade Vector Visualizations
│       └── plots.py                  #     Pareto frontier, privacy-utility, and loss curves
│
├── outputs/                          # Evaluation outputs & publication charts
│   ├── leaderboard.md                #   Standardized Markdown leaderboard table
│   ├── leaderboard.csv               #   Machine-readable CSV results
│   ├── benchmark_results.json        #   Full metric JSON records
│   ├── pareto_frontier.svg           #   Vector Pareto frontier visualization
│   ├── privacy_utility_tradeoff.svg  #   Vector privacy-utility tradeoff visualization
│   └── loss_distributions.svg        #   Vector empirical loss distribution curves
│
├── checkpoints/                      # Model weights (M_base, M_full, M_unlearned_*)
│
├── tests/                            # Comprehensive Automated Test Suite
│   ├── test_splits.py                #   Enforces strict 85/10/5 split proportions & disjointness
│   ├── test_data_masking.py          #   Verifies label masking (-100) on question tokens
│   ├── test_metrics.py               #   ROUGE, KS-test, and perplexity mathematical tests
│   ├── test_audit.py                 #   Membership Inference Attack (MIA) tests
│   ├── test_cli.py                   #   CLI subcommands and argument validation
│   └── test_visualization.py         #   SVG generation and styling tests
│
├── train_model.py                    # Phase 1 Model Trainer (M_base and M_full memorization)
├── run_unlearning_gd.py              # Method 1: Gradient Difference (GD) Runner
├── run_unlearning_fisher.py          # Method 2: Fisher Information Masking Runner
├── run_unlearning_tv.py              # Method 3: Task Vector Negation + TIES Runner
├── run_unlearning_generic.py         # Method 4: Generic-Label Fine-Tuning Runner
├── run_local.py                      # Interactive terminal menu workflow runner
├── requirements.txt                  # Python dependencies
└── setup.py                          # Makes package installable (`pip install -e .`)
```

### What lives where, in one line each

| Component | Purpose |
|-----------|---------|
| `unlearn_ai/` | Core library: data, algorithms, evaluation, auditing, and CLI |
| `run_unlearning_*.py` | Direct standalone runners for individual unlearning methods |
| `train_model.py` | Trains $M_{\text{base}}$ and $M_{\text{full}}$ on the 85/10 split |
| `run_local.py` | Interactive terminal interface to train, unlearn, and evaluate |
| `outputs/` | Visualizations (`.svg`), leaderboard (`.md`/`.csv`), and benchmark results (`.json`) |
| `checkpoints/` | Model weights for $M_{\text{full}}$ and all unlearned models |
| `tests/` | Automated unit test suite verifying splits, metrics, auditing, and UI |

## 7. Getting started

**Local (VS Code):**

```bash
git clone git@github.com:Emotional-Vantage/aparichit.git
cd Aparichit
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
python -m unittest discover tests

# 1. Run full comparative benchmark across all checkpoints:
aparichit benchmark

# 2. Evaluate a specific checkpoint on Forget/Retain/Unseen splits:
aparichit evaluate --checkpoint checkpoints/m_unlearned_gd --max-generate 3 --audit

# 3. Perform a Membership Inference Attack (MIA) privacy audit:
aparichit audit --checkpoint checkpoints/m_unlearned_gd
```

**Colab:**

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["UNLEARN_CKPT_DIR"] = "/content/drive/MyDrive/Aparichit/checkpoints"
os.makedirs(os.environ["UNLEARN_CKPT_DIR"], exist_ok=True)

!git clone https://github.com/Suryanshu-01/Aparichit.git
%cd Aparichit
!pip install -e . -q

# Run automated tests
!python -m unittest discover tests

# Execute Gradient Difference unlearning
!python run_unlearning_gd.py

# Evaluate with visual scorecard
!aparichit evaluate --checkpoint checkpoints/m_unlearned_gd --max-generate 3 --audit
```

Switch the runtime to a GPU (Runtime → Change runtime type) before running.

## 8. Results & Benchmark Leaderboard

The table below presents the standardized benchmark results across the four evaluated algorithms on the TOFU benchmark (strictly 85% Retain / 10% Forget / 5% Unseen Holdout split with `forget10`, GPT-2 Small, Seed 42). All evaluations were executed with the standardized [`aparichit benchmark`](file:///home/sridhar_1906/Documents/TAI_PROJECT/unlearn_ai/benchmark/compare_all.py) suite:

| Method / Checkpoint | Forget ROUGE-L (↓) | Retain ROUGE-L (↑) | Forget Quality ($p$-val ↑) | MIA ROC-AUC (≈0.50) | Privacy Leakage (↓) | Eval Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Memorized Full)** | 0.8970 | 1.0000 | 9.80e-85 | 0.9282 | 0.8564 | 8.3s |
| **Method 1 (Gradient Difference - GD)** | 0.4453 | 0.8915 | **1.18e-04** | **0.6017** | **0.2033** | 7.8s |
| **Method 2 (Fisher Masking)** | **0.1846** | 0.9636 | 1.71e-39 | 0.2179 | 0.5641 | 7.7s |
| **Method 3 (Task Vectors + TIES)** | 0.2793 | 0.6333 | 3.31e-55 | 0.8629 | 0.7258 | 8.1s |
| **Method 4 (Generic-Label Tier 2)** | 0.5691 | **1.0000** | 9.46e-79 | 0.9139 | 0.8279 | 7.7s |

### Key Findings & Insights:
1. **Memorization Vulnerability in Baseline:** The baseline fine-tuned model exhibits severe privacy leakage with an adversarial Membership Inference Attack (MIA) ROC-AUC of **0.9282** and Forget ROUGE-L of **0.8970**.
2. **Method 1: Gradient Difference (GD):** Achieves the best balanced privacy-defense profile, bringing MIA ROC-AUC closest to the ideal random guess of 0.50 (**0.6017**) and cutting privacy leakage down to **0.2033** while preserving **89.15%** retain knowledge.
3. **Method 2: Fisher Information Masking:** Delivers strong erasure among weight-noise approaches, suppressing Forget ROUGE-L to **0.1846** while maintaining **96.36%** factual recall on retain facts.
4. **Method 3: Task Vectors + TIES Merging:** Offers extreme computational efficiency with direct weight arithmetic and sign-consensus regularization, cutting Forget ROUGE-L down to **0.2793**.
5. **Method 4: Generic-Label Fine-Tuning (Tier 2):** Preserves **100.0%** retain knowledge without any degradation while steering forget queries toward safe, generic domains.

### Publication-Grade Visualizations:
Generated vector SVG charts are available in `outputs/`:
- **Pareto Frontier (Erasure vs. Utility):** [`outputs/pareto_frontier.svg`](file:///home/sridhar_1906/Documents/TAI_PROJECT/outputs/pareto_frontier.svg)
- **Privacy vs. Utility Tradeoff:** [`outputs/privacy_utility_tradeoff.svg`](file:///home/sridhar_1906/Documents/TAI_PROJECT/outputs/privacy_utility_tradeoff.svg)
- **Cross-Entropy Loss Distributions:** [`outputs/loss_distributions.svg`](file:///home/sridhar_1906/Documents/TAI_PROJECT/outputs/loss_distributions.svg)

## 9. Scope tiers

| Tier | Contents | Status |
|------|----------|:------:|
| **Minimum viable** | GPT-2 Small, 85/10/5 split (`forget10`), `M_base` / `M_full`, GD (Gradient Difference), Task Vector, TOFU metrics + perplexity, Pareto plot and table | **Completed** |
| **Recommended** | MVP + Fisher + Generic-label FT, loss-based MIA with held-out 5% control, cost table | **Completed** |
| **Production Product**| Unified CLI (`aparichit`), SVG Pareto Frontier & Privacy plots, Interactive HTML Dashboard, full automated unit test suite | **Completed** |

If time runs short, cut the wrappers (CLI, PDF report), not the retain model or the evaluation.

## 10. Limitations

- **Approximate, not certified.** No method here gives a formal guarantee.
- **Small model, one benchmark.** Results may not transfer to larger models or real pretraining data.
- **Fine-tuned knowledge only.** We unlearn what was learned in fine-tuning, not pretraining knowledge.
- **Entity-level.** Whole authors are forgotten, not single facts.
- **Adapted methods.** Fisher, task vectors and generic-label FT were not tested on this setup by their authors.
- **Metric limits.** MIA and perplexity are weak alone, Forget Quality can be high for a broken model, and the LLM judge is imperfect.
- **No legal claim.** This project does not provide GDPR or any other compliance.

## 11. References

- Maini, Feng, Schwarzschild, Lipton, Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* 2024.
- Golatkar, Achille, Soatto. *Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks.* CVPR 2020.
- Ilharco et al. *Editing Models with Task Arithmetic.* ICLR 2023.
- Eldan, Russinovich. *Who's Harry Potter? Approximate Unlearning in LLMs.* 2023.
- Cao, Yang. *Towards Making Systems Forget with Machine Unlearning.* IEEE S&P 2015.
- Ginart, Guan, Valiant, Zou. *Making AI Forget You: Data Deletion in Machine Learning.* NeurIPS 2019.
- Neel, Roth, Sharifi-Malvajerdi. *Descent-to-Delete: Gradient-Based Methods for Machine Unlearning.* ALT 2021.
