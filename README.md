# Aparichit

**A machine unlearning toolkit and benchmark for language models.**

*Aparichit* (अपिरिचित) means "unfamiliar" or "unknown". That is the goal of this project: take a language model that has learned and memorized sensitive information, and make it behave as if that data had always been unfamiliar to it.

---

## Table of contents

1. [The Problem](#1-the-problem)
2. [What This Project Is (and Is Not)](#2-what-this-project-is-and-is-not)
3. [Dataset, Split Architecture, and Models](#3-dataset-split-architecture-and-models)
4. [Unlearning Methods Compared](#4-unlearning-methods-compared)
5. [Evaluation Protocol & Three-Way Evaluation](#5-evaluation-protocol--three-way-evaluation)
6. [Computational Cost & Timing Decomposition](#6-computational-cost--timing-decomposition)
7. [Project File Structure](#7-project-file-structure)
8. [Getting Started](#8-getting-started)
9. [Running Experiments](#9-running-experiments)
10. [Empirical Benchmark Results](#10-empirical-benchmark-results)
11. [Limitations](#11-limitations)
12. [References](#12-references)

---

## 1. The Problem

Language models do not only generalize patterns; they can **memorize verbatim facts**. Specific personal profiles, proprietary text, and sensitive entities from training data can be extracted through targeted prompting. Deleting the source record from an underlying database does not erase its footprint from the neural network's parameters.

Retraining an entire model from scratch without the forgotten data is computationally prohibitive at scale. **Approximate machine unlearning** seeks to selectively remove the influence of targeted data points efficiently, without degrading the model's core capabilities.

**Core Research Question:**
> *How close can approximate unlearning methods bring a fine-tuned language model to a model that was never trained on the forgotten data, and what does each method cost in retained knowledge, general fluency, privacy resistance, and computation?*

Success is evaluated **relative to an independently trained retain-only reference model ($M_{\text{retain}}$)**—not merely by inducing catastrophic failure on the forgotten facts.

---

## 2. What This Project Is (and Is Not)

| It **Is** | It **Is Not** |
|:---|:---|
| A controlled empirical benchmark comparing four approximate unlearning families under a unified, reproducible protocol | A proprietary or ad-hoc unlearning heuristic |
| A standardized evaluation suite comparing all models against a genuine, trained $M_{\text{retain}}$ reference model | A theoretical guarantee or formal cryptographic erasure certificate |
| A rigorous three-way evaluation across Forget, Retain, and Do-Not-Train data partitions | A system providing legal (GDPR / CCPA) compliance guarantees |
| An open-source research and experimentation toolkit with separated computational timing | A reproduction of a single paper |

---

## 3. Dataset, Split Architecture, and Models

### 3.1 Dataset: TOFU (Task of Fictitious Unlearning)
We benchmark on the **TOFU** dataset (Maini et al., 2024), containing 200 fictitious author profiles with 20 QA pairs each (4,000 QA pairs total). Because all biographies and entities are completely synthetic, any knowledge exhibited by the model originates strictly from our controlled training phases.

### 3.2 Standardized Split Architecture (85% Retain / 10% Forget / 5% Unseen Holdout)
The dataset is partitioned into three strictly disjoint sets:

| Partition | Proportion | Total QA Pairs | Role in Aparichit |
|:---|:---:|:---:|:---|
| **Forget Split** (`forget10`) | **10%** | 400 QA pairs (20 authors) | Private facts targeted for unlearning. |
| **Retain Partition** (`retain90` rows 0..3400) | **85%** | 3,400 QA pairs (170 authors) | Factual domain knowledge that must remain intact. |
| **Do-Not-Train Control** (`retain90` rows 3400..3600) | **5%** | 200 QA pairs (10 authors) | **Strictly excluded from all training and hyperparameter tuning.** Serves as the unseen baseline distribution for control perplexity and privacy auditing. |

#### Rationale for Active Retain Training Size (`retain_active_size = 400`)
In fine-tuning and unlearning, using the full 3,400 retain examples against 400 forget examples creates an 8.5:1 data imbalance. Such dominance allows retain gradients to overwhelm forget objectives, leading to inadequate unlearning dynamics. 

To maintain experimental balance, Aparichit uses an **active retain subset of 400 QA pairs** (`retain_active_size = 400`) for training $M_{\text{full}}$, $M_{\text{retain}}$, and retain regularizers, while keeping the full 3,400 examples available for evaluation.

### 3.3 Model States
All experiments utilize GPT-2 Small (~124M parameters), trained and evaluated across four distinct model states:

```
                  ┌──────────────────────┐
                  │ Pretrained GPT-2 Base│
                  │       (M_base)       │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│     Fine-Tuned Full Model     │ │     Retain-Only Reference     │
│           (M_full)            │ │          (M_retain)           │
│  Trained on Forget + Retain   │ │ Trained on Active Retain Only │
│         (800 QA total)        │ │         (400 QA total)        │
└───────────────┬───────────────┘ └───────────────────────────────┘
                │
                ▼ (Apply Unlearning Algorithm)
┌───────────────────────────────┐
│        Unlearned Model        │
│        (M_unlearned_*)        │
└───────────────────────────────┘
```

1. **$M_{\text{base}}$ (Pretrained Reference)**: Untouched OpenAI GPT-2 checkpoint; lower bound evaluating performance prior to TOFU fine-tuning.
2. **$M_{\text{full}}$ (Memorized Baseline)**: Fine-tuned from $M_{\text{base}}$ on both forget (400 QA) and active retain (400 QA) examples.
3. **$M_{\text{retain}}$ (Retain-Only Reference)**: Independently fine-tuned from $M_{\text{base}}$ **strictly on the 400 active retain QA pairs**. It is trained without the forget or unseen control examples, serving as the retrained reference model for ideal unlearning behavior.
4. **$M_{\text{unlearned}}$ (Unlearned Model)**: Result of applying an approximate unlearning algorithm to $M_{\text{full}}$.

---

## 4. Unlearning Methods Compared

Aparichit implements and benchmarks four major unlearning paradigms:

### Method 1: Calibrated Gradient Difference (GD)
- **Concept**: Ascends the loss on forget batches while descending the loss on retain batches:
  $$\mathcal{L}_{\text{GD}} = -\text{clamp}(\mathcal{L}_{\text{forget}}, \tau) + \alpha_{\text{retain}} \cdot \mathcal{L}_{\text{retain}}$$
- **Implementation**: Enforces active loss thresholding ($\tau = 5.2$) to prevent loss divergence, accompanied by gradient clipping and AdamW optimization (`unlearn_ai/algorithms/gradient_difference.py`).

### Method 2: Empirical Diagonal Fisher Retain-Anchored Unlearning
- **Concept**: Computes the empirical diagonal Fisher Information Matrix $\mathcal{F}_{\text{retain}}$ on retain batches to quantify parameter importance for retained knowledge:
  $$\mathcal{F}_{ii} = \frac{1}{|D_{\text{retain}}|} \sum_{x \in D_{\text{retain}}} \left( \frac{\partial \log p(x)}{\partial \theta_i} \right)^2$$
  Unlearning ascends forget loss while penalizing parameter displacement proportionally to $\mathcal{F}_{\text{retain}}$:
  $$\mathcal{L}_{\text{Fisher}} = -\text{clamp}(\mathcal{L}_{\text{forget}}, \tau) + \alpha \mathcal{L}_{\text{retain}} + \beta \sum_i \mathcal{F}_{ii} (\theta_i - \theta_{0, i})^2$$
- **Precision**: Targets multi-head attention projections (`c_attn`, `c_proj`) and feed-forward MLP layers (`c_fc`, `c_proj`), leaving token embeddings and layer normalizations untouched. This is retain-anchored gradient unlearning, not random noise injection.

### Method 3: Magnitude-Trimmed Task Vector Negation
- **Concept**: Trains a dedicated forget branch $M_{\text{forget}}$ on forget data, extracts the task vector $\tau = \theta_{M_{\text{forget}}} - \theta_{M_{\text{base}}}$, applies magnitude trimming (Yadav et al., 2023) to zero out the bottom 80% diffuse noise, and subtracts the scaled vector from $M_{\text{full}}$:
  $$\theta_{\text{unlearned}} = \theta_{M_{\text{full}}} - \lambda \cdot \text{Trim}(\tau)$$
- **Precision**: Single-vector magnitude trimming rather than multi-task sign consensus voting. The unseen control set is strictly excluded from $\lambda$ calibration.

### Method 4: Generic-Label Fine-Tuning & Reinforcement Bootstrapping
- **Tier 1 (Anchor-Sanitized Generic Fine-Tuning)**: Replaces sensitive author entities with generic descriptions (e.g. *"an established author recognized for their literary contributions"*) and fine-tunes with a retain anchor.
- **Tier 2 (Reinforcement Bootstrapping)**: Constructs an over-trained forget branch $M_{\text{reinforced}}$, computes the logit discrepancy over base model predictions ($\Delta v = \text{ReLU}(v_{\text{reinf}} - v_{\text{base}})$), and penalizes probability shifts via KL divergence toward $v_{\text{base}} - \alpha \Delta v$.

---

## 5. Evaluation Protocol & Three-Way Evaluation

### 5.1 Mandatory Three-Way Evaluation
Every model checkpoint ($M_{\text{base}}, M_{\text{full}}, M_{\text{retain}}, M_{\text{unlearned}}$) is evaluated on three distinct data partitions:
1. **Forget Set**: Assesses whether factual memorization has been erased (target: loss approaches $M_{\text{retain}}$ / unseen control, ROUGE-L drops).
2. **Retain Set**: Assesses whether core domain knowledge is preserved (target: loss and ROUGE-L remain stable).
3. **Do-Not-Train Set (Unseen Control)**: Assesses general model capability and checks for catastrophic collapse on held-out text.

### 5.2 Comparative Reference Analysis
For every unlearned model, we report:
- **$\Delta \text{Metric}(M_{\text{unlearned}}, M_{\text{retain}})$**: Difference relative to the independently retrained retain-only reference model.
- **$\Delta \text{Metric}(M_{\text{unlearned}}, M_{\text{full}})$**: Net change relative to the initial memorized model.

### 5.3 Privacy Auditing & Direction-Adjusted MIA ROC-AUC
We execute Membership Inference Attacks (MIA) comparing forget samples against unseen control samples using sequence log-likelihood and Min-$k$% probability.
- **Score Direction**: Higher log probability indicates higher membership confidence.
- **Direction-Adjusted Metric**: If an unlearning algorithm aggressively over-penalizes forget tokens, their log probability can drop below unseen holdout text (raw AUC < 0.50). An adversary can invert their decision rule to achieve $(1 - \text{AUC})$ accuracy. We therefore report:
  $$\text{AUC}_{\text{adjusted}} = \max(\text{AUC}, 1 - \text{AUC})$$

---

## 6. Computational Cost & Timing Decomposition

To enable transparent resource comparisons, Aparichit explicitly breaks down execution wall-clock time with CUDA synchronization into three independent phases:

1. **Initial Training Time ($T_{\text{training}}$)**: Time required to fine-tune $M_{\text{full}}$ and $M_{\text{retain}}$ from base weights.
2. **Unlearning Method Time ($T_{\text{method}} = T_{\text{auxiliary}} + T_{\text{unlearning}}$)**:
   - $T_{\text{auxiliary}}$: Pre-computation time (e.g. Fisher curvature computation, task-vector forget branch training, or Tier 2 reinforced model preparation).
   - $T_{\text{unlearning}}$: Core unlearning parameter update time.
3. **Evaluation Time ($T_{\text{evaluation}}$)**: Time spent evaluating the three data splits and executing the privacy audit.

---

## 7. Project File Structure

```
aparichit/
│
├── unlearn_ai/                       # Core Library & Benchmark Suite
│   ├── __init__.py                   #   Package exports and metadata
│   ├── cli.py                        #   Unified CLI entrypoint (`aparichit`)
│   ├── config.py                     #   Standardized paths and session configuration
│   ├── data.py                       #   TOFU dataset loading, active retain subset, 85/10/5 splits
│   ├── utils.py                      #   Seeding, synchronized timing, checkpoint I/O
│   │
│   ├── algorithms/                   #   Core Algorithm Implementations
│   │   ├── base.py                   #     Abstract base class (`BaseAlgorithm`)
│   │   ├── gradient_difference.py    #     Method 1: Calibrated Gradient Difference
│   │   ├── fisher.py                 #     Method 2: Fisher Retain-Anchored Unlearning
│   │   ├── task_vectors.py           #     Method 3: Magnitude-Trimmed Task Vector Negation
│   │   └── generic_label_ft.py       #     Method 4: Generic-Label Fine-Tuning (Tiers 1 & 2)
│   │
│   ├── evaluation/                   #   Standardized Evaluation Pipeline
│   │   ├── evaluator.py              #     Three-way split loss, perplexity, ROUGE evaluation
│   │   ├── metrics.py                #     ROUGE-L, Kolmogorov-Smirnov test, truth ratio
│   │   └── display.py                #     Terminal scorecards and probe comparisons
│   │
│   ├── auditing/                     #   Privacy & Security Auditing
│   │   └── privacy_audit.py          #     Membership Inference Attacks & direction-adjusted AUC
│   │
│   ├── benchmark/                    #   Comparative Multi-Checkpoint Benchmarking
│   │   └── compare_all.py            #     Leaderboard generator and JSON serializer
│   │
│   └── visualization/                #   Publication-Grade Vector Visualizations
│       └── plots.py                  #     Pareto frontier, privacy-utility, and loss curves
│
├── outputs/                          # Evaluation outputs (Markdown, CSV, JSON, SVG)
│   └── README.md                     #   Overview of generated benchmark artifacts
│
├── tests/                            # Automated Unit Test Suite
│   ├── test_splits.py                #   Validates 85/10/5 proportions and disjointness
│   ├── test_retain_model.py          #   Validates M_retain training isolation and metadata
│   ├── test_data_masking.py          #   Verifies label masking on prompt tokens
│   ├── test_metrics.py               #   ROUGE, KS-test, and perplexity mathematical tests
│   ├── test_audit.py                 #   Membership Inference Attack tests
│   ├── test_cli.py                   #   CLI subcommands and argument validation
│   └── test_visualization.py         #   SVG generation and styling tests
│
├── train_model.py                    # Trains M_base, M_full, and M_retain
├── run_train_retain.py               # Dedicated trainer for M_retain (Retain-only reference)
├── run_unlearning_gd.py              # Method 1: Gradient Difference runner
├── run_unlearning_fisher.py          # Method 2: Fisher Retain-Anchored runner
├── run_unlearning_tv.py              # Method 3: Magnitude-Trimmed Task Vector runner
├── run_unlearning_generic.py         # Method 4: Generic-Label FT runner
├── run_benchmark.py                  # Master comparative benchmark runner
├── requirements.txt                  # Python dependencies
└── setup.py                          # Package installer (`pip install -e .`)
```

---

## 8. Getting Started

### Installation
Clone the repository and install the package:

```bash
git clone https://github.com/Emotional-Vantage/aparichit.git
cd aparichit
pip install -r requirements.txt
pip install -e .
```

Run the unit test suite:
```bash
python -m unittest discover tests
```

---

## 9. Running Experiments

### Step 1: Train Reference Models ($M_{\text{base}}$, $M_{\text{full}}$, and $M_{\text{retain}}$)
Trains all three reference models in sequence with active retain size = 400:
```bash
python train_model.py --session experiment_1 --epochs 13 --retain-active-size 400
```
Or train $M_{\text{retain}}$ independently:
```bash
python run_train_retain.py --session experiment_1 --epochs 13 --retain-active-size 400
```

### Step 2: Run Unlearning Algorithms
Execute each unlearning method against the trained $M_{\text{full}}$:
```bash
# Method 1: Gradient Difference
python run_unlearning_gd.py --session experiment_1 --epochs 4 --lr 2.5e-5

# Method 2: Fisher Retain-Anchored
python run_unlearning_fisher.py --session experiment_1 --epochs 4 --beta-fisher 25.0

# Method 3: Task Vector Negation
python run_unlearning_tv.py --session experiment_1 --forget-epochs 12 --trim-ratio 0.20

# Method 4: Generic-Label Fine-Tuning (Tier 2)
python run_unlearning_generic.py --session experiment_1 --tier 2 --epochs 5
```

### Step 3: Run Comparative Benchmark & Generate Leaderboard
Evaluate all discovered model checkpoints across Forget, Retain, and Do-Not-Train splits, computing relative metrics against $M_{\text{full}}$ and $M_{\text{retain}}$:
```bash
python run_benchmark.py --checkpoint-dir checkpoints/experiment_1 --output-dir outputs
```

This generates:
- `outputs/leaderboard.md`: Comparison table for all model states across all three partitions.
- `outputs/leaderboard.csv`: Tabular CSV of all metrics.
- `outputs/benchmark_results.json`: Complete serialized metrics and timing data.
- SVG visualizations of Pareto frontiers, privacy-utility tradeoffs, and loss curves.

---

## 10. Empirical Benchmark Results

Below are the empirical results from the standardized three-way evaluation across all reference baselines and approximate unlearning methods on the TOFU benchmark (400 Forget QA pairs, 400 Active Retain QA pairs, 200 Unseen Control QA pairs):

### 10.1 Master Leaderboard

| Model / Method | Forget Loss | Retain Loss | DNT Loss | Forget ROUGE-L | Retain ROUGE-L | $\Delta \mathcal{L}_{\text{fgt}}$ vs $M_{\text{retain}}$ | $\Delta \mathcal{L}_{\text{ret}}$ vs $M_{\text{retain}}$ | MIA Raw AUC | MIA Adj AUC | Unlearn Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference: Pretrained ($M_{\text{base}}$)** | 2.819 | 2.928 | 2.836 | 0.193 | 0.162 | -1.250 | +2.895 | 0.506 | 0.506 | — |
| **Baseline: Memorized Full ($M_{\text{full}}$)** | 0.042 | 0.029 | 3.978 | 0.870 | 0.927 | -4.027 | -0.004 | 0.998 | 0.998 | — |
| **Reference: Retain-Only ($M_{\text{retain}}$)** | 4.069 | 0.033 | 4.109 | 0.273 | 0.927 | +0.000 | +0.000 | 0.502 | 0.502 | — |
| **Method 1: Gradient Difference (GD)** | 10.324 | 0.015 | 5.479 | 0.195 | 0.961 | +6.255 | -0.018 | 0.276 | 0.724 | 93.1s |
| **Method 2: Fisher Retain-Anchored** | 9.113 | 0.013 | 4.994 | 0.237 | 0.963 | +5.044 | -0.020 | 0.349 | 0.651 | 120.0s |
| **Method 3: Task Vector Negation** | 3.482 | 1.879 | 3.851 | 0.119 | 0.130 | -0.587 | +1.846 | 0.638 | 0.638 | 175.8s |
| **Method 4: Generic-Label Tier 1** | 0.337 | 0.008 | 4.350 | 0.751 | 0.972 | -3.732 | -0.025 | 0.999 | 0.999 | 73.1s |

*All runs executed on NVIDIA CUDA GPU with seed 42. Visualizations and full serialized metrics are generated in `outputs/` (`leaderboard.md`, `leaderboard.csv`, `benchmark_results.json`).*

### 10.2 Key Empirical Insights

1. **Near-Total Extraction Vulnerability in $M_{\text{full}}$**:
   - Fine-tuning without unlearning drives forget loss down to `0.042` and generation recall to `0.870`.
   - The Membership Inference Attack (MIA) achieves a raw ROC-AUC of `0.998` against unseen control text, demonstrating that standard regularized fine-tuning produces near-deterministic extraction vulnerability.

2. **The Independently Retrained Reference Model ($M_{\text{retain}}$)**:
   - Trained exclusively on the 400 active retain examples, $M_{\text{retain}}$ exhibits nearly identical loss on both the forget set (`4.069`) and the unseen Do-Not-Train control partition (`4.109`, delta `0.040`).
   - Its MIA ROC-AUC is `0.502`, confirming the theoretical ideal of random guessing ($\approx 0.50$) and establishing a rigorous empirical anchor for unlearning success.

3. **Method 2 (Fisher Retain-Anchored) Achieves Pareto Optimality**:
   - Curvature anchoring via the empirical diagonal Fisher information matrix suppresses forget generation (ROUGE-L `0.237`, approaching $M_{\text{retain}}$'s `0.273`) while maintaining superior retain fidelity (`0.013` loss, `0.963` ROUGE-L).
   - Constrains collateral drift on the held-out control set to `4.994` (outperforming GD's `5.479`).
   - Provides the most resilient privacy defense among gradient techniques with direction-adjusted MIA ROC-AUC of `0.651`.

4. **The Over-Unlearning Paradox (Method 1: GD)**:
   - Gradient Difference suppresses forget recall to `0.195` but drives forget loss up to `10.324`—far above natural unseen text (`5.479`).
   - This causes raw MIA AUC to drop to `0.276`. An adversary simply inverts the decision rule to achieve an adjusted attack accuracy of `0.724` ($1 - \text{AUC}$), proving empirically that over-unlearning creates exploitable privacy leakage identical to memorization.

5. **Task Vector Negation (Method 3) & Collateral Retain Trade-off**:
   - Single-vector magnitude trimming suppresses factual recall to `0.119` while maintaining holdout stability (`3.851` DNT loss, `0.638` MIA AUC without rule inversion).
   - However, parameter subtraction without active retain gradient regularization degrades retain loss to `1.879` and retain ROUGE-L to `0.130`, demonstrating the necessity of active retain regularization during vector arithmetic.

6. **Generic-Label Fine-Tuning (Method 4) vs. Distributional Unlearning**:
   - Delivers the fastest runtime (`73.1s`), highest retain recall (`0.972`), and well-behaved control loss (`4.350`).
   - However, because it replaces entities with generic sentences without gradient ascent on forget tokens, forget loss remains low (`0.337`) and MIA ROC-AUC remains at `0.999`. This highlights that surface-level behavioral rewriting does not constitute distributional unlearning.

---

## 11. Limitations

- **Approximate, Not Certified**: Heuristic unlearning methods do not provide formal differential privacy guarantees.
- **Model Scale**: Benchmarked on GPT-2 Small (~124M parameters) for reproducibility; scaling to 7B+ models introduces different parameter redundancy dynamics.
- **Task Scope**: Evaluated on synthetic author profiles where facts are localized to biographical QA pairs.
- **Metric Limitations**: No single metric completely characterizes unlearning; KS tests, ROUGE-L, and MIA ROC-AUC must be inspected together.

---

## 12. References

- Maini et al. *TOFU: A Task of Fictitious Unlearning for LLMs.* 2024.
- Golatkar et al. *Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks.* CVPR 2020.
- Ilharco et al. *Editing Models with Task Arithmetic.* ICLR 2023.
- Yadav et al. *Resolving Interference When Merging Models (TIES-Merging).* NeurIPS 2023.
- Eldan, Russinovich. *Who's Harry Potter? Approximate Unlearning in LLMs.* Microsoft Research, 2023.
- Shi et al. *Detecting Pretraining Data from Large Language Models (Min-k% Prob).* ICLR 2024.
- Carlini et al. *Membership Inference Attacks From First Principles.* IEEE S&P 2022.
