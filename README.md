# Aparichit

**A machine unlearning toolkit and benchmark for language models.**

*Aparichit* (अपरिचित) means "unfamiliar" or "unknown". That is the goal of this project: take a language model that has learned something, and make it behave as if that data had always been unfamiliar to it.


---

## Table of contents

1. [The problem](#1-the-problem)
2. [What this project is (and is not)](#2-what-this-project-is-and-is-not)
3. [Dataset and models](#3-dataset-and-models)
4. [Unlearning methods compared](#4-unlearning-methods-compared)
5. [How results are evaluated](#5-how-results-are-evaluated)
6. [Project file structure](#6-project-file-structure)
7. [Workflow: VS Code, GitHub, Colab, Drive](#7-workflow-vs-code-github-colab-drive)
8. [Getting started](#8-getting-started)
9. [Results](#9-results)
10. [Scope tiers](#10-scope-tiers)
11. [Limitations](#11-limitations)
12. [References](#12-references)

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

Wording rule: the project is *relevant to* deletion requests. It does not claim compliance.

## 3. Dataset and models

**Dataset: TOFU** (Task of Fictitious Unlearning, Maini et al., 2024). 200 synthetic author profiles with 20 QA pairs each (4,000 QA pairs). The authors are fictional, so any knowledge the model has after fine-tuning can be attributed to the fine-tuning.

| Split | Use in this project |
|-------|---------------------|
| `forget05` | Primary experiment: 5% of data (10 authors) to be forgotten |
| `retain95` | The remaining authors the model must keep. We hold out 10 of them as a **non-member control** for membership inference, leaving 180 retain authors |
| `real_authors`, `world_facts` | Utility checks for nearby and distant knowledge |
| `forget01`, `forget10` | Optional extensions |

**Models: GPT-2 Small (~124M parameters)**, chosen for feasibility on free Colab. (TOFU itself used Llama-2-7B and Phi-1.5, so our numbers are not comparable to TOFU's tables.)

| Model | Trained on | Role |
|-------|-----------|------|
| `M_base` | Pre-trained GPT-2 only | Lower bound: never saw TOFU |
| `M_full` | Fine-tuned on forget + retain authors | The model we unlearn |
| `M_retain` | Fine-tuned on retain authors only | **Gold standard**: what "never saw the forget set" looks like |

## 4. Unlearning methods compared

Four method families, five runs (Gradient Ascent has two variants).

| Run | Idea | Source | Status |
|-----|------|--------|--------|
| **GA** (Gradient Ascent) | Maximize loss on the forget set | TOFU | Paper baseline |
| **GD** (Gradient Difference) | GA on forget set + normal loss on retain set | TOFU | Paper baseline |
| **Fisher forgetting** | Add noise to weights, scaled inversely to their importance (Fisher information) for the retain set | Golatkar et al., 2020 | Adapted from image models to GPT-2 |
| **Task vector negation** | Build a vector from a forget-only fine-tune and subtract it from `M_full` | Ilharco et al., 2023 | Adapted from task-level to entity-level forgetting |
| **Generic-label fine-tuning** | Fine-tune toward what a model that never saw the target would predict ("reinforcement bootstrapping") | Eldan & Russinovich, 2023 | Adapted / simplified |

Everything adapted is labelled as adapted. None of these methods was evaluated by its authors on GPT-2 with TOFU.

## 5. How results are evaluated

Every unlearned model is compared against `M_retain`.

| Layer | Metrics | Origin |
|-------|---------|--------|
| **Primary** | Probability, ROUGE-L, Truth Ratio, Model Utility, Forget Quality (KS test) | TOFU |
| **Supplementary** | Loss-threshold Membership Inference Attack (AUC) with held-out non-member control | Ours |
| **Supplementary** | Perplexity on forget set, retain set, and general English | Ours |
| **Supplementary** | LLM-as-a-judge with 4 labels: `MEMORIZED`, `GIBBERISH`, `GENERIC/INVENTED`, `REFUSAL` (about 50 responses manually checked) | Ours |
| **Cost** | Wall-clock time and memory per method | Ours |

Results are read on the **Forget Quality vs Model Utility** plane, using epoch-wise checkpoints. No single metric is treated as proof of forgetting.

## 6. Project file structure

```
Aparichit/
│
├── unlearn_ai/                       # The Python library. All real logic lives here.
│   ├── __init__.py                   #   Package entry; exposes Unlearner, Evaluator, LLMJudge
│   ├── config.py                     #   Paths (CKPT_DIR from env var), seed, defaults
│   ├── data.py                       #   Loads TOFU, builds forget/retain/control splits
│   ├── finetuner.py                  #   Trains M_full, M_retain, and the forget-only model
│   ├── unlearner.py                  #   Orchestrator: picks an algorithm and runs it
│   ├── utils.py                      #   set_seed, logging, timing, checkpoint save/load
│   │
│   ├── algorithms/                   #   The unlearning methods
│   │   ├── __init__.py
│   │   ├── base.py                   #     Abstract class every method inherits
│   │   ├── gradient_ascent.py        #     GA and GD (use_retain flag)
│   │   ├── fisher.py                 #     Retain-set Fisher information + noise
│   │   ├── task_vectors.py           #     Forget-only task vector + lambda sweep
│   │   └── generic_label_ft.py       #     Anchors, reinforced model, generic labels
│   │
│   ├── evaluator/                    #   Everything that measures results
│   │   ├── __init__.py
│   │   ├── tofu_metrics.py           #     Probability, ROUGE-L, Truth Ratio, Utility, Forget Quality
│   │   ├── mia_attack.py             #     Loss-threshold membership inference + AUC
│   │   ├── perplexity.py             #     Perplexity on forget / retain / general text
│   │   ├── llm_judge.py              #     4-label LLM judge
│   │   └── relearn.py                #     Optional: re-learn time test
│   │
│   ├── report_generator.py           #   Builds the PDF evaluation report (built last)
│   └── cli.py                        #   Command line: python -m unlearn_ai run --config ...
│
├── experiments/                      # YAML configs that drive the pipeline
│   ├── quick_test.yaml               #   Tiny run for fast local testing on CPU
│   ├── mvp.yaml                      #   Minimum viable run: GA, GD, Task Vector on forget05
│   └── full_benchmark.yaml           #   All methods, all evaluations
│
├── data/
│   └── splits/                       # Seeded split files (committed for reproducibility)
│       ├── forget05_authors.json     #   The 10 authors to forget
│       ├── retain_authors.json       #   The 180 retain authors
│       └── control_authors.json      #   The 10 held-out non-member authors
│
├── notebooks/                        # Interactive demo notebooks (thin: call the library)
│   ├── 01_optional_mnist_sanity_check.ipynb
│   ├── 02_gpt2_finetuning_and_retain_model.ipynb
│   ├── 03_baseline_evaluation.ipynb
│   ├── 04_gradient_ascent_and_difference.ipynb
│   ├── 05_fisher_forgetting.ipynb
│   ├── 06_task_vectors.ipynb
│   ├── 07_generic_label_finetuning.ipynb
│   ├── 08_full_comparison.ipynb      #   Main demo notebook
│   └── 09_optional_cross_model.ipynb
│
├── results/                          # Small output files, committed to Git
│   ├── metrics/                      #   JSON/CSV per run (M_base, M_full, GA, ...)
│   ├── plots/                        #   Forget-quality vs utility plane, etc.
│   └── judge_labels/                 #   Judge outputs and manual labels
│
├── checkpoints/                      # Model weights. NOT in Git (Colab saves these to Drive)
├── reports/                          # Generated PDF reports. NOT in Git
├── research_docs/
│   └── PROJECT_MASTER_DOCUMENT_v2.md #   Full project design and paper provenance
│
├── tests/                            # Small tests, also run by GitHub Actions
│   ├── test_data_splits.py           #   Splits are disjoint and reproducible
│   ├── test_metrics.py               #   AUC / KS on dummy numbers
│   └── test_algorithms_smoke.py      #   One training step on a tiny model
│
├── scripts/
│   └── colab_setup.sh                # Clone, install, set UNLEARN_CKPT_DIR in Colab
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # Runs tests on every push
│
├── .gitignore                        # Keeps checkpoints, reports, secrets out of Git
├── .env.example                      # Lists needed env vars (no real keys)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── setup.py                          # Makes the library installable (pip install -e .)
```

### What lives where, in one line each

| Folder | Purpose |
|--------|---------|
| `unlearn_ai/` | The library: data, training, algorithms, evaluation, CLI |
| `experiments/` | Config files that decide what a run does |
| `data/splits/` | Fixed author lists so every run uses the same splits |
| `notebooks/` | Demo and walkthrough, calling into the library |
| `results/` | Small, committed outputs: metrics, plots, judge labels |
| `checkpoints/` | Heavy model weights, kept on Drive, never in Git |
| `tests/` | Quick correctness checks, run on every push |

## 7. Workflow: VS Code, GitHub, Colab, Drive

```
VS Code (write code) → git push → GitHub → git pull in Colab → train → save to Drive
```

| Tool | Role |
|------|------|
| **VS Code** | Write and test code locally (`quick_test.yaml` runs on CPU) |
| **GitHub** | Source of truth for code, configs, splits, small results |
| **Google Colab** | GPU for training and unlearning runs |
| **Google Drive** | Stores model checkpoints so they survive Colab disconnects |

Rules:

- Checkpoints go to Drive via the `UNLEARN_CKPT_DIR` environment variable, never into Git.
- Keep notebooks thin and logic in `unlearn_ai/`, so almost all edits happen in VS Code.
- Fix seeds and commit the split files for reproducibility.
- Keep API keys (for the LLM judge) in Colab Secrets or a local `.env`, never in the repo.

## 8. Getting started

**Local (VS Code):**

```bash
git clone git@github.com:Emotional-Vantage/aparichit.git
cd Aparichit
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest tests/
python -m unlearn_ai run --config experiments/quick_test.yaml
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
!python -m unlearn_ai run --config experiments/mvp.yaml
```

Switch the runtime to a GPU (Runtime → Change runtime type) before running.

## 10. Scope tiers

| Tier | Contents |
|------|----------|
| **Minimum viable** | GPT-2 Small, `forget05` only, `M_base` / `M_full` / `M_retain`, GA, GD, Task Vector, TOFU metrics + perplexity, one plane plot and one table |
| **Recommended** | MVP + Fisher + Generic-label FT, loss-based MIA with held-out control, 4-label LLM judge, epoch-wise checkpoints, cost table, 2-3 seeds |
| **Optional** | `forget01` / `forget10`, GPT-2 Medium, Phi-1.5, re-learn time, paraphrase robustness, CLI and PDF report polish |

If time runs short, cut the wrappers (CLI, PDF report), not the retain model or the evaluation.

## 11. Limitations

- **Approximate, not certified.** No method here gives a formal guarantee.
- **Small model, one benchmark.** Results may not transfer to larger models or real pretraining data.
- **Fine-tuned knowledge only.** We unlearn what was learned in fine-tuning, not pretraining knowledge.
- **Entity-level.** Whole authors are forgotten, not single facts.
- **Adapted methods.** Fisher, task vectors and generic-label FT were not tested on this setup by their authors.
- **Metric limits.** MIA and perplexity are weak alone, Forget Quality can be high for a broken model, and the LLM judge is imperfect.
- **No legal claim.** This project does not provide GDPR or any other compliance.

## 12. References

- Maini, Feng, Schwarzschild, Lipton, Kolter. *TOFU: A Task of Fictitious Unlearning for LLMs.* 2024.
- Golatkar, Achille, Soatto. *Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks.* CVPR 2020.
- Ilharco et al. *Editing Models with Task Arithmetic.* ICLR 2023.
- Eldan, Russinovich. *Who's Harry Potter? Approximate Unlearning in LLMs.* 2023.
- Cao, Yang. *Towards Making Systems Forget with Machine Unlearning.* IEEE S&P 2015.
- Ginart, Guan, Valiant, Zou. *Making AI Forget You: Data Deletion in Machine Learning.* NeurIPS 2019.
- Neel, Roth, Sharifi-Malvajerdi. *Descent-to-Delete: Gradient-Based Methods for Machine Unlearning.* ALT 2021.