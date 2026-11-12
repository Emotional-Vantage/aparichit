"""
Aparichit Evaluation Suite:
Provides standardized metrics (ROUGE, Truth Ratio, KS-Test Forget Quality)
and evaluation runners across Forget, Retain, and Unseen splits.
"""

from .metrics import compute_rouge, compute_ks_test, compute_perplexity, compute_truth_ratio
from .evaluator import ModelEvaluator

__all__ = [
    "compute_rouge",
    "compute_ks_test",
    "compute_perplexity",
    "compute_truth_ratio",
    "ModelEvaluator",
]

