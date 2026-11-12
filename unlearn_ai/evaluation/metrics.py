import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import ks_2samp
from rouge_score import rouge_scorer


def compute_rouge(
    predictions: List[str],
    references: List[str],
    metric_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes mean ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) across paired texts.
    Returns average F1 measures scaled to [0, 1].
    Safely handles empty strings and mismatched lengths.
    """
    if metric_names is None:
        metric_names = ["rouge1", "rouge2", "rougeL"]

    if not predictions or not references:
        return {m: 0.0 for m in metric_names}

    scorer = rouge_scorer.RougeScorer(metric_names, use_stemmer=True)
    accum = {m: [] for m in metric_names}

    for pred, ref in zip(predictions, references):
        pred_clean = (pred or "").strip()
        ref_clean = (ref or "").strip()

        # If both are empty strings, score is 1.0; if one is empty, 0.0
        if not pred_clean and not ref_clean:
            for m in metric_names:
                accum[m].append(1.0)
            continue
        elif not pred_clean or not ref_clean:
            for m in metric_names:
                accum[m].append(0.0)
            continue

        scores = scorer.score(ref_clean, pred_clean)
        for m in metric_names:
            accum[m].append(scores[m].fmeasure)

    return {
        m: float(np.mean(accum[m])) if accum[m] else 0.0
        for m in metric_names
    }


def compute_perplexity(sample_losses: List[float]) -> float:
    """
    Computes Perplexity = exp(mean cross-entropy loss per token).
    Safely filters out NaNs, Infinities, and clamps to 1e5 to prevent math overflow.
    """
    if not sample_losses:
        return float("nan")

    valid_losses = [float(l) for l in sample_losses if not (math.isnan(l) or math.isinf(l))]
    if not valid_losses:
        return float("nan")

    mean_loss = float(np.mean(valid_losses))
    if mean_loss > 15.0:
        return 1e5
    return float(math.exp(mean_loss))


def compute_ks_test(
    forget_losses: List[float],
    unseen_losses: List[float],
) -> Dict[str, float]:
    """
    Two-sample Kolmogorov-Smirnov test between evaluation losses on the Forget set
    and the Unseen Holdout Control set.

    In the TOFU benchmark:
    - High p-value (> 0.05): We cannot reject that Forget and Unseen distributions match.
      The model treats forgotten knowledge the same as knowledge it never saw.
    - Low p-value (< 0.01): Significant divergence between Forget and Unseen distributions.
    """
    clean_forget = [float(l) for l in forget_losses if not (math.isnan(l) or math.isinf(l))]
    clean_unseen = [float(l) for l in unseen_losses if not (math.isnan(l) or math.isinf(l))]

    if len(clean_forget) < 2 or len(clean_unseen) < 2:
        return {"ks_stat": 1.0, "p_value": 0.0}

    stat, p_val = ks_2samp(clean_forget, clean_unseen)
    return {
        "ks_stat": float(stat),
        "p_value": float(p_val),
    }


def compute_truth_ratio(
    target_losses: List[float],
    perturbed_losses: Optional[List[float]] = None,
) -> float:
    """
    Computes the Truth Ratio:
    Ratio of probability assigned to the ground-truth target answer vs perturbed answers.
    When perturbed answers are not provided, computes the normalized probability:
    P(target) = exp(-mean_target_loss).
    """
    clean_target = [float(l) for l in target_losses if not (math.isnan(l) or math.isinf(l))]
    if not clean_target:
        return 0.0

    mean_target_loss = float(np.mean(clean_target))
    prob_target = math.exp(-min(mean_target_loss, 20.0))

    if perturbed_losses:
        clean_perturb = [float(l) for l in perturbed_losses if not (math.isnan(l) or math.isinf(l))]
        if clean_perturb:
            mean_perturb_loss = float(np.mean(clean_perturb))
            prob_perturb = math.exp(-min(mean_perturb_loss, 20.0))
            denom = prob_target + prob_perturb
            return float(prob_target / denom) if denom > 1e-12 else 0.5

    return float(prob_target)
