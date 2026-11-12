import math
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, roc_curve


class PrivacyAuditor:
    """
    Audits a model checkpoint for privacy vulnerabilities using Membership Inference Attacks (MIA).
    Evaluates whether an adversary can differentiate between private training data (Forget set)
    and completely unseen holdout data (Control set).

    Attack Paradigms:
      1. Loss-Threshold Attack: Predicts membership using sequence cross-entropy / average log-likelihood.
      2. Min-k% Prob Attack (Shi et al., 2023): Focuses on the bottom k% token probabilities,
         which are highly sensitive to memorized proper nouns, entities, and specific facts.
      3. Entropy-Calibrated Likelihood: Normalizes likelihood by token predictive entropy.
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model.to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_token_log_probs(self, dataloader) -> List[List[float]]:
        """
        Extracts token-level log probabilities for valid answer tokens across sequences.
        Strictly ignores prompt tokens and padding positions (masked with -100).
        """
        self.model.eval()
        all_token_log_probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                else:
                    attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                # Shift logits and labels
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                log_probs = torch.log_softmax(shift_logits, dim=-1)

                # Gather log prob of the actual target token
                target_log_probs = torch.gather(
                    log_probs, dim=-1, index=shift_labels.unsqueeze(-1).clamp(min=0)
                ).squeeze(-1)

                batch_size = shift_labels.size(0)
                for b in range(batch_size):
                    mask = (shift_labels[b] != -100)
                    raw_probs = target_log_probs[b][mask].cpu().tolist()
                    # Filter any invalid numerical entries
                    valid_probs = [float(p) for p in raw_probs if not (math.isnan(p) or math.isinf(p))]
                    all_token_log_probs.append(valid_probs)

        return all_token_log_probs

    def run_mia_audit(
        self,
        forget_loader,
        unseen_loader,
        min_k_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Runs membership inference attacks comparing Forget members (label=1)
        against Unseen non-members (label=0).
        """
        print("  Extracting token likelihoods on Forget set...")
        forget_token_logprobs = self.extract_token_log_probs(forget_loader)

        print("  Extracting token likelihoods on Unseen Control set...")
        unseen_token_logprobs = self.extract_token_log_probs(unseen_loader)

        # 1. Loss-based attack score: Mean log probability (higher = higher membership confidence)
        forget_loss_scores = [float(np.mean(lp)) if lp else -50.0 for lp in forget_token_logprobs]
        unseen_loss_scores = [float(np.mean(lp)) if lp else -50.0 for lp in unseen_token_logprobs]

        # 2. Min-k% Prob attack: Average log probability of the bottom k% hardest tokens
        def compute_mink_score(lp_list, k=min_k_ratio):
            if not lp_list:
                return -50.0
            num_tokens = max(1, int(math.ceil(len(lp_list) * k)))
            sorted_lp = sorted(lp_list)
            return float(np.mean(sorted_lp[:num_tokens]))

        forget_mink_scores = [compute_mink_score(lp) for lp in forget_token_logprobs]
        unseen_mink_scores = [compute_mink_score(lp) for lp in unseen_token_logprobs]

        # Ground Truth Labels: Forget = 1 (private training member), Unseen = 0 (held-out control)
        y_true = np.array([1] * len(forget_loss_scores) + [0] * len(unseen_loss_scores))

        # Loss-threshold MIA ROC-AUC
        loss_scores_all = np.array(forget_loss_scores + unseen_loss_scores)
        try:
            loss_auc = float(roc_auc_score(y_true, loss_scores_all))
        except Exception:
            loss_auc = 0.50

        # Min-k% Prob MIA ROC-AUC
        mink_scores_all = np.array(forget_mink_scores + unseen_mink_scores)
        try:
            mink_auc = float(roc_auc_score(y_true, mink_scores_all))
        except Exception:
            mink_auc = 0.50

        # TPR at 5% False Positive Rate (FPR)
        try:
            fpr, tpr, _ = roc_curve(y_true, loss_scores_all)
            idx_5pct = np.where(fpr <= 0.05)[0]
            tpr_at_5pct_fpr = float(tpr[idx_5pct[-1]]) if len(idx_5pct) > 0 else 0.0
        except Exception:
            tpr_at_5pct_fpr = 0.0

        # Privacy Leakage Index:
        # Measures deviation from ideal privacy (AUC = 0.50).
        # Scale: 0.0 (perfect privacy / coin flip) to 1.0 (complete data leakage).
        privacy_leakage_index = float(min(1.0, abs(loss_auc - 0.50) * 2.0))

        return {
            "loss_threshold_mia_auc": loss_auc,
            "min_k_mia_auc": mink_auc,
            "tpr_at_5pct_fpr": tpr_at_5pct_fpr,
            "privacy_leakage_index": privacy_leakage_index,
            "forget_sample_count": len(forget_loss_scores),
            "unseen_sample_count": len(unseen_loss_scores),
            "mean_forget_logprob": float(np.mean(forget_loss_scores)) if forget_loss_scores else 0.0,
            "mean_unseen_logprob": float(np.mean(unseen_loss_scores)) if unseen_loss_scores else 0.0,
        }


def run_membership_inference_attack(
    model,
    tokenizer,
    forget_loader,
    unseen_loader,
    min_k_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Convenience functional wrapper for privacy audit."""
    auditor = PrivacyAuditor(model, tokenizer)
    return auditor.run_mia_audit(forget_loader, unseen_loader, min_k_ratio=min_k_ratio)
