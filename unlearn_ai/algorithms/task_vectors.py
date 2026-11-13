import copy
import math
import time
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from .base import BaseAlgorithm
from .gradient_difference import evaluate_loss
from unlearn_ai.utils import SynchronizedTimer


def is_targeted_layer(name: str) -> bool:
    """
    Memorization of factual associations resides in linear projection layers:
      - Multi-head attention: c_attn, c_proj
      - Feed-Forward / MLP:   mlp.c_fc, mlp.c_proj
    STRICTLY EXCLUDES:
      - Embeddings:           wte, wpe
      - Layer Normalization:  ln_1, ln_2, ln_f
    """
    if any(ex in name for ex in ["wte", "wpe", "ln_1", "ln_2", "ln_f"]):
        return False
    if any(inc in name for inc in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]):
        return True
    return False


def apply_magnitude_trimming(
    tau: Dict[str, torch.Tensor],
    trim_ratio: float = 0.20,
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """
    Magnitude Trimming (Yadav et al., 2023):
    Retains top `trim_ratio` (e.g. 20%) highest-magnitude weight displacements;
    zeros out bottom (1 - trim_ratio) diffuse noise.
    """
    active_params = 0
    total_targeted = 0

    trimmed_tau = {}
    for name, diff in tau.items():
        if is_targeted_layer(name) and diff.dtype in (torch.float32, torch.float16):
            total_targeted += diff.numel()
            if trim_ratio < 1.0:
                abs_diff = torch.abs(diff)
                k_val = max(1, min(int((1.0 - trim_ratio) * abs_diff.numel()), abs_diff.numel() - 1))
                threshold = torch.kthvalue(abs_diff.flatten(), k_val).values
                mask = (abs_diff >= threshold).to(diff.dtype)
                trimmed_tau[name] = diff * mask
                active_params += int(mask.sum().item())
            else:
                trimmed_tau[name] = diff
                active_params += diff.numel()
        else:
            trimmed_tau[name] = diff

    return trimmed_tau, active_params, total_targeted


# Backward compatibility alias
apply_ties_trimming = apply_magnitude_trimming


class TaskVectorUnlearning(BaseAlgorithm):
    """
    Method 3: Magnitude-Trimmed Task Vector Negation for Language Model Unlearning.
    Reference: Ilharco et al. (ICLR 2023) task vector arithmetic with Yadav et al. (NeurIPS 2023) magnitude trimming.

    Methodological Precision:
      We apply magnitude trimming to zero out diffuse, low-magnitude noise from the task vector
      before negation. Note that full TIES involves multi-task sign consensus resolution across
      conflicting task vectors; because unlearning here operates on a single task vector
      (theta_{M_forget} - theta_{M_base}), we perform magnitude-trimmed task-vector negation.

    Formulation:
      1. Train forget branch: M_forget fine-tuned solely on forget set for deep convergence.
      2. Construct task vector: tau = theta_{M_forget} - theta_{M_base} on Attention & MLP projections.
      3. Apply magnitude trimming: retain top trim_ratio deltas, zeroing diffuse noise.
      4. Subtract scaled vector: theta_{unlearned} = theta_{M_full} - lambda * tau
      5. Tune lambda (or use fixed lambda) strictly WITHOUT touching the unseen control split.
    """

    def __init__(
        self,
        model: nn.Module,
        base_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        base_model_name: str = "openai-community/gpt2",
        trim_ratio: float = 0.20,
        forget_epochs: int = 12,
        forget_lr: float = 1e-4,
        target_retain_ceiling: float = 3.4,
        target_forget_loss: float = 5.0,
    ):
        super().__init__(model, tokenizer, device, config)
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.trim_ratio = trim_ratio
        self.forget_epochs = forget_epochs
        self.forget_lr = forget_lr
        self.target_retain_ceiling = target_retain_ceiling
        self.target_forget_loss = target_forget_loss
        self.tau_forget: Dict[str, torch.Tensor] = {}
        self.timing: Dict[str, float] = {}

    def extract_task_vector(
        self,
        forget_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Trains M_forget branch on forget data and extracts the trimmed task vector.
        """
        if self.base_model is None:
            if verbose:
                print(f"\nInitializing base reference model ({self.base_model_name})...")
            self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)
            if self.tokenizer and self.tokenizer.pad_token_id is not None:
                self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

        base_state_cpu = {k: v.detach().cpu().clone() for k, v in self.base_model.state_dict().items()}

        if verbose:
            print(f"\nTraining M_forget branch for {self.forget_epochs} epochs (Deep Convergence)...")

        model_forget = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)
        if self.tokenizer and self.tokenizer.pad_token_id is not None:
            model_forget.config.pad_token_id = self.tokenizer.pad_token_id

        forget_opt = torch.optim.AdamW(model_forget.parameters(), lr=self.forget_lr)
        scaler = torch.amp.GradScaler('cuda') if (self.device and self.device.type == 'cuda') else None

        for epoch in range(1, self.forget_epochs + 1):
            t_start = time.time()
            model_forget.train()
            running_loss = 0.0
            batches = 0

            for batch in forget_loader:
                forget_opt.zero_grad(set_to_none=True)
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model_forget(**inputs)
                        loss = outputs.loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(forget_opt)
                    torch.nn.utils.clip_grad_norm_(model_forget.parameters(), max_norm=1.0)
                    scaler.step(forget_opt)
                    scaler.update()
                else:
                    outputs = model_forget(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_forget.parameters(), max_norm=1.0)
                    forget_opt.step()

                running_loss += loss.item()
                batches += 1

            if verbose and (epoch % 3 == 0 or epoch == self.forget_epochs):
                avg_loss = running_loss / max(batches, 1)
                t_elapsed = time.time() - t_start
                print(f"M_forget Epoch {epoch:02d}/{self.forget_epochs:02d} | Loss: {avg_loss:.4f} | Time: {t_elapsed:.1f}s")

        # Compute raw task vector: tau = theta_forget - theta_base
        raw_tau = {}
        for name, p_forget in model_forget.state_dict().items():
            if is_targeted_layer(name):
                raw_tau[name] = (p_forget.detach().cpu() - base_state_cpu[name]).clone()
            else:
                raw_tau[name] = torch.zeros_like(base_state_cpu[name])

        del model_forget, forget_opt
        if scaler is not None:
            del scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Apply magnitude trimming
        self.tau_forget, active_p, total_p = apply_magnitude_trimming(raw_tau, trim_ratio=self.trim_ratio)

        if verbose:
            sparsity = 100.0 * (1.0 - active_p / max(total_p, 1))
            print(f"Magnitude Trimming: Kept {active_p:,} active parameters ({100.0 - sparsity:.1f}%), zeroed {sparsity:.1f}% diffuse noise.")

        return self.tau_forget

    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        reference_retain_model: Optional[nn.Module] = None,
        lambda_val: Optional[float] = None,
        candidate_lambdas: Optional[List[float]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Executes Task Vector Negation unlearning.
        The unseen control split is STRICTLY EXCLUDED from this method.
        Tuning operates either with a fixed lambda or by referencing the target forget loss
        and retain loss ceiling.
        """
        # 1. Auxiliary Phase: Train M_forget branch & extract trimmed task vector
        with SynchronizedTimer() as aux_timer:
            if not self.tau_forget:
                self.extract_task_vector(forget_loader, verbose=verbose)

        # 2. Unlearning Phase: Apply trimmed task vector subtraction
        with SynchronizedTimer() as unlearn_timer:
            full_state_cpu = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if lambda_val is not None:
                candidates = [lambda_val]
            else:
                candidates = candidate_lambdas or [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

            if verbose:
                print(f"\n--- Calibrating Lambda Sweep across {candidates} ---")
                print(f"{'Lambda':<8} | {'Forget Loss':<12} | {'Forget PPL':<12} | {'Retain Loss':<12} | {'Retain PPL':<12} | {'Score':<8}")
                print("-" * 75)

            best_lambda = candidates[0]
            best_score = float("inf")
            best_state = None

            # Determine target forget reference without touching unseen control
            if reference_retain_model is not None:
                ref_target = evaluate_loss(reference_retain_model, forget_loader, self.device)
            else:
                ref_target = self.target_forget_loss

            for l_val in candidates:
                trial_state = {}
                for name, p_full in full_state_cpu.items():
                    if is_targeted_layer(name):
                        trial_state[name] = (p_full - l_val * self.tau_forget[name]).to(self.device)
                    else:
                        trial_state[name] = p_full.to(self.device)

                self.model.load_state_dict(trial_state)

                l_f = evaluate_loss(self.model, forget_loader, self.device)
                l_r = evaluate_loss(self.model, retain_loader, self.device) if retain_loader else 0.0

                ppl_f = math.exp(min(l_f, 20.0))
                ppl_r = math.exp(min(l_r, 20.0))

                gap = abs(l_f - ref_target)
                penalty = 2.5 * max(0.0, l_r - self.target_retain_ceiling)
                score = gap + penalty

                if verbose:
                    print(f"{l_val:<8.2f} | {l_f:<12.4f} | {ppl_f:<12.2f} | {l_r:<12.4f} | {ppl_r:<12.2f} | {score:<8.4f}")

                if score < best_score:
                    best_score = score
                    best_lambda = l_val
                    best_state = trial_state

            if verbose:
                print("-" * 75)
                print(f"[OK] Selected optimal lambda* = {best_lambda} (Score = {best_score:.4f})")

            self.model.load_state_dict(best_state)

        self.timing = {
            "auxiliary_time": round(aux_timer.elapsed, 2),
            "unlearning_time": round(unlearn_timer.elapsed, 2),
            "total_unlearn_time": round(aux_timer.elapsed + unlearn_timer.elapsed, 2),
        }
        return self.model
