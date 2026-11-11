import copy
import math
import random
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import BaseAlgorithm


def evaluate_loss(model: nn.Module, dataloader: DataLoader, device: Optional[torch.device] = None) -> float:
    """Computes average cross-entropy loss over a dataloader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            if device is not None:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


class FisherUnlearning(BaseAlgorithm):
    """
    Method 2: Fisher-Constrained Unlearning on TOFU Benchmark.
    
    Targeting: Multi-head attention projections (c_attn, c_proj) and MLP layers.
    Objective:
      Loss = - clamp_ascent(L_forget, max=target_forget_loss)
             + alpha_retain * L_retain
             + beta_fisher * sum( F_retain * (theta - theta_0)^2 )
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        alpha_retain: float = 1.0,
        beta_fisher: float = 25.0,
        target_forget_loss: float = 6.0,
        lr: float = 2.5e-5,
        max_grad_norm: float = 1.0,
        target_keywords: Tuple[str, ...] = ("c_attn", "c_proj", "mlp"),
    ):
        super().__init__(model, tokenizer, device, config)
        self.alpha_retain = alpha_retain
        self.beta_fisher = beta_fisher
        self.target_forget_loss = target_forget_loss
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.target_keywords = target_keywords
        self.fisher_retain: Dict[str, torch.Tensor] = {}
        self.baseline_state: Dict[str, torch.Tensor] = {}

    def compute_fisher(self, retain_loader: DataLoader, verbose: bool = True) -> Dict[str, torch.Tensor]:
        """
        Computes the Empirical Diagonal Fisher Information Matrix over retain data
        for the targeted attention and MLP weight matrices.
        """
        self.model.eval()
        self.fisher_retain = {}
        targeted_params = 0

        for name, param in self.model.named_parameters():
            if any(k in name for k in self.target_keywords) and param.requires_grad:
                self.fisher_retain[name] = torch.zeros_like(param.data)
                targeted_params += param.numel()

        if verbose:
            print(f"\nComputing Empirical Fisher on {len(self.fisher_retain)} layers ({targeted_params:,} parameters)...")

        total_batches = 0
        for batch in retain_loader:
            self.model.zero_grad(set_to_none=True)
            if self.device is not None:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            outputs = self.model(**batch)
            outputs.loss.backward()

            for name, param in self.model.named_parameters():
                if name in self.fisher_retain and param.grad is not None:
                    self.fisher_retain[name] += param.grad.detach() ** 2

            total_batches += 1

        for name in self.fisher_retain:
            self.fisher_retain[name] /= max(total_batches, 1)

        if verbose:
            print(f"Successfully computed Fisher curvature across {total_batches} retain batches.")

        return self.fisher_retain

    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 4,
        verbose: bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Executes Fisher-constrained gradient unlearning.
        """
        if retain_loader is None:
            raise ValueError("Fisher unlearning requires a valid retain_loader to compute curvature and retain anchor.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cache baseline state theta_0 if not already cached
        self.baseline_state = copy.deepcopy(self.model.state_dict())

        # Compute retain Fisher if not already computed
        if not self.fisher_retain:
            self.compute_fisher(retain_loader, verbose=verbose)

        unlearn_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if verbose:
            print(f"\n--- Starting Fisher-Constrained Unlearning for {epochs} epochs ---")
            print(f"{'Epoch':<8} | {'Forget Loss':<13} | {'Forget PPL':<12} | {'Retain Loss':<13} | {'Retain PPL':<12}")
            print("-" * 70)

        for epoch in range(1, epochs + 1):
            self.model.train()
            retain_iter = iter(retain_loader)

            for forget_batch in forget_loader:
                unlearn_optimizer.zero_grad(set_to_none=True)

                # 1. Forget Batch Ascent (Only ascend if below target to prevent collapse)
                f_batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in forget_batch.items()
                }
                outputs_forget = self.model(**f_batch)
                loss_f = outputs_forget.loss

                if loss_f.item() < self.target_forget_loss:
                    (-loss_f).backward()
                del outputs_forget

                # 2. Retain Batch Descent
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)

                r_batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in retain_batch.items()
                }
                outputs_retain = self.model(**r_batch)
                loss_r = outputs_retain.loss
                (self.alpha_retain * loss_r).backward()
                del outputs_retain

                # 3. Fisher Elastic Anchor Penalty
                fisher_penalty = 0.0
                for name, param in self.model.named_parameters():
                    if name in self.fisher_retain:
                        diff = param - self.baseline_state[name]
                        fisher_penalty = fisher_penalty + (self.fisher_retain[name] * (diff ** 2)).sum()

                (self.beta_fisher * fisher_penalty).backward()

                # 4. Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                unlearn_optimizer.step()

            # Evaluation
            curr_f = evaluate_loss(self.model, forget_loader, self.device)
            curr_r = evaluate_loss(self.model, retain_loader, self.device)
            f_ppl = math.exp(min(curr_f, 20.0))
            r_ppl = math.exp(min(curr_r, 20.0))

            if verbose:
                print(f"Epoch {epoch:02d} | {curr_f:<13.4f} | {f_ppl:<12.1f} | {curr_r:<13.4f} | {r_ppl:<12.2f}")

        return self.model
