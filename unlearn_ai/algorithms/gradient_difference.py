import copy
import math
import random
from typing import Optional, Dict, Any
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


class GradientDifference(BaseAlgorithm):
    """
    Calibrated Gradient Difference (GD) and Gradient Ascent (GA) unlearning.

    - "GA" (Pure Gradient Ascent):
        Pushes forget loss UP with active thresholding.

    - "GD" (Gradient Difference):
        Pushes forget loss UP while simultaneously anchoring retain loss DOWN:
        Loss = - alpha_forget * L_forget + beta_retain * L_retain
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "GD",  # "GD" or "GA"
        use_retain: Optional[bool] = None,
        alpha_retain: float = 1.2,
        target_forget_loss: float = 6.0,
        lr: float = 2.5e-5,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(model, tokenizer, device, config)
        if use_retain is not None:
            self.mode = "GD" if use_retain else "GA"
        else:
            self.mode = mode.upper()

        self.alpha_retain = alpha_retain
        self.target_forget_loss = target_forget_loss
        self.lr = lr
        self.max_grad_norm = max_grad_norm

    def evaluate_loss(self, dataloader: DataLoader) -> float:
        return evaluate_loss(self.model, dataloader, self.device)

    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 4,
        verbose: bool = True,
        **kwargs,
    ) -> nn.Module:
        if self.mode == "GD" and retain_loader is None:
            raise ValueError("mode='GD' requires a valid retain_loader. For forget-only ascent, set mode='GA'.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        unlearn_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if verbose:
            mode_desc = "Calibrated Gradient Difference (GD)" if self.mode == "GD" else "Gradient Ascent (GA)"
            print(f"\n--- Starting {mode_desc} Unlearning for {epochs} epochs ---")
            print(f"{'Epoch':<8} | {'Forget Loss':<13} | {'Forget PPL':<12} | {'Retain Loss':<13} | {'Retain PPL':<12}")
            print("-" * 70)

        for epoch in range(1, epochs + 1):
            self.model.train()
            retain_iter = iter(retain_loader) if (self.mode == "GD" and retain_loader is not None) else None

            for forget_batch in forget_loader:
                unlearn_optimizer.zero_grad(set_to_none=True)

                # 1. Forget Batch Ascent with Active Thresholding
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
                if self.mode == "GD" and retain_iter is not None:
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

                # 3. Clip gradients and update
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                unlearn_optimizer.step()

            # Per-epoch evaluation
            curr_f = self.evaluate_loss(forget_loader)
            f_ppl = math.exp(min(curr_f, 20.0))

            if self.mode == "GD" and retain_loader is not None:
                curr_r = self.evaluate_loss(retain_loader)
                r_ppl = math.exp(min(curr_r, 20.0))
                if verbose:
                    print(f"Epoch {epoch:02d} | {curr_f:<13.4f} | {f_ppl:<12.1f} | {curr_r:<13.4f} | {r_ppl:<12.2f}")
            else:
                if verbose:
                    print(f"Epoch {epoch:02d} | {curr_f:<13.4f} | {f_ppl:<12.1f}")

        return self.model
