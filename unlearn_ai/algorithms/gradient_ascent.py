import math
from typing import Optional, Dict, Any, Union
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


class GradientAscent(BaseAlgorithm):
    """
    Gradient Difference (GD) and Gradient Ascent (GA) unlearning algorithm.

    - "GA" (Pure Gradient Ascent):
        Pushes forget loss UP:
        Loss = - clamp(Loss(forget), max=loss_clamp)

    - "GD" (Gradient Difference):
        Pushes forget loss UP while simultaneously anchoring retain loss DOWN:
        Loss = - clamp(Loss(forget), max=loss_clamp) + alpha_retain * Loss(retain)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "GD",  # "GD" or "GA"
        use_retain: Optional[bool] = None,
        alpha_retain: float = 1.0,
        loss_clamp: float = 5.0,
        lr: float = 5e-5,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(model, tokenizer, device, config)
        # Support both `mode="GD"/"GA"` and boolean `use_retain`
        if use_retain is not None:
            self.mode = "GD" if use_retain else "GA"
        else:
            self.mode = mode.upper()

        self.alpha_retain = alpha_retain
        self.loss_clamp = loss_clamp
        self.lr = lr
        self.max_grad_norm = max_grad_norm

    def evaluate_loss(self, dataloader: DataLoader) -> float:
        """Helper to compute evaluation loss on a dataloader."""
        return evaluate_loss(self.model, dataloader, self.device)

    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        verbose: bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Execute GA or GD unlearning with paired batches and per-epoch evaluation.
        """
        if self.mode == "GD" and retain_loader is None:
            raise ValueError("mode='GD' requires a valid retain_loader. For forget-only ascent, set mode='GA'.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        unlearn_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if verbose:
            mode_desc = "Gradient Difference (GD)" if self.mode == "GD" else "Gradient Ascent (GA)"
            print(f"\n--- Starting {mode_desc} Unlearning for {epochs} epochs ---")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_forget_loss = 0.0
            epoch_retain_loss = 0.0
            steps = 0

            # Pair each forget batch with a retain batch
            retain_iter = iter(retain_loader) if (self.mode == "GD" and retain_loader is not None) else None

            for forget_batch in forget_loader:
                unlearn_optimizer.zero_grad(set_to_none=True)

                # Move forget batch to device
                f_batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in forget_batch.items()
                }

                # 1. Forget Batch (Push Loss UP)
                outputs_forget = self.model(**f_batch)
                loss_f = outputs_forget.loss
                loss_f_clamped = torch.clamp(loss_f, max=self.loss_clamp)

                # Backpropagate forget gradient immediately to free its memory graph
                (-loss_f_clamped).backward()
                epoch_forget_loss += loss_f.item()
                del outputs_forget

                # 2. Retain Batch (Anchor Loss DOWN)
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

                    # Gradients accumulate by default in PyTorch: adds retain gradient to weights
                    (self.alpha_retain * loss_r).backward()
                    epoch_retain_loss += loss_r.item()
                    del outputs_retain

                # 3. Clip gradients and update parameters
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                unlearn_optimizer.step()
                steps += 1

            # Per-epoch evaluation on loaders
            curr_f_loss = self.evaluate_loss(forget_loader)
            curr_f_ppl = math.exp(min(curr_f_loss, 20.0))

            if self.mode == "GD" and retain_loader is not None:
                curr_r_loss = self.evaluate_loss(retain_loader)
                curr_r_ppl = math.exp(min(curr_r_loss, 20.0))
                if verbose:
                    print(
                        f"Epoch {epoch:02d}/{epochs:02d} | "
                        f"Forget Loss: {curr_f_loss:.4f} (PPL: {curr_f_ppl:8.1f}) | "
                        f"Retain Loss: {curr_r_loss:.4f} (PPL: {curr_r_ppl:8.1f})"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch:02d}/{epochs:02d} | "
                        f"Forget Loss: {curr_f_loss:.4f} (PPL: {curr_f_ppl:8.1f})"
                    )

        return self.model


def run_gradient_unlearning(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: Optional[DataLoader] = None,
    mode: str = "GD",
    alpha_retain: float = 1.0,
    loss_clamp: float = 5.0,
    lr: float = 5e-5,
    epochs: int = 5,
    max_grad_norm: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Standalone function to run Gradient Difference or Gradient Ascent unlearning.
    """
    unlearner = GradientAscent(
        model=model,
        device=device,
        mode=mode,
        alpha_retain=alpha_retain,
        loss_clamp=loss_clamp,
        lr=lr,
        max_grad_norm=max_grad_norm,
    )
    return unlearner.unlearn(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        epochs=epochs,
        verbose=verbose,
    )
