from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseAlgorithm(ABC):
    """
    Abstract base class that all unlearning algorithms inherit.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config or {}
        self.model.to(self.device)

    @abstractmethod
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Execute unlearning on self.model.

        Args:
            forget_loader: DataLoader yielding batches of forget set data.
            retain_loader: Optional DataLoader yielding batches of retain set data.
            **kwargs: Additional algorithm-specific hyperparameters.

        Returns:
            The unlearned model.
        """
        pass
