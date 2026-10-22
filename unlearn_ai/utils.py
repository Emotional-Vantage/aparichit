import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, Any


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def is_checkpoint_valid(path: Path) -> bool:
    """Checks whether a directory contains valid saved Hugging Face model files."""
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or any(path.glob("model-*.safetensors"))
    )
    return has_config and has_weights


def ask_user_choice(prompt_text: str, default: str = "1") -> str:
    """
    Asks the user for input with a default option.
    Handles non-interactive environments (CI, pipes, non-TTY) by returning default.
    """
    if not sys.stdin.isatty():
        print(f"{prompt_text} -> [Non-interactive mode: using default {default}]")
        return default
    try:
        user_input = input(f"{prompt_text} (default: [{default}]): ").strip()
        return user_input if user_input else default
    except (EOFError, KeyboardInterrupt):
        print(f"\n[Using default: {default}]")
        return default


def save_checkpoint(
    model: Any,
    tokenizer: Optional[Any],
    save_dir: Path,
    model_name: str = "Model",
) -> None:
    """Saves model and tokenizer to the target directory."""
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Saving {model_name} to: {save_dir}")
    model.save_pretrained(str(save_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(save_dir))
    print(f"✓ {model_name} saved successfully at: {save_dir}")


def load_checkpoint(
    load_dir: Path,
    device: Any,
) -> Tuple[Any, Any]:
    """Loads model and tokenizer from a checkpoint directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n>>> Loading checkpoint from: {load_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(load_dir))
    model = AutoModelForCausalLM.from_pretrained(str(load_dir))
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"✓ Checkpoint loaded successfully from: {load_dir}")
    return model, tokenizer
