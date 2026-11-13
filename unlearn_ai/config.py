import os
from pathlib import Path
from typing import Optional, Dict

DEFAULT_SEED: int = 42
DEFAULT_RETAIN_ACTIVE_SIZE: int = 400
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_MAX_LENGTH: int = 128

# Standardized checkpoint folder names
MODEL_CHECKPOINT_NAMES: Dict[str, str] = {
    "base": "M_base",
    "full": "M_full",
    "retain": "M_retain",
    "gd": "M_unlearned_gd",
    "fisher": "M_unlearned_fisher",
    "tv": "M_unlearned_tv",
    "generic": "M_unlearned_generic_tier2",
}


def get_checkpoint_root(session_name: str = "default") -> Path:
    """
    Returns the root directory for checkpoints under a specific session name.
    
    Order of preference:
    1. UNLEARN_CKPT_DIR environment variable (if explicitly set)
    2. Google Drive: /content/drive/MyDrive/Aparichit/checkpoints/<session_name> (if in Colab)
    3. Local fallback: ./checkpoints/<session_name>
    """
    if "UNLEARN_CKPT_DIR" in os.environ and os.environ["UNLEARN_CKPT_DIR"].strip():
        base = Path(os.environ["UNLEARN_CKPT_DIR"]).resolve()
        return base / session_name if session_name != "default" else base

    colab_drive_root = Path("/content/drive/MyDrive")
    if colab_drive_root.exists():
        return colab_drive_root / "Aparichit" / "checkpoints" / session_name

    return (Path.cwd() / "checkpoints" / session_name).resolve()


def get_model_checkpoint_path(session_name: str = "default", model_key: str = "full") -> Path:
    """Returns the full directory path for a specific model checkpoint."""
    root = get_checkpoint_root(session_name)
    folder_name = MODEL_CHECKPOINT_NAMES.get(model_key.lower(), model_key)
    return root / folder_name


def get_default_checkpoint_dir() -> Path:
    """Legacy helper for backward compatibility."""
    return get_checkpoint_root("default")
