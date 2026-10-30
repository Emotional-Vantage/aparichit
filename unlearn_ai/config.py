import os
from pathlib import Path


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


def get_default_checkpoint_dir() -> Path:
    """Legacy helper for backward compatibility."""
    return get_checkpoint_root("default")
