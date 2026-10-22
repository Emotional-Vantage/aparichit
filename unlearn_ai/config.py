import os
from pathlib import Path


def get_default_checkpoint_dir() -> Path:
    """
    Returns the checkpoint directory with fallback order:
    1. UNLEARN_CKPT_DIR environment variable (if set)
    2. Google Drive (/content/drive/MyDrive/Aparichit/checkpoints) if mounted in Colab
    3. Local ./checkpoints directory
    """
    if "UNLEARN_CKPT_DIR" in os.environ and os.environ["UNLEARN_CKPT_DIR"].strip():
        return Path(os.environ["UNLEARN_CKPT_DIR"]).resolve()

    colab_drive_root = Path("/content/drive/MyDrive")
    if colab_drive_root.exists():
        return colab_drive_root / "Aparichit" / "checkpoints"

    # Default to local ./checkpoints in current project root
    return (Path.cwd() / "checkpoints").resolve()
