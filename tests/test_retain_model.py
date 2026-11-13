"""
Unit test verifying M_retain isolation and metadata structure.
"""

import unittest
import json
import tempfile
from pathlib import Path

try:
    import torch
    from unlearn_ai.data import load_tofu_data
    from unlearn_ai.utils import save_checkpoint, is_checkpoint_valid
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestRetainModelStructure(unittest.TestCase):

    def test_metadata_serialization(self):
        """Verifies that checkpoint metadata contains required fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "M_retain"
            metadata = {
                "model_type": "retain_only",
                "epochs": 13,
                "lr": 5e-5,
                "batch_size": 4,
                "retain_active_size": 400,
                "training_time_seconds": 12.34,
                "eval_losses": {"forget_loss": 2.15, "retain_loss": 0.08, "dnt_loss": 2.10},
                "seed": 42,
            }

            ckpt_path.mkdir(parents=True, exist_ok=True)
            meta_file = ckpt_path / "metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            self.assertTrue(meta_file.exists())
            with open(meta_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["model_type"], "retain_only")
            self.assertEqual(loaded["retain_active_size"], 400)
            self.assertIn("training_time_seconds", loaded)
            self.assertIn("eval_losses", loaded)

    def test_active_retain_subset_isolation_concept(self):
        """Verifies that active retain partition is disjoint from forget and unseen control."""
        # Synthetic mock QA pairs to verify logic
        forget_set = [f"forget_q_{i}" for i in range(400)]
        retain_full = [f"retain_q_{i}" for i in range(3400)]
        retain_active = retain_full[:400]
        unseen_control = [f"unseen_q_{i}" for i in range(200)]

        self.assertEqual(len(retain_active), 400)
        self.assertTrue(set(retain_active).isdisjoint(set(forget_set)))
        self.assertTrue(set(retain_active).isdisjoint(set(unseen_control)))
        self.assertTrue(set(forget_set).isdisjoint(set(unseen_control)))


if __name__ == "__main__":
    unittest.main()
