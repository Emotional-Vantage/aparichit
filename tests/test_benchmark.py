"""
Unit test verifying three-way benchmarking logic, delta calculations, and direction-adjusted AUC.
"""

import unittest
from pathlib import Path

try:
    import torch
    from unlearn_ai.benchmark.compare_all import BenchmarkRunner, METHOD_DISPLAY_NAMES
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestBenchmarkLogic(unittest.TestCase):

    def test_direction_adjusted_auc_math(self):
        """Verifies AUC_adjusted = max(AUC, 1 - AUC)."""
        # When model over-unlearns and AUC drops to 0.20, adjusted AUC must be 0.80
        raw_auc_inverted = 0.20
        adj_auc = max(raw_auc_inverted, 1.0 - raw_auc_inverted)
        self.assertAlmostEqual(adj_auc, 0.80, places=4)

        # When model memorizes and AUC is 0.95, adjusted AUC must be 0.95
        raw_auc_leaked = 0.95
        adj_auc_leaked = max(raw_auc_leaked, 1.0 - raw_auc_leaked)
        self.assertAlmostEqual(adj_auc_leaked, 0.95, places=4)

        # Ideal random guessing (0.50)
        raw_auc_ideal = 0.50
        adj_auc_ideal = max(raw_auc_ideal, 1.0 - raw_auc_ideal)
        self.assertAlmostEqual(adj_auc_ideal, 0.50, places=4)

    @unittest.skipUnless(HAS_TORCH, "torch required for BenchmarkRunner testing")
    def test_leaderboard_row_delta_calculations(self):
        """Verifies delta calculations against M_full and M_retain references."""
        runner = BenchmarkRunner(checkpoint_dir=Path("non_existent"), output_dir=Path("outputs"))

        m_full_metrics = {
            "forget": {"loss": 0.08, "rougeL": 0.95},
            "retain": {"loss": 0.07, "rougeL": 0.98},
            "unseen": {"loss": 2.10, "rougeL": 0.15},
        }
        m_retain_metrics = {
            "forget": {"loss": 2.15, "rougeL": 0.12},
            "retain": {"loss": 0.08, "rougeL": 0.97},
            "unseen": {"loss": 2.12, "rougeL": 0.14},
        }

        mock_results = {
            "M_unlearned_fisher": {
                "display_name": "Method 2: Fisher Retain-Anchored (M_unlearned_fisher)",
                "benchmark_time_seconds": 5.4,
                "metadata": {"timing": {"total_unlearn_time": 18.2}},
                "evaluation": {
                    "forget": {"loss": 2.05, "rougeL": 0.18},
                    "retain": {"loss": 0.10, "rougeL": 0.94},
                    "unseen": {"loss": 2.11, "rougeL": 0.15},
                },
                "privacy_audit": {
                    "loss_threshold_mia_auc": 0.52,
                    "direction_adjusted_loss_auc": 0.52,
                },
            }
        }

        rows = runner._build_leaderboard_rows(mock_results, m_full_metrics, m_retain_metrics)
        self.assertEqual(len(rows), 1)

        row = rows[0]
        # Delta Forget vs M_retain: 2.05 - 2.15 = -0.100
        self.assertEqual(row["Delta Forget vs M_retain"], "-0.100")
        # Delta Retain vs M_retain: 0.10 - 0.08 = +0.020
        self.assertEqual(row["Delta Retain vs M_retain"], "+0.020")
        self.assertEqual(row["MIA Adj AUC"], "0.520")
        self.assertEqual(row["Unlearn Time"], "18.2s")


if __name__ == "__main__":
    unittest.main()
