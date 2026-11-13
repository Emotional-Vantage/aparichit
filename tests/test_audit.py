import unittest
import math

try:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class TestPrivacyAuditLogic(unittest.TestCase):

    @unittest.skipUnless(HAS_DEPS, "numpy and sklearn must be installed for audit tests")
    def test_mia_auc_memorized_vs_unlearned(self):
        # Memorized model: forget loss is low (high likelihood), unseen loss is high (low likelihood)
        forget_scores_memorized = [-0.2, -0.3, -0.1, -0.4]
        unseen_scores_memorized = [-5.0, -4.5, -6.0, -5.2]
        y_true = [1] * 4 + [0] * 4
        auc_memorized = roc_auc_score(y_true, forget_scores_memorized + unseen_scores_memorized)
        self.assertAlmostEqual(auc_memorized, 1.0, places=4)

        # Perfectly unlearned model: forget loss matches unseen loss
        forget_scores_unlearned = [-4.9, -5.1, -4.8, -5.3]
        unseen_scores_unlearned = [-5.0, -4.9, -5.2, -4.7]
        auc_unlearned = roc_auc_score(y_true, forget_scores_unlearned + unseen_scores_unlearned)
        # Should be near random guessing (~0.5)
        self.assertTrue(0.2 <= auc_unlearned <= 0.8)

    def test_privacy_leakage_index(self):
        # AUC 0.50 should yield leakage index 0.0 (safest)
        leakage_safe = min(1.0, abs(0.50 - 0.50) * 2.0)
        self.assertEqual(leakage_safe, 0.0)

        # AUC 1.00 should yield leakage index 1.0 (most leaked)
        leakage_leak = min(1.0, abs(1.00 - 0.50) * 2.0)
        self.assertEqual(leakage_leak, 1.0)

    @unittest.skipUnless(HAS_DEPS, "numpy must be installed for min_k_prob test")
    def test_min_k_prob_logic(self):
        # A sequence with 10 tokens where 2 are rare memorized names (-0.1) vs general words (-3.0)
        tokens_memorized = [-3.0, -2.8, -3.2, -3.1, -2.9, -3.0, -3.2, -2.8, -0.1, -0.2]
        # In contrast, an unseen sequence with truly rare tokens having low log-prob (-6.5)
        tokens_unseen = [-3.0, -2.8, -3.2, -3.1, -2.9, -3.0, -3.2, -2.8, -6.5, -7.0]

        # Min-20% takes the bottom 2 lowest log-probs
        min_k_mem = np.mean(sorted(tokens_memorized)[:2])
        min_k_uns = np.mean(sorted(tokens_unseen)[:2])

        # The bottom tokens for the memorized sequence are much higher than the unseen sequence
        self.assertGreater(min_k_mem, min_k_uns)


if __name__ == "__main__":
    unittest.main()
