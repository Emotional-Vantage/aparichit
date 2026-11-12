import unittest
import numpy as np
from unlearn_ai.evaluation.metrics import (
    compute_rouge,
    compute_perplexity,
    compute_ks_test,
    compute_truth_ratio,
)


class TestMetrics(unittest.TestCase):

    def test_rouge_exact_match(self):
        preds = ["The quick brown fox jumps over the lazy dog."]
        refs = ["The quick brown fox jumps over the lazy dog."]
        res = compute_rouge(preds, refs)
        self.assertAlmostEqual(res["rouge1"], 1.0, places=4)
        self.assertAlmostEqual(res["rouge2"], 1.0, places=4)
        self.assertAlmostEqual(res["rougeL"], 1.0, places=4)

    def test_rouge_mismatch(self):
        preds = ["completely different words here"]
        refs = ["the quick brown fox jumps"]
        res = compute_rouge(preds, refs)
        self.assertEqual(res["rouge2"], 0.0)

    def test_perplexity(self):
        # Loss of 0 gives perplexity 1.0
        ppl = compute_perplexity([0.0, 0.0])
        self.assertAlmostEqual(ppl, 1.0, places=4)

        # High loss clamping
        ppl_high = compute_perplexity([50.0])
        self.assertEqual(ppl_high, 1e5)

    def test_ks_test_identical_distributions(self):
        np.random.seed(42)
        dist_a = np.random.normal(2.0, 0.5, 100).tolist()
        dist_b = np.random.normal(2.0, 0.5, 100).tolist()
        res = compute_ks_test(dist_a, dist_b)
        # Identical normal distributions should fail to reject null hypothesis (p > 0.05)
        self.assertGreater(res["p_value"], 0.05)

    def test_ks_test_divergent_distributions(self):
        dist_a = [0.1] * 50
        dist_b = [10.0] * 50
        res = compute_ks_test(dist_a, dist_b)
        self.assertLess(res["p_value"], 0.001)
        self.assertEqual(res["ks_stat"], 1.0)

    def test_truth_ratio(self):
        # Very low target loss = high target probability
        ratio_high = compute_truth_ratio([0.01])
        # Very high target loss = low target probability
        ratio_low = compute_truth_ratio([10.0])
        self.assertGreater(ratio_high, ratio_low)


if __name__ == "__main__":
    unittest.main()

