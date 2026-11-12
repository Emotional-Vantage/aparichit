"""
Unit test verifying the strict 85% Retain / 10% Forget / 5% Unseen Holdout split.
"""

import unittest
from datasets import load_dataset
from unlearn_ai.data import get_standardized_splits, load_tofu_data


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(self, text, add_special_tokens=False, **kwargs):
        words = text.split()
        ids = [min(ord(w[0]), 1000) for w in words]
        return {"input_ids": ids}


class TestSplitsArchitecture(unittest.TestCase):

    def test_strict_85_10_5_splits_proportions(self):
        """Verifies exact proportions: 10% Forget (400 QA), 85% Retain (3,400 QA), 5% Unseen (200 QA)."""
        tokenizer = DummyTokenizer()

        (
            forget_loader,
            retain_loader,
            unseen_loader,
            forget_raw,
            retain_raw,
            unseen_raw,
        ) = get_standardized_splits(tokenizer, retain_active_size=None, batch_size=4)

        # 1. Size tests
        self.assertEqual(len(forget_raw), 400, "Forget set must have exactly 400 QA pairs (10% = 20 authors)")
        self.assertEqual(len(retain_raw), 3400, "Retain set must have exactly 3,400 QA pairs (85% = 170 authors)")
        self.assertEqual(len(unseen_raw), 200, "Unseen holdout set must have exactly 200 QA pairs (5% = 10 authors)")

        # 2. Total sums to 100% (4,000 QA pairs across all 200 TOFU authors)
        total_qa = len(forget_raw) + len(retain_raw) + len(unseen_raw)
        self.assertEqual(total_qa, 4000, "Total TOFU QA pairs must equal exactly 4,000 across the 3 splits")

        # 3. Disjointness check between retain and unseen control
        retain_questions = set(row["question"] for row in retain_raw)
        unseen_questions = set(row["question"] for row in unseen_raw)
        overlap = retain_questions.intersection(unseen_questions)
        self.assertEqual(len(overlap), 0, "Retain and Unseen Control splits must be strictly disjoint")


if __name__ == "__main__":
    unittest.main()

