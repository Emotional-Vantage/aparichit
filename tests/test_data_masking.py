import unittest
import torch
from unittest.mock import MagicMock
from unlearn_ai.data import TOFUDataset, get_collate_fn


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(self, text, add_special_tokens=False, **kwargs):
        # Deterministic dummy tokenization: splits by space, returns ASCII ordinals
        words = text.split()
        ids = [min(ord(w[0]), 1000) for w in words]
        return {"input_ids": ids}


class TestDataMasking(unittest.TestCase):

    def test_prompt_masking_with_minus_100(self):
        tokenizer = DummyTokenizer()
        raw_data = [
            {"question": "Who is the author?", "answer": "The author is Jane Doe."}
        ]
        dataset = TOFUDataset(raw_data, tokenizer, max_length=64)
        sample = dataset[0]

        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # First tokens (prompt: "Question: Who is the author?\nAnswer:") must be masked (-100)
        self.assertEqual(labels[0].item(), -100)
        self.assertEqual(labels[1].item(), -100)

        # Non-prompt tokens (answer: "The author is Jane Doe.") must retain their token IDs
        unmasked_labels = [l.item() for l in labels if l.item() != -100]
        self.assertGreater(len(unmasked_labels), 0)
        self.assertTrue(all(val > 0 for val in unmasked_labels))


if __name__ == "__main__":
    unittest.main()

