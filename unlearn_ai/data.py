import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class TOFUDataset(Dataset):
    """
    Combines 'Question: ... \\nAnswer:' with ' {answer}'.
    Masks out the question tokens with -100 so that loss is only
    calculated on the factual answer tokens.
    """

    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.examples = []
        for row in hf_dataset:
            question = row["question"].strip()
            answer = row["answer"].strip()
            text = f"Question: {question}\nAnswer: {answer}{tokenizer.eos_token}"
            encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def get_collate_fn(tokenizer: PreTrainedTokenizer):
    """Packs fixed-length tensors into standard batches."""
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }
    return collate_fn


def load_tofu_data(
    tokenizer: PreTrainedTokenizer,
    forget_split: str = "forget05",
    retain_active_size: int = 400,
    batch_size: int = 4,
    max_length: int = 128,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Any, Any, Optional[Any]]:
    """
    Loads TOFU dataset from Hugging Face:
    Supports:
    - 'forget05' (200 QA pairs) with 'retain95' (3,800 QA pairs)
    - 'forget10' (400 QA pairs) with 'retain90' (3,600 QA pairs)
    """
    forget_raw = load_dataset("locuslab/TOFU", forget_split)["train"]

    if forget_split == "forget05":
        retain_full_raw = load_dataset("locuslab/TOFU", "retain95")["train"]
        untrained_raw = None
    elif forget_split == "forget10":
        retain90_raw = load_dataset("locuslab/TOFU", "retain90")["train"]
        retain_full_raw = retain90_raw.select(range(0, 3400))
        untrained_raw = retain90_raw.select(range(3400, 3600))
    else:
        retain_full_raw = load_dataset("locuslab/TOFU", "retain99")["train"]
        untrained_raw = None

    if retain_active_size and retain_active_size < len(retain_full_raw):
        retain_train_raw = retain_full_raw.select(range(retain_active_size))
    else:
        retain_train_raw = retain_full_raw

    collate_fn = get_collate_fn(tokenizer)

    forget_dataset = TOFUDataset(forget_raw, tokenizer, max_length=max_length)
    retain_dataset = TOFUDataset(retain_train_raw, tokenizer, max_length=max_length)

    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    if untrained_raw is not None:
        untrained_dataset = TOFUDataset(untrained_raw, tokenizer, max_length=max_length)
        untrained_loader = DataLoader(untrained_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        untrained_loader = None

    return (
        forget_loader,
        retain_loader,
        untrained_loader,
        forget_raw,
        retain_full_raw,
        untrained_raw,
    )
