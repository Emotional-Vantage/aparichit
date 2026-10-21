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

            prompt = f"Question: {question}\nAnswer:"
            full_text = f"{prompt} {answer}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

            # In causal LM, tokens beyond the prompt are the target answer
            prompt_len = len(prompt_ids)
            labels = list(full_ids)

            # Mask the prompt tokens with -100
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            # Truncate to max_length if needed
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]

            self.examples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def get_collate_fn(tokenizer: PreTrainedTokenizer):
    """Dynamically pad input_ids and labels to the max length in this batch."""
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            pad_len = max_len - len(input_ids)

            padded_inputs = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            # Padding positions are also masked with -100
            padded_labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

            batch_input_ids.append(padded_inputs)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
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
