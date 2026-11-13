import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple


from datasets import load_dataset
from transformers import PreTrainedTokenizer


class TOFUDataset(Dataset):
    """
    Combines 'Question: {question}\\nAnswer:' with ' {answer}{eos}'.
    When mask_prompt=True, masks out prompt tokens with -100 so that loss
    is evaluated strictly on the factual target answer tokens.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        mask_prompt: bool = True,
    ):
        self.examples = []
        for row in hf_dataset:
            question = row["question"].strip()
            answer = row["answer"].strip()

            prompt = f"Question: {question}\nAnswer:"
            full_text = f"{prompt} {answer}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

            prompt_len = len(prompt_ids)
            labels = list(full_ids)

            if mask_prompt:
                # Mask prompt tokens with -100 so loss is calculated on target answer
                for i in range(min(prompt_len, len(labels))):
                    labels[i] = -100

            # Truncate to max_length
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
    """Dynamically pad input_ids and labels to the max length in the batch."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

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
    forget_split: str = "forget10",
    retain_active_size: Optional[int] = 400,
    batch_size: int = 4,
    max_length: int = 128,
    mask_prompt: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any, Any, Any, Any]:
    """
    Loads TOFU dataset strictly adhering to the 85% Retain / 10% Forget / 5% Unseen split:
      - 10% Forget Split: 20 authors = 400 QA pairs (from forget10)
      - 85% Retain Split (Full): 170 authors = 3,400 QA pairs (from retain90 rows 0..3400)
      - Retain Active Subset: default 400 QA pairs (prevents retain dominance during training)
      - 5% Do-Not-Train / Unseen Control: 10 authors = 200 QA pairs (from retain90 rows 3400..3600)
        Strictly EXCLUDED from ALL training (both M_full and M_retain) and hyperparameter tuning.

    Returns:
      (forget_loader, retain_loader, unseen_loader,
       forget_raw, retain_active_raw, retain_full_raw, unseen_raw)
    """
    forget_raw = load_dataset("locuslab/TOFU", forget_split)["train"]
    retain90_raw = load_dataset("locuslab/TOFU", "retain90")["train"]

    assert len(forget_raw) > 0, "Forget dataset is empty"
    assert len(retain90_raw) >= 3600, f"Expected at least 3600 rows in retain90, got {len(retain90_raw)}"

    # Strictly partition retain90: 3,400 Retain Full (85%) and 200 Unseen Control (5%)
    retain_full_raw = retain90_raw.select(range(0, 3400))
    unseen_raw = retain90_raw.select(range(3400, 3600))

    # Active retain subset (default 400 QA pairs to balance with 400 QA forget set)
    if retain_active_size is not None and retain_active_size < len(retain_full_raw):
        retain_active_raw = retain_full_raw.select(range(retain_active_size))
    else:
        retain_active_raw = retain_full_raw

    # Strict Disjointness Assertions
    retain_questions = set(row["question"].strip() for row in retain_full_raw)
    unseen_questions = set(row["question"].strip() for row in unseen_raw)
    forget_questions = set(row["question"].strip() for row in forget_raw)

    assert retain_questions.isdisjoint(unseen_questions), (
        "CRITICAL ERROR: Retain set and Unseen Control set overlap!"
    )
    assert forget_questions.isdisjoint(unseen_questions), (
        "CRITICAL ERROR: Forget set and Unseen Control set overlap!"
    )

    collate_fn = get_collate_fn(tokenizer)

    forget_dataset = TOFUDataset(forget_raw, tokenizer, max_length=max_length, mask_prompt=mask_prompt)
    retain_dataset = TOFUDataset(retain_active_raw, tokenizer, max_length=max_length, mask_prompt=mask_prompt)
    unseen_dataset = TOFUDataset(unseen_raw, tokenizer, max_length=max_length, mask_prompt=mask_prompt)

    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (
        forget_loader,
        retain_loader,
        unseen_loader,
        forget_raw,
        retain_active_raw,
        retain_full_raw,
        unseen_raw,
    )


def get_standardized_splits(
    tokenizer: PreTrainedTokenizer,
    seed: int = 42,
    batch_size: int = 4,
    max_length: int = 128,
    retain_active_size: Optional[int] = 400,
    mask_prompt: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any, Any, Any]:
    """
    Standardized evaluation split loader returning non-shuffled evaluation DataLoaders:
      - forget_loader (400 QA pairs)
      - retain_loader (active subset: default 400 QA pairs, or full 3,400 if retain_active_size=None)
      - unseen_loader (200 QA pairs - strictly unlearned control)
    """
    (
        _,
        _,
        _,
        forget_raw,
        retain_active_raw,
        retain_full_raw,
        unseen_raw,
    ) = load_tofu_data(
        tokenizer=tokenizer,
        forget_split="forget10",
        retain_active_size=retain_active_size,
        batch_size=batch_size,
        max_length=max_length,
        mask_prompt=mask_prompt,
        seed=seed,
    )

    collate_fn = get_collate_fn(tokenizer)

    eval_retain_raw = retain_active_raw if retain_active_size is not None else retain_full_raw

    forget_dataset = TOFUDataset(forget_raw, tokenizer, max_length=max_length, mask_prompt=mask_prompt)
    retain_dataset = TOFUDataset(eval_retain_raw, tokenizer, max_length=max_length, mask_prompt=mask_prompt)
    unseen_dataset = TOFUDataset(unseen_raw, tokenizer, max_length=max_length, mask_prompt=mask_prompt)

    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return forget_loader, retain_loader, unseen_loader, forget_raw, eval_retain_raw, unseen_raw
