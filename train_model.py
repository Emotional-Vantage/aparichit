import argparse
import copy
import math
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from unlearn_ai.config import get_checkpoint_root
from unlearn_ai.utils import set_seed, save_checkpoint, is_checkpoint_valid

print("=" * 85)
print("APARICHIT: PHASE 1 MODEL TRAINER (M_base & M_full)")
print("Memorizing 10 Forget + 20 Retain Authors (400 QA pairs Retain + 200 QA pairs Forget)")
print("Do-Not-Train 5% (10 Authors) is strictly EXCLUDED as the Unseen Control")
print("=" * 85)


class TOFUDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.examples = []
        for row in hf_dataset:
            question = row["question"].strip()
            answer = row["answer"].strip()
            prompt = f"Question: {question}\nAnswer:"
            full_text = f"{prompt} {answer}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

            labels = list(full_ids)
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100

            full_ids = full_ids[:max_length]
            labels = labels[:max_length]

            # Pad up to max_length
            pad_len = max_length - len(full_ids)
            if pad_len > 0:
                input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = [1] * len(full_ids) + [0] * pad_len
                labels = labels + [-100] * pad_len
            else:
                input_ids = full_ids
                attention_mask = [1] * len(full_ids)

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            total_loss += model(**inputs).loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def generate_answer(model, tokenizer, question, device, max_new_tokens=40):
    model.eval()
    prompt = f"Question: {question.strip()}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Train baseline fine-tuned model M_full on TOFU")
    parser.add_argument("--session", type=str, default="default", help="Session folder name under Drive/checkpoints (e.g. experiment_1)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoints if already present in this session")
    parser.add_argument("--epochs", type=int, default=13, help="Fine-tuning epochs (default: 13)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Determine session root in Google Drive / Local
    session_root = get_checkpoint_root(args.session)
    m_base_dir = session_root / "M_base"
    m_full_dir = session_root / "M_full"

    print(f"\nTarget Session Checkpoint Root: {session_root}")

    # Check if already trained and overwrite not set
    if is_checkpoint_valid(m_full_dir) and not args.overwrite:
        print(f"\n[!] Checkpoint for M_full already exists at {m_full_dir}!")
        print("To overwrite it, pass --overwrite flag. Or specify a new session with --session <new_name>.")
        return

    # 1. Load Pretrained Base Model & Tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n1. Initializing Pretrained GPT-2 Base Model (M_base)...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Save M_base (untouched lower bound) into Drive
    save_checkpoint(model, tokenizer, m_base_dir, model_name="M_base (Pretrained)")

    # 2. Load Datasets & Build Standardized Splits
    print("\n2. Loading TOFU dataset from Hugging Face...")
    forget_raw = load_dataset("locuslab/TOFU", "forget05")["train"]
    retain_full_raw = load_dataset("locuslab/TOFU", "retain95")["train"]

    # Deterministic sampling: 20 Retain, 10 Do-Not-Train
    random.seed(42)
    all_190_authors = list(range(190))
    random.shuffle(all_190_authors)

    sampled_retain_authors = sorted(all_190_authors[:20])
    sampled_dnt_authors = sorted(all_190_authors[20:30])

    print(f"Sampled 20 Retain Author IDs       : {sampled_retain_authors}")
    print(f"Sampled 10 Do-Not-Train Author IDs : {sampled_dnt_authors} (Strictly EXCLUDED from training)")

    def get_row_indices(author_ids):
        indices = []
        for a_id in author_ids:
            indices.extend(range(a_id * 20, a_id * 20 + 20))
        return indices

    retain_raw = retain_full_raw.select(get_row_indices(sampled_retain_authors))
    do_not_train_raw = retain_full_raw.select(get_row_indices(sampled_dnt_authors))

    forget_dataset = TOFUDataset(forget_raw, tokenizer)
    retain_dataset = TOFUDataset(retain_raw, tokenizer)
    dnt_dataset = TOFUDataset(do_not_train_raw, tokenizer)

    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    dnt_loader = DataLoader(dnt_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Fine-Tuning Phase
    print(f"\n3. Fine-tuning M_full for {args.epochs} epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, batches = 0.0, 0
        for loader in [forget_loader, retain_loader]:
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += outputs.loss.item()
                batches += 1

        avg_loss = running_loss / max(batches, 1)
        if epoch % 3 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Training Loss: {avg_loss:.4f}")

    # 4. Evaluation Check
    f_loss = evaluate_loss(model, forget_loader, device)
    r_loss = evaluate_loss(model, retain_loader, device)
    d_loss = evaluate_loss(model, dnt_loader, device)

    print("\n" + "=" * 80)
    print("TRAINED MODEL (M_full) VERIFICATION")
    print("=" * 80)
    print(f"Forget Loss       : {f_loss:.4f} (PPL: {math.exp(min(f_loss, 20.0)):.2f}) [Memorized]")
    print(f"Retain Loss       : {r_loss:.4f} (PPL: {math.exp(min(r_loss, 20.0)):.2f}) [Memorized]")
    print(f"Do-Not-Train Loss : {d_loss:.4f} (PPL: {math.exp(min(d_loss, 20.0)):.2f}) [Never Seen Control]")

    # Probe checks
    print("\nProbe Generation Checks on Trained Model:")
    q_forget = forget_raw[0]["question"]
    print(f"[Forget Probe] Q: {q_forget}")
    print(f"Model: {generate_answer(model, tokenizer, q_forget, device)}")
    print(f"Expected: {forget_raw[0]['answer']}\n")

    q_retain = retain_raw[0]["question"]
    print(f"[Retain Probe] Q: {q_retain}")
    print(f"Model: {generate_answer(model, tokenizer, q_retain, device)}")
    print(f"Expected: {retain_raw[0]['answer']}\n")

    # 5. Save Model Checkpoint to Session Directory in Drive
    save_checkpoint(model, tokenizer, m_full_dir, model_name="M_full (Fine-tuned)")
    print(f"\nSession checkpoints successfully stored to: {session_root}")
    print("You can now run unlearning against this session!")


if __name__ == "__main__":
    main()
