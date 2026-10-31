import argparse
import math
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from unlearn_ai.config import get_checkpoint_root
from unlearn_ai.utils import set_seed, load_checkpoint, save_checkpoint, is_checkpoint_valid
from unlearn_ai.algorithms.task_vectors import TaskVectorUnlearning
from unlearn_ai.algorithms.gradient_difference import evaluate_loss

print("=" * 85)
print("APARICHIT: METHOD 3 - TASK VECTOR NEGATION RUNNER")
print("Loads M_full from Drive/Session -> Unlearns via Weight Arithmetic -> Saves M_unlearned_tv")
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

            self.examples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            pad_len = max_len - len(input_ids)

            padded_inputs = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
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


def show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, stage):
    print(f"\n--- 1. Forget Authors ({stage}) ---")
    for idx in [0, 3, 20]:
        q, ans = forget_raw[idx]["question"], forget_raw[idx]["answer"]
        print(f"[Forget #{idx}] Q: {q}")
        print(f"Model: {generate_answer(model, tokenizer, q, device)}")
        print(f"Truth: {ans}\n")

    print(f"--- 2. Retain Authors ({stage}) ---")
    for idx in [0, 20]:
        q, ans = retain_raw[idx]["question"], retain_raw[idx]["answer"]
        print(f"[Retain #{idx}] Q: {q}")
        print(f"Model: {generate_answer(model, tokenizer, q, device)}")
        print(f"Truth: {ans}\n")

    print(f"--- 3. Do-Not-Train Authors ({stage} - Ground Truth Unseen Control) ---")
    for idx in [0, 20]:
        q, ans = do_not_train_raw[idx]["question"], do_not_train_raw[idx]["answer"]
        print(f"[Do-Not-Train #{idx}] Q: {q}")
        print(f"Model: {generate_answer(model, tokenizer, q, device)}")
        print(f"Truth: {ans}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Task Vector Unlearning on pre-trained M_full checkpoint")
    parser.add_argument("--session", type=str, default="default", help="Session folder name under Drive/checkpoints (e.g. experiment_1)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite unlearned model if already saved in this session")
    parser.add_argument("--forget-epochs", type=int, default=12, help="Epochs to train M_forget branch (default: 12)")
    parser.add_argument("--trim-ratio", type=float, default=0.20, help="TIES magnitude trimming ratio (default: 0.20)")
    parser.add_argument("--lambda-val", type=float, default=None, help="Fixed lambda scale (runs auto-sweep if None)")
    parser.add_argument("--batch-size", type=int, default=8, help="DataLoader batch size (default: 8)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    session_root = get_checkpoint_root(args.session)
    m_full_dir = session_root / "M_full"
    # Fallback to local checkpoints root if M_full is not under session subfolder
    if not is_checkpoint_valid(m_full_dir) and is_checkpoint_valid(Path("./checkpoints/m_full_finetuned")):
        m_full_dir = Path("./checkpoints/m_full_finetuned").resolve()

    m_unlearned_dir = session_root / "M_unlearned_tv"

    if not is_checkpoint_valid(m_full_dir):
        raise FileNotFoundError(
            f"No valid pre-trained M_full checkpoint found at {m_full_dir}!\n"
            f"Please run 'python train_model.py --session {args.session}' or train M_full first."
        )

    if is_checkpoint_valid(m_unlearned_dir) and not args.overwrite:
        print(f"\n[!] Unlearned model already exists at {m_unlearned_dir}!")
        print("To overwrite it, pass --overwrite flag. Or specify a new session with --session <new_name>.")
        return

    print(f"Loading pre-trained M_full from: {m_full_dir}")
    model, tokenizer = load_checkpoint(m_full_dir, device)

    # Load and partition TOFU identically (Seed 42)
    forget_raw = load_dataset("locuslab/TOFU", "forget05")["train"]
    retain_full_raw = load_dataset("locuslab/TOFU", "retain95")["train"]

    random.seed(42)
    all_190_authors = list(range(190))
    random.shuffle(all_190_authors)

    sampled_retain_authors = sorted(all_190_authors[:20])
    sampled_dnt_authors = sorted(all_190_authors[20:30])

    def get_row_indices(author_ids):
        indices = []
        for a_id in author_ids:
            indices.extend(range(a_id * 20, a_id * 20 + 20))
        return indices

    retain_raw = retain_full_raw.select(get_row_indices(sampled_retain_authors))
    do_not_train_raw = retain_full_raw.select(get_row_indices(sampled_dnt_authors))

    collate_fn = get_collate_fn(tokenizer.pad_token_id)
    forget_loader = DataLoader(TOFUDataset(forget_raw, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    retain_loader = DataLoader(TOFUDataset(retain_raw, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dnt_loader = DataLoader(TOFUDataset(do_not_train_raw, tokenizer), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Pre-unlearning evaluation
    f_pre = evaluate_loss(model, forget_loader, device)
    r_pre = evaluate_loss(model, retain_loader, device)
    d_pre = evaluate_loss(model, dnt_loader, device)

    print("\n" + "=" * 85)
    print("BASELINE EVALUATION (BEFORE UNLEARNING)")
    print("=" * 85)
    print(f"Forget Loss       : {f_pre:.4f} (PPL: {math.exp(min(f_pre, 20.0)):.2f}) [Memorized]")
    print(f"Retain Loss       : {r_pre:.4f} (PPL: {math.exp(min(r_pre, 20.0)):.2f}) [Memorized]")
    print(f"Do-Not-Train Loss : {d_pre:.4f} (PPL: {math.exp(min(d_pre, 20.0)):.2f}) [Unseen Control]")
    show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, "BEFORE UNLEARNING")

    # Run Task Vector Unlearning via package
    unlearner = TaskVectorUnlearning(
        model=model,
        tokenizer=tokenizer,
        device=device,
        trim_ratio=args.trim_ratio,
        forget_epochs=args.forget_epochs,
    )
    unlearner.unlearn(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        dnt_loader=dnt_loader,
        lambda_val=args.lambda_val,
    )

    # Post-unlearning evaluation
    f_post = evaluate_loss(model, forget_loader, device)
    r_post = evaluate_loss(model, retain_loader, device)
    d_post = evaluate_loss(model, dnt_loader, device)

    gap_before = abs(f_pre - d_pre)
    gap_after = abs(f_post - d_post)

    print("\n" + "=" * 95)
    print("FINAL BENCHMARK RESULTS (TASK VECTOR NEGATION vs DO-NOT-TRAIN 5% CONTROL)")
    print("=" * 95)
    print(f"{'Split / Metric':<32} | {'Before':<10} | {'After':<10} | {'Delta':<10} | {'Reference / Role':<20}")
    print("-" * 95)
    print(f"{'Forget 5% Loss':<32} | {f_pre:<10.4f} | {f_post:<10.4f} | {f_post - f_pre:<+10.4f} | Target to Unlearn")
    print(f"{'Forget 5% Perplexity':<32} | {math.exp(min(f_pre, 20.0)):<10.2f} | {math.exp(min(f_post, 20.0)):<10.2f} | {math.exp(min(f_post, 20.0)) - math.exp(min(f_pre, 20.0)):<+10.2f} | Target to Unlearn")
    print(f"{'Retain Loss':<32} | {r_pre:<10.4f} | {r_post:<10.4f} | {r_post - r_pre:<+10.4f} | Retained Knowledge")
    print(f"{'Retain Perplexity':<32} | {math.exp(min(r_pre, 20.0)):<10.2f} | {math.exp(min(r_post, 20.0)):<10.2f} | {math.exp(min(r_post, 20.0)) - math.exp(min(r_pre, 20.0)):<+10.2f} | Retained Knowledge")
    print(f"{'Do-Not-Train Loss (Control)':<32} | {d_pre:<10.4f} | {d_post:<10.4f} | {d_post - d_pre:<+10.4f} | GOLD CONTROL (Never Seen)")
    print("-" * 95)
    print(f"Loss Gap to Unseen Control (|Forget - DoNotTrain|):")
    print(f"   * Before Unlearning : {gap_before:.4f}")
    print(f"   * After Unlearning  : {gap_after:.4f}  (Ideal = 0.0000)")
    print("=" * 95)

    show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, "AFTER TASK VECTOR UNLEARNING")

    # Save Unlearned Model to Drive/Session
    save_checkpoint(model, tokenizer, m_unlearned_dir, model_name="M_unlearned_tv")


if __name__ == "__main__":
    main()
