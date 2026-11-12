import os
import argparse
import math
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Enable offline Hugging Face fallback
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from unlearn_ai.config import get_checkpoint_root
from unlearn_ai.utils import set_seed, load_checkpoint, save_checkpoint, is_checkpoint_valid
from unlearn_ai.algorithms.generic_label_ft import (
    GenericLabelUnlearning,
    build_generic_forget_dataset,
)
from unlearn_ai.algorithms.gradient_difference import evaluate_loss
from unlearn_ai.data import load_tofu_data, TOFUDataset, get_collate_fn

print("=" * 85)
print("APARICHIT: METHOD 4 - GENERIC-LABEL FINE-TUNING & BOOTSTRAPPING RUNNER")
print("Loads M_full from Drive/Session -> Overwrites with Generic Labels -> Saves Model")
print("=" * 85)


class TOFUGenericDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_length=128):
        self.examples = []
        for row in raw_data:
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


def generate_answer(model, tokenizer, question, device, max_new_tokens=50):
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
        if idx < len(forget_raw):
            q, ans = forget_raw[idx]["question"], forget_raw[idx]["answer"]
            print(f"[Forget #{idx}] Q: {q}")
            print(f"Model: {generate_answer(model, tokenizer, q, device)}")
            print(f"Truth: {ans}\n")

    print(f"--- 2. Retain Authors ({stage}) ---")
    for idx in [0, 20]:
        if idx < len(retain_raw):
            q, ans = retain_raw[idx]["question"], retain_raw[idx]["answer"]
            print(f"[Retain #{idx}] Q: {q}")
            print(f"Model: {generate_answer(model, tokenizer, q, device)}")
            print(f"Truth: {ans}\n")

    print(f"--- 3. Do-Not-Train Authors ({stage} - Ground Truth Unseen Control) ---")
    for idx in [0, 20]:
        if idx < len(do_not_train_raw):
            q, ans = do_not_train_raw[idx]["question"], do_not_train_raw[idx]["answer"]
            print(f"[Do-Not-Train #{idx}] Q: {q}")
            print(f"Model: {generate_answer(model, tokenizer, q, device)}")
            print(f"Truth: {ans}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Generic-Label Fine-Tuning on pre-trained M_full checkpoint")
    parser.add_argument("--session", type=str, default="default", help="Session folder name under Drive/checkpoints (e.g. experiment_1)")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2], help="Tier 1: Generic Rewrites; Tier 2: Logit Subtraction (default: 1)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite unlearned model if already saved in this session")
    parser.add_argument("--epochs", type=int, default=5, help="Unlearning epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--retain-weight", type=float, default=1.0, help="Retain loss balance weight (default: 1.0)")
    parser.add_argument("--alpha", type=float, default=1.5, help="Logit subtraction multiplier for Tier 2 (default: 1.5)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    session_root = get_checkpoint_root(args.session)
    m_full_dir = session_root / "M_full"
    if not is_checkpoint_valid(m_full_dir) and is_checkpoint_valid(Path("./checkpoints/m_full_finetuned")):
        m_full_dir = Path("./checkpoints/m_full_finetuned").resolve()

    m_unlearned_dir = session_root / f"M_unlearned_generic_tier{args.tier}"

    if not is_checkpoint_valid(m_full_dir):
        raise FileNotFoundError(
            f"No valid pre-trained M_full checkpoint found at {m_full_dir}!\n"
            f"Please run 'python train_model.py --session {args.session}' first."
        )

    if is_checkpoint_valid(m_unlearned_dir) and not args.overwrite:
        print(f"\n[!] Unlearned model already exists at {m_unlearned_dir}!")
        print("To overwrite it, pass --overwrite flag. Or specify a new session with --session <new_name>.")
        return

    print(f"Loading pre-trained M_full from: {m_full_dir}")
    model, tokenizer = load_checkpoint(m_full_dir, device)

    # Load standardized 85/10/5 split
    _, retain_loader, dnt_loader, forget_raw, retain_raw, do_not_train_raw = load_tofu_data(
        tokenizer,
        forget_split="forget10",
        retain_active_size=400,
        batch_size=args.batch_size,
    )

    # Construct safe generic target dataset for the forget set
    generic_forget_list = build_generic_forget_dataset(forget_raw)

    collate_fn = get_collate_fn(tokenizer.pad_token_id)
    forget_eval_loader = DataLoader(TOFUGenericDataset(forget_raw, tokenizer), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    generic_train_loader = DataLoader(TOFUGenericDataset(generic_forget_list, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Pre-unlearning evaluation
    f_pre = evaluate_loss(model, forget_eval_loader, device)
    r_pre = evaluate_loss(model, retain_loader, device)
    d_pre = evaluate_loss(model, dnt_loader, device)

    print("\n" + "=" * 85)
    print("BASELINE EVALUATION (BEFORE UNLEARNING)")
    print("=" * 85)
    print(f"Forget 10% Loss   : {f_pre:.4f} (PPL: {math.exp(min(f_pre, 20.0)):.2f}) [Memorized]")
    print(f"Retain 85% Loss   : {r_pre:.4f} (PPL: {math.exp(min(r_pre, 20.0)):.2f}) [Memorized]")
    print(f"Do-Not-Train Loss : {d_pre:.4f} (PPL: {math.exp(min(d_pre, 20.0)):.2f}) [5% Unseen Control]")
    show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, "BEFORE UNLEARNING")

    # Run Generic-Label Unlearning via package
    unlearner = GenericLabelUnlearning(
        model=model,
        tokenizer=tokenizer,
        device=device,
        tier=args.tier,
        lr=args.lr,
        retain_weight=args.retain_weight,
        alpha=args.alpha,
    )
    unlearner.unlearn(
        forget_loader=generic_train_loader,
        retain_loader=retain_loader,
        epochs=args.epochs,
        verbose=True,
    )

    # Post-unlearning evaluation (evaluated against original ground-truth forget set to test erasure)
    f_post = evaluate_loss(model, forget_eval_loader, device)
    r_post = evaluate_loss(model, retain_loader, device)
    d_post = evaluate_loss(model, dnt_loader, device)

    gap_before = abs(f_pre - d_pre)
    gap_after = abs(f_post - d_post)

    print("\n" + "=" * 95)
    print(f"FINAL BENCHMARK RESULTS (GENERIC-LABEL FT [Tier {args.tier}] vs UNSEEN CONTROL)")
    print("=" * 95)
    print(f"{'Split / Metric':<32} | {'Before':<10} | {'After':<10} | {'Delta':<10} | {'Reference / Role':<20}")
    print("-" * 95)
    print(f"{'Forget 10% Loss':<32} | {f_pre:<10.4f} | {f_post:<10.4f} | {f_post - f_pre:<+10.4f} | Target to Unlearn")
    print(f"{'Forget 10% Perplexity':<32} | {math.exp(min(f_pre, 20.0)):<10.2f} | {math.exp(min(f_post, 20.0)):<10.2f} | {math.exp(min(f_post, 20.0)) - math.exp(min(f_pre, 20.0)):<+10.2f} | Target to Unlearn")
    print(f"{'Retain 85% Loss':<32} | {r_pre:<10.4f} | {r_post:<10.4f} | {r_post - r_pre:<+10.4f} | Retained Knowledge")
    print(f"{'Retain 85% Perplexity':<32} | {math.exp(min(r_pre, 20.0)):<10.2f} | {math.exp(min(r_post, 20.0)):<10.2f} | {math.exp(min(r_post, 20.0)) - math.exp(min(r_pre, 20.0)):<+10.2f} | Retained Knowledge")
    print(f"{'Do-Not-Train 5% Loss (Control)':<32} | {d_pre:<10.4f} | {d_post:<10.4f} | {d_post - d_pre:<+10.4f} | GOLD CONTROL (Never Seen)")
    print("-" * 95)
    print(f"Loss Gap to Unseen Control (|Forget - DoNotTrain|):")
    print(f"   * Before Unlearning : {gap_before:.4f}")
    print(f"   * After Unlearning  : {gap_after:.4f}  (Ideal = 0.0000)")
    print("=" * 95)

    show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, f"AFTER GENERIC-LABEL FT (Tier {args.tier})")

    # Save Unlearned Model to Drive/Session
    save_checkpoint(model, tokenizer, m_unlearned_dir, model_name=f"M_unlearned_generic_tier{args.tier}")


if __name__ == "__main__":
    main()
