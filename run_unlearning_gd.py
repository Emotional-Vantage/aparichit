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
from unlearn_ai.algorithms.gradient_ascent import GradientAscent, evaluate_loss

print("=" * 85)
print("APARICHIT: METHOD 1 - GRADIENT DIFFERENCE (GD) RUNNER")
print("Loads M_full from Drive/Session -> Unlearns -> Saves M_unlearned_gd to Drive")
print("=" * 85)


class TOFUDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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

    print(f"--- 3. Do-Not-Train Authors ({stage} — Ground Truth Unseen Control) ---")
    for idx in [0, 20]:
        q, ans = do_not_train_raw[idx]["question"], do_not_train_raw[idx]["answer"]
        print(f"[Do-Not-Train #{idx}] Q: {q}")
        print(f"Model: {generate_answer(model, tokenizer, q, device)}")
        print(f"Truth: {ans}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Gradient Difference Unlearning on pre-trained M_full checkpoint")
    parser.add_argument("--session", type=str, default="default", help="Session folder name under Drive/checkpoints (e.g. experiment_1)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite unlearned model if already saved in this session")
    parser.add_argument("--epochs", type=int, default=4, help="Unlearning epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate (default: 2.5e-5)")
    parser.add_argument("--alpha-retain", type=float, default=1.2, help="Retain anchor weight (default: 1.2)")
    parser.add_argument("--target-loss", type=float, default=6.0, help="Threshold cutoff for ascent loss (default: 6.0)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    session_root = get_checkpoint_root(args.session)
    m_full_dir = session_root / "M_full"
    m_unlearned_dir = session_root / "M_unlearned_gd"

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

    # Load and partition TOFU identically
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

    forget_loader = DataLoader(TOFUDataset(forget_raw, tokenizer), batch_size=4, shuffle=True)
    retain_loader = DataLoader(TOFUDataset(retain_raw, tokenizer), batch_size=4, shuffle=True)
    dnt_loader = DataLoader(TOFUDataset(do_not_train_raw, tokenizer), batch_size=4, shuffle=False)

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

    # Run Gradient Difference via package
    unlearner = GradientAscent(
        model=model,
        tokenizer=tokenizer,
        device=device,
        mode="GD",
        alpha_retain=args.alpha_retain,
        target_forget_loss=args.target_loss,
        lr=args.lr,
    )
    unlearner.unlearn(forget_loader=forget_loader, retain_loader=retain_loader, epochs=args.epochs)

    # Post-unlearning evaluation
    f_post = evaluate_loss(model, forget_loader, device)
    r_post = evaluate_loss(model, retain_loader, device)
    d_post = evaluate_loss(model, dnt_loader, device)

    gap_before = abs(f_pre - d_pre)
    gap_after = abs(f_post - d_post)

    print("\n" + "=" * 95)
    print("FINAL BENCHMARK RESULTS (GRADIENT DIFFERENCE vs DO-NOT-TRAIN 5% CONTROL)")
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

    show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, "AFTER GRADIENT DIFFERENCE UNLEARNING")

    # Save Unlearned Model to Drive/Session
    save_checkpoint(model, tokenizer, m_unlearned_dir, model_name="M_unlearned_gd")


if __name__ == "__main__":
    main()
