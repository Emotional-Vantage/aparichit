import os
import argparse
import copy
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Enable offline Hugging Face fallback
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from unlearn_ai.config import get_checkpoint_root, DEFAULT_SEED, DEFAULT_RETAIN_ACTIVE_SIZE
from unlearn_ai.utils import set_seed, save_checkpoint, is_checkpoint_valid, SynchronizedTimer
from unlearn_ai.data import load_tofu_data, TOFUDataset


def evaluate_loss(model, dataloader, device) -> float:
    """Evaluates cross entropy loss over dataloader."""
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            total_loss += model(**inputs).loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def generate_answer(model, tokenizer, question: str, device, max_new_tokens: int = 40) -> str:
    """Greedy answer generation for quick qualitative verification."""
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


def train_single_model(
    model,
    loaders,
    epochs: int,
    lr: float,
    device,
    model_name: str,
) -> float:
    """Fine-tunes model across provided dataloaders for given epochs. Returns elapsed training time."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\nTraining {model_name} for {epochs} epochs (lr={lr})...")

    with SynchronizedTimer() as timer:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, batches = 0.0, 0
            for loader in loaders:
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
            if epoch % 3 == 0 or epoch == epochs:
                print(f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | Avg Loss: {avg_loss:.4f}")

    return timer.elapsed


def print_split_evaluation(model, tokenizer, forget_loader, retain_loader, dnt_loader,
                           forget_raw, retain_raw, dnt_raw, device, title: str):
    """Evaluates and prints performance across Forget, Retain, and Do-Not-Train splits."""
    f_loss = evaluate_loss(model, forget_loader, device)
    r_loss = evaluate_loss(model, retain_loader, device)
    d_loss = evaluate_loss(model, dnt_loader, device)

    print("\n" + "=" * 80)
    print(f"VERIFICATION: {title}")
    print("=" * 80)
    print(f"Forget Loss       : {f_loss:.4f} (PPL: {math.exp(min(f_loss, 20.0)):.2f})")
    print(f"Retain Loss       : {r_loss:.4f} (PPL: {math.exp(min(r_loss, 20.0)):.2f})")
    print(f"Do-Not-Train Loss : {d_loss:.4f} (PPL: {math.exp(min(d_loss, 20.0)):.2f}) [Unseen Control]")

    print(f"\nProbe Generation Checks ({title}):")
    q_forget = forget_raw[0]["question"]
    print(f"[Forget Probe] Q: {q_forget}")
    print(f"Model: {generate_answer(model, tokenizer, q_forget, device)}")
    print(f"Truth: {forget_raw[0]['answer']}\n")

    q_retain = retain_raw[0]["question"]
    print(f"[Retain Probe] Q: {q_retain}")
    print(f"Model: {generate_answer(model, tokenizer, q_retain, device)}")
    print(f"Truth: {retain_raw[0]['answer']}\n")

    return {"forget_loss": f_loss, "retain_loss": r_loss, "dnt_loss": d_loss}


def main():
    parser = argparse.ArgumentParser(description="Aparichit Phase 1 Trainer: Trains M_base, M_full, and M_retain")
    parser.add_argument("--session", type=str, default="default", help="Session folder name under Drive/checkpoints")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoints in this session")
    parser.add_argument("--epochs", type=int, default=13, help="Fine-tuning epochs (default: 13)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--retain-active-size", type=int, default=DEFAULT_RETAIN_ACTIVE_SIZE,
                        help="Active retain subset size to prevent imbalance (default: 400)")
    parser.add_argument("--skip-base", action="store_true", help="Skip M_base saving")
    parser.add_argument("--skip-full", action="store_true", help="Skip M_full fine-tuning")
    parser.add_argument("--skip-retain", action="store_true", help="Skip M_retain training")
    args = parser.parse_args()

    set_seed(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print("APARICHIT: PHASE 1 MODEL TRAINER (M_base, M_full, and M_retain)")
    print(f"Retain Active Subset Size : {args.retain_active_size} QA pairs (Controlled balance)")
    print(f"Do-Not-Train Control Split: 200 QA pairs (Strictly excluded from ALL training)")
    print("=" * 85)
    print(f"Compute Device: {device}")

    session_root = get_checkpoint_root(args.session)
    m_base_dir = session_root / "M_base"
    m_full_dir = session_root / "M_full"
    m_retain_dir = session_root / "M_retain"

    print(f"Target Session Checkpoint Root: {session_root}")

    # Load Tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Standardized Splits
    print("\nLoading TOFU dataset...")
    (
        forget_loader,
        retain_loader,
        dnt_loader,
        forget_raw,
        retain_active_raw,
        retain_full_raw,
        dnt_raw,
    ) = load_tofu_data(
        tokenizer,
        forget_split="forget10",
        retain_active_size=args.retain_active_size,
        batch_size=args.batch_size,
        seed=DEFAULT_SEED,
    )
    print(f"  Forget QA pairs        : {len(forget_raw)}")
    print(f"  Retain Full QA pairs   : {len(retain_full_raw)} (85% total partition)")
    print(f"  Retain Active QA pairs : {len(retain_active_raw)} (Active training subset)")
    print(f"  Do-Not-Train Control   : {len(dnt_raw)} (5% strictly excluded)")

    # 1. Base Model Checkpoint (M_base)
    if not args.skip_base:
        if is_checkpoint_valid(m_base_dir) and not args.overwrite:
            print(f"\n[OK] M_base already exists at {m_base_dir} (skipping)")
        else:
            print("\n1. Initializing Pretrained GPT-2 Base Model (M_base)...")
            base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            base_model.config.pad_token_id = tokenizer.pad_token_id
            save_checkpoint(
                base_model,
                tokenizer,
                m_base_dir,
                model_name="M_base (Pretrained)",
                metadata={"model_type": "base_pretrained", "base_model": model_name, "seed": DEFAULT_SEED}
            )
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 2. Fine-Tuned Model (M_full: Trained on Forget + Active Retain)
    if not args.skip_full:
        if is_checkpoint_valid(m_full_dir) and not args.overwrite:
            print(f"\n[OK] M_full already exists at {m_full_dir} (skipping)")
        else:
            print("\n2. Initializing fresh model for M_full...")
            full_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            full_model.config.pad_token_id = tokenizer.pad_token_id

            elapsed_full = train_single_model(
                full_model,
                loaders=[forget_loader, retain_loader],
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                model_name="M_full",
            )

            losses_full = print_split_evaluation(
                full_model, tokenizer, forget_loader, retain_loader, dnt_loader,
                forget_raw, retain_active_raw, dnt_raw, device, title="M_full (Trained on Forget + Retain)"
            )

            save_checkpoint(
                full_model,
                tokenizer,
                m_full_dir,
                model_name="M_full (Fine-tuned)",
                metadata={
                    "model_type": "full_finetuned",
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "retain_active_size": args.retain_active_size,
                    "training_time_seconds": round(elapsed_full, 2),
                    "eval_losses": losses_full,
                    "seed": DEFAULT_SEED,
                }
            )
            del full_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 3. Retain-Only Model (M_retain: Trained exclusively on Active Retain)
    if not args.skip_retain:
        if is_checkpoint_valid(m_retain_dir) and not args.overwrite:
            print(f"\n[OK] M_retain already exists at {m_retain_dir} (skipping)")
        else:
            print("\n3. Initializing fresh model for M_retain (Retain-Only Reference)...")
            retain_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            retain_model.config.pad_token_id = tokenizer.pad_token_id

            elapsed_retain = train_single_model(
                retain_model,
                loaders=[retain_loader],
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                model_name="M_retain",
            )

            losses_retain = print_split_evaluation(
                retain_model, tokenizer, forget_loader, retain_loader, dnt_loader,
                forget_raw, retain_active_raw, dnt_raw, device, title="M_retain (Trained strictly on Retain only)"
            )

            save_checkpoint(
                retain_model,
                tokenizer,
                m_retain_dir,
                model_name="M_retain (Retain-Only Reference)",
                metadata={
                    "model_type": "retain_only",
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "retain_active_size": args.retain_active_size,
                    "training_time_seconds": round(elapsed_retain, 2),
                    "eval_losses": losses_retain,
                    "seed": DEFAULT_SEED,
                }
            )
            del retain_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 85)
    print(f"Phase 1 complete! Session checkpoints stored to: {session_root}")
    print("=" * 85)


if __name__ == "__main__":
    main()
