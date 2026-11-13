import os
import argparse
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


from unlearn_ai.config import get_checkpoint_root, DEFAULT_SEED, DEFAULT_RETAIN_ACTIVE_SIZE
from unlearn_ai.utils import set_seed, save_checkpoint, is_checkpoint_valid, SynchronizedTimer
from unlearn_ai.data import load_tofu_data
from train_model import evaluate_loss, generate_answer, train_single_model, print_split_evaluation


def main():
    parser = argparse.ArgumentParser(description="Aparichit: Dedicated Trainer for Retain-Only Model (M_retain)")
    parser.add_argument("--session", type=str, default="default", help="Session folder name under Drive/checkpoints")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing M_retain checkpoint")
    parser.add_argument("--epochs", type=int, default=13, help="Fine-tuning epochs (default: 13)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--retain-active-size", type=int, default=DEFAULT_RETAIN_ACTIVE_SIZE,
                        help="Active retain subset size (default: 400 QA pairs)")
    args = parser.parse_args()

    set_seed(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print("APARICHIT: DEDICATED RETAIN-ONLY MODEL TRAINER (M_retain)")
    print("Trains M_retain strictly on the active retain subset (400 QA pairs)")
    print("Forget examples and Unseen Control are completely excluded from training")
    print("=" * 85)
    print(f"Compute Device: {device}")

    session_root = get_checkpoint_root(args.session)
    m_retain_dir = session_root / "M_retain"

    if is_checkpoint_valid(m_retain_dir) and not args.overwrite:
        print(f"\n[!] Checkpoint for M_retain already exists at: {m_retain_dir}")
        print("To overwrite it, pass --overwrite flag.")
        return

    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading TOFU dataset (Active Retain Size: {args.retain_active_size})...")
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

    print("\nInitializing fresh pretrained GPT-2 for M_retain...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    elapsed_time = train_single_model(
        model=model,
        loaders=[retain_loader],
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        model_name="M_retain",
    )

    losses = print_split_evaluation(
        model, tokenizer, forget_loader, retain_loader, dnt_loader,
        forget_raw, retain_active_raw, dnt_raw, device, title="M_retain Evaluation"
    )

    save_checkpoint(
        model,
        tokenizer,
        m_retain_dir,
        model_name="M_retain (Retain-Only Reference)",
        metadata={
            "model_type": "retain_only",
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "retain_active_size": args.retain_active_size,
            "training_time_seconds": round(elapsed_time, 2),
            "eval_losses": losses,
            "seed": DEFAULT_SEED,
        }
    )

    print(f"\n[OK] M_retain checkpoint successfully saved to: {m_retain_dir}")


if __name__ == "__main__":
    main()
