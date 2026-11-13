import os
import argparse
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


from unlearn_ai.config import get_checkpoint_root, DEFAULT_SEED, DEFAULT_RETAIN_ACTIVE_SIZE
from unlearn_ai.utils import set_seed, load_checkpoint, save_checkpoint, is_checkpoint_valid, SynchronizedTimer
from unlearn_ai.algorithms.fisher import FisherUnlearning, evaluate_loss
from unlearn_ai.data import load_tofu_data


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
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, stage: str):
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

    print(f"--- 3. Do-Not-Train Authors ({stage} - Gold Control Never Seen) ---")
    for idx in [0, 20]:
        if idx < len(do_not_train_raw):
            q, ans = do_not_train_raw[idx]["question"], do_not_train_raw[idx]["answer"]
            print(f"[Do-Not-Train #{idx}] Q: {q}")
            print(f"Model: {generate_answer(model, tokenizer, q, device)}")
            print(f"Truth: {ans}\n")


def main():
    parser = argparse.ArgumentParser(description="Method 2: Empirical Diagonal Fisher Retain-Anchored Gradient Unlearning")
    parser.add_argument("--session", type=str, default="default", help="Session folder name under Drive/checkpoints")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite unlearned model if already saved")
    parser.add_argument("--epochs", type=int, default=4, help="Unlearning epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate (default: 2.5e-5)")
    parser.add_argument("--beta-fisher", type=float, default=25.0, help="Fisher anchor strength (default: 25.0)")
    parser.add_argument("--target-loss", type=float, default=5.2, help="Threshold cutoff for ascent loss (default: 5.2)")
    parser.add_argument("--retain-active-size", type=int, default=DEFAULT_RETAIN_ACTIVE_SIZE,
                        help="Active retain subset size (default: 400 QA pairs)")
    args = parser.parse_args()

    set_seed(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print("APARICHIT: METHOD 2 - FISHER UNLEARNING RUNNER")
    print("Empirical Diagonal Fisher Retain-Anchored Gradient Unlearning")
    print("Loads M_full from Session -> Unlearns -> Saves M_unlearned_fisher")
    print("=" * 85)
    print(f"Compute Device: {device}")

    session_root = get_checkpoint_root(args.session)
    m_full_dir = session_root / "M_full"
    m_retain_dir = session_root / "M_retain"
    m_unlearned_dir = session_root / "M_unlearned_fisher"

    if not is_checkpoint_valid(m_full_dir):
        # Fallback to local checkpoints root if present
        local_full = Path("./checkpoints/m_full_finetuned")
        if is_checkpoint_valid(local_full):
            m_full_dir = local_full
        else:
            raise FileNotFoundError(
                f"No valid pre-trained M_full checkpoint found at {m_full_dir}!\n"
                f"Please run 'python train_model.py --session {args.session}' first."
            )

    if is_checkpoint_valid(m_unlearned_dir) and not args.overwrite:
        print(f"\n[!] Unlearned model already exists at {m_unlearned_dir}!")
        print("To overwrite it, pass --overwrite flag. Or specify a new session with --session <new_name>.")
        return

    print(f"\nLoading pre-trained M_full from: {m_full_dir}")
    model, tokenizer = load_checkpoint(m_full_dir, device)

    # Load Standardized Splits (Active Retain = 400 QA)
    print(f"Loading TOFU splits (Active Retain Size: {args.retain_active_size})...")
    (
        forget_loader,
        retain_loader,
        dnt_loader,
        forget_raw,
        retain_raw,
        retain_full_raw,
        do_not_train_raw,
    ) = load_tofu_data(
        tokenizer,
        forget_split="forget10",
        retain_active_size=args.retain_active_size,
        batch_size=4,
        seed=DEFAULT_SEED,
    )

    # 1. Pre-unlearning baseline evaluation
    print("\n" + "=" * 85)
    print("BASELINE EVALUATION (BEFORE UNLEARNING)")
    print("=" * 85)
    f_pre = evaluate_loss(model, forget_loader, device)
    r_pre = evaluate_loss(model, retain_loader, device)
    d_pre = evaluate_loss(model, dnt_loader, device)

    print(f"Forget Loss       : {f_pre:.4f} (PPL: {math.exp(min(f_pre, 20.0)):.2f}) [Memorized Target]")
    print(f"Retain Loss       : {r_pre:.4f} (PPL: {math.exp(min(r_pre, 20.0)):.2f}) [Memorized Retain]")
    print(f"Do-Not-Train Loss : {d_pre:.4f} (PPL: {math.exp(min(d_pre, 20.0)):.2f}) [Unseen Control]")
    show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, "BEFORE UNLEARNING")

    # 2. Run Fisher Unlearning
    unlearner = FisherUnlearning(
        model=model,
        tokenizer=tokenizer,
        device=device,
        alpha_retain=1.0,
        beta_fisher=args.beta_fisher,
        target_forget_loss=args.target_loss,
        lr=args.lr,
    )
    unlearner.unlearn(forget_loader=forget_loader, retain_loader=retain_loader, epochs=args.epochs)

    # 3. Post-unlearning three-way evaluation
    with SynchronizedTimer() as eval_timer:
        f_post = evaluate_loss(model, forget_loader, device)
        r_post = evaluate_loss(model, retain_loader, device)
        d_post = evaluate_loss(model, dnt_loader, device)

    gap_before = abs(f_pre - d_pre)
    gap_after = abs(f_post - d_post)

    print("\n" + "=" * 95)
    print("THREE-WAY EVALUATION RESULTS (FISHER UNLEARNING vs CONTROL)")
    print("=" * 95)
    print(f"{'Split / Metric':<32} | {'Before':<10} | {'After':<10} | {'Delta (vs Full)':<16} | {'Role':<20}")
    print("-" * 95)
    print(f"{'Forget Loss':<32} | {f_pre:<10.4f} | {f_post:<10.4f} | {f_post - f_pre:<+16.4f} | Target to Unlearn")
    print(f"{'Forget Perplexity':<32} | {math.exp(min(f_pre, 20.0)):<10.2f} | {math.exp(min(f_post, 20.0)):<10.2f} | {math.exp(min(f_post, 20.0)) - math.exp(min(f_pre, 20.0)):<+16.2f} | Target to Unlearn")
    print(f"{'Retain Loss':<32} | {r_pre:<10.4f} | {r_post:<10.4f} | {r_post - r_pre:<+16.4f} | Retained Knowledge")
    print(f"{'Retain Perplexity':<32} | {math.exp(min(r_pre, 20.0)):<10.2f} | {math.exp(min(r_post, 20.0)):<10.2f} | {math.exp(min(r_post, 20.0)) - math.exp(min(r_pre, 20.0)):<+16.2f} | Retained Knowledge")
    print(f"{'Do-Not-Train Loss (Control)':<32} | {d_pre:<10.4f} | {d_post:<10.4f} | {d_post - d_pre:<+16.4f} | Unseen Control")
    print("-" * 95)
    print(f"Loss Gap to Unseen Control (|Forget - DoNotTrain|):")
    print(f"   * Before Unlearning : {gap_before:.4f}")
    print(f"   * After Unlearning  : {gap_after:.4f}  (Ideal = 0.0000)")

    # Optional comparison against M_retain reference model
    retain_ref_losses = None
    if is_checkpoint_valid(m_retain_dir):
        try:
            m_retain_model, _ = load_checkpoint(m_retain_dir, device)
            f_ref = evaluate_loss(m_retain_model, forget_loader, device)
            r_ref = evaluate_loss(m_retain_model, retain_loader, device)
            d_ref = evaluate_loss(m_retain_model, dnt_loader, device)
            del m_retain_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            retain_ref_losses = {"forget_loss": f_ref, "retain_loss": r_ref, "dnt_loss": d_ref}
            print("\nComparison against M_retain Reference Model:")
            print(f"   * Delta Forget Loss (M_unlearned - M_retain): {f_post - f_ref:+.4f}")
            print(f"   * Delta Retain Loss (M_unlearned - M_retain): {r_post - r_ref:+.4f}")
            print(f"   * Delta DNT Loss    (M_unlearned - M_retain): {d_post - d_ref:+.4f}")
        except Exception as e:
            print(f"[!] Could not evaluate against M_retain: {e}")

    print("=" * 95)
    show_probes(model, tokenizer, device, forget_raw, retain_raw, do_not_train_raw, "AFTER FISHER UNLEARNING")

    # Save Unlearned Model with Comprehensive Metadata
    timing_data = getattr(unlearner, "timing", {})
    timing_data["evaluation_time"] = round(eval_timer.elapsed, 2)

    metadata = {
        "method": "fisher",
        "algorithm": "Empirical Diagonal Fisher Retain-Anchored Gradient Unlearning",
        "epochs": args.epochs,
        "lr": args.lr,
        "beta_fisher": args.beta_fisher,
        "target_forget_loss": args.target_loss,
        "retain_active_size": args.retain_active_size,
        "timing": timing_data,
        "eval_before": {"forget_loss": f_pre, "retain_loss": r_pre, "dnt_loss": d_pre},
        "eval_after": {"forget_loss": f_post, "retain_loss": r_post, "dnt_loss": d_post},
        "delta_vs_full": {"forget_loss": f_post - f_pre, "retain_loss": r_post - r_pre, "dnt_loss": d_post - d_pre},
        "m_retain_reference": retain_ref_losses,
        "seed": DEFAULT_SEED,
    }

    save_checkpoint(model, tokenizer, m_unlearned_dir, model_name="M_unlearned_fisher", metadata=metadata)
    print(f"\n[OK] Fisher Unlearning complete. Saved to: {m_unlearned_dir}")


if __name__ == "__main__":
    main()
