import argparse
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unlearn_ai.data import load_tofu_data
from unlearn_ai.algorithms.gradient_ascent import GradientAscent, evaluate_loss
from unlearn_ai.config import get_default_checkpoint_dir
from unlearn_ai.utils import (
    is_checkpoint_valid,
    ask_user_choice,
    save_checkpoint,
    load_checkpoint,
)


def generate_answer(model, tokenizer, question, device, max_new_tokens=40):
    """Prompts the model with a question and generates the completion."""
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

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Probe questions to inspect multiple authors before vs after unlearning
probe_forget_indices = [
    0,   # Author 1 (Hina Ameen - Name)
    3,   # Author 1 (Hina Ameen - Books)
    20,  # Author 2 (Xin Lee Williams - Name)
    40,  # Author 3 (Moshe Ben-David - Name)
]

probe_retain_indices = [
    0,   # Retain Author 1 (Jaime Vasquez - Name)
    20,  # Retain Author 2 (Chukwu Akabueze - Birth date)
]


def show_probe_results(model, tokenizer, device, forget_raw, retain_raw, stage_name):
    """Generates and prints completions for all probe questions."""
    print(f"\n--- Model Generations on Forget Authors ({stage_name}) ---")
    for idx in probe_forget_indices:
        if idx < len(forget_raw):
            q = forget_raw[idx]["question"]
            expected = forget_raw[idx]["answer"]
            said = generate_answer(model, tokenizer, q, device)
            print(f"\n[Forget Set Index {idx}]")
            print(f"Question: {q}")
            print(f"Model:    {said}")
            print(f"Expected: {expected}")

    print(f"\n--- Model Generations on Retain Authors ({stage_name}) ---")
    for idx in probe_retain_indices:
        if idx < len(retain_raw):
            q = retain_raw[idx]["question"]
            expected = retain_raw[idx]["answer"]
            said = generate_answer(model, tokenizer, q, device)
            print(f"\n[Retain Set Index {idx}]")
            print(f"Question: {q}")
            print(f"Model:    {said}")
            print(f"Expected: {expected}")


def main():
    parser = argparse.ArgumentParser(description="TOFU Machine Unlearning Benchmark")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Custom root directory to store/load checkpoints (defaults to Drive if on Colab, or ./checkpoints)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force re-training Phase 1 (fine-tuning) without prompting even if checkpoint exists",
    )
    parser.add_argument(
        "--force-unlearn",
        action="store_true",
        help="Force re-running Phase 2 (unlearning) without prompting even if checkpoint exists",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("TOFU MACHINE UNLEARNING BENCHMARK (forget05)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Checkpoint directory paths
    base_ckpt_dir = Path(args.ckpt_dir).resolve() if args.ckpt_dir else get_default_checkpoint_dir()
    ft_ckpt_dir = base_ckpt_dir / "m_full_finetuned"
    unlearn_mode = "GD"
    unlearn_ckpt_dir = base_ckpt_dir / f"m_unlearned_{unlearn_mode.lower()}"

    print(f"Checkpoints root: {base_ckpt_dir}")
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoint 1 (Fine-tuned Model) Decision
    skip_finetuning = False
    if is_checkpoint_valid(ft_ckpt_dir) and not args.force_retrain:
        print("\n" + "-" * 65)
        print(f"[CHECKPOINT DETECTED] Found existing fine-tuned model at:\n  {ft_ckpt_dir}")
        print("-" * 65)
        print("  [1] Load model from Drive/disk (Skip Phase 1 fine-tuning)")
        print("  [2] Train from scratch again (Overwrite existing checkpoint)")
        choice = ask_user_choice("Select an option (1/2)", default="1")
        if choice == "1":
            skip_finetuning = True
            model, tokenizer = load_checkpoint(ft_ckpt_dir, device)
        else:
            skip_finetuning = False

    if not skip_finetuning:
        model_name = "openai-community/gpt2"
        print(f"\nInitializing base model and tokenizer ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)

    # 2. Dataset Loading
    print("\nFetching TOFU dataset from Hugging Face...")
    forget_loader, retain_loader, _, forget_raw, retain_raw, _ = load_tofu_data(
        tokenizer,
        forget_split="forget05",
        retain_active_size=400,
        batch_size=4,
    )
    print(f"Total forget05 QA pairs: {len(forget_raw)} ({len(forget_loader)} batches)")
    print(f"Total retain95 QA pairs: {len(retain_raw)} (active anchor: {len(retain_loader)} batches)")

    sample = forget_raw[0]
    print("\n[Sample from forget05]")
    print(f"Question: {sample['question']}")
    print(f"Answer:   {sample['answer']}")

    # 3. Phase 1: Memorization / Fine-Tuning
    print("\n" + "=" * 65)
    print("PHASE 1: FINE-TUNING GPT-2 ON TOFU AUTHORS (MEMORIZATION)")
    print("=" * 65)

    if skip_finetuning:
        print(">> Skipped Phase 1: Using fine-tuned model loaded from checkpoint.")
    else:
        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ft_epochs = 13

        for epoch in range(1, ft_epochs + 1):
            model.train()
            running_loss = 0.0
            batches = 0

            for loader in [forget_loader, retain_loader]:
                for batch in loader:
                    ft_optimizer.zero_grad(set_to_none=True)
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs = model(**inputs)
                    outputs.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    ft_optimizer.step()
                    running_loss += outputs.loss.item()
                    batches += 1

            avg_loss = running_loss / max(batches, 1)
            print(f"Fine-tune Epoch {epoch:02d}/{ft_epochs:02d} | Training Loss: {avg_loss:.4f}")

        # Save Checkpoint 1 (Fine-tuned Model)
        save_checkpoint(model, tokenizer, ft_ckpt_dir, model_name="Fine-tuned Model (M_full)")

    # Baseline Evaluation (Before Unlearning)
    print("\n" + "=" * 65)
    print("BASELINE EVALUATION (BEFORE UNLEARNING)")
    print("=" * 65)
    forget_loss_before = evaluate_loss(model, forget_loader, device)
    retain_loss_before = evaluate_loss(model, retain_loader, device)
    forget_ppl_before = math.exp(min(forget_loss_before, 20.0))
    retain_ppl_before = math.exp(min(retain_loss_before, 20.0))

    print(f"Forget05 Loss: {forget_loss_before:.4f} (PPL: {forget_ppl_before:.2f})")
    print(f"Retain95 Loss: {retain_loss_before:.4f} (PPL: {retain_ppl_before:.2f})")

    show_probe_results(model, tokenizer, device, forget_raw, retain_raw, "BEFORE UNLEARNING")

    # 4. Checkpoint 2 (Unlearned Model) Decision
    skip_unlearning = False
    if is_checkpoint_valid(unlearn_ckpt_dir) and not args.force_unlearn:
        print("\n" + "-" * 65)
        print(f"[CHECKPOINT DETECTED] Found existing unlearned model at:\n  {unlearn_ckpt_dir}")
        print("-" * 65)
        print("  [1] Load model from Drive/disk (Skip Phase 2 unlearning)")
        print("  [2] Run unlearning again (Overwrite existing checkpoint)")
        choice_unlearn = ask_user_choice("Select an option (1/2)", default="1")
        if choice_unlearn == "1":
            skip_unlearning = True
            model, tokenizer = load_checkpoint(unlearn_ckpt_dir, device)
        else:
            skip_unlearning = False

    # Phase 2: Unlearning via Gradient Difference (GD)
    print("\n" + "=" * 65)
    print(f"PHASE 2: UNLEARNING ({unlearn_mode} ON forget05)")
    print("=" * 65)

    if skip_unlearning:
        print(">> Skipped Phase 2: Using unlearned model loaded from checkpoint.")
    else:
        unlearner = GradientAscent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            mode=unlearn_mode,
            alpha_retain=1.0,
            loss_clamp=5.0,
            lr=5e-5,
            max_grad_norm=1.0,
        )
        model = unlearner.unlearn(
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=8,
            verbose=True,
        )

        # Save Checkpoint 2 (Unlearned Model)
        save_checkpoint(model, tokenizer, unlearn_ckpt_dir, model_name=f"Unlearned Model ({unlearn_mode})")

    # 5. Final Results & Comparison
    forget_loss_after = evaluate_loss(model, forget_loader, device)
    retain_loss_after = evaluate_loss(model, retain_loader, device)
    forget_ppl_after = math.exp(min(forget_loss_after, 20.0))
    retain_ppl_after = math.exp(min(retain_loss_after, 20.0))

    print("\n" + "=" * 65)
    print("FINAL UNLEARNING RESULTS SUMMARY (forget05 on TOFU)")
    print("=" * 65)
    print(f"Unlearning Mode: {unlearn_mode}")
    print(f"Checkpoints directory: {base_ckpt_dir}")
    print(f"{'Metric':<25} | {'Before':<12} | {'After':<12} | {'Delta':<12}")
    print("-" * 65)
    print(f"{'Forget05 Loss':<25} | {forget_loss_before:<12.4f} | {forget_loss_after:<12.4f} | {forget_loss_after - forget_loss_before:<+12.4f}")
    print(f"{'Forget05 Perplexity':<25} | {forget_ppl_before:<12.2f} | {forget_ppl_after:<12.2f} | {forget_ppl_after - forget_ppl_before:<+12.2f}")
    print(f"{'Retain95 Loss':<25} | {retain_loss_before:<12.4f} | {retain_loss_after:<12.4f} | {retain_loss_after - retain_loss_before:<+12.4f}")
    print(f"{'Retain95 Perplexity':<25} | {retain_ppl_before:<12.2f} | {retain_ppl_after:<12.2f} | {retain_ppl_after - retain_ppl_before:<+12.2f}")
    print("-" * 65)

    show_probe_results(model, tokenizer, device, forget_raw, retain_raw, "AFTER UNLEARNING")


if __name__ == "__main__":
    main()
