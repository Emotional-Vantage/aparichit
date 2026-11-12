import sys
import os
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from unlearn_ai.data import get_standardized_splits
from unlearn_ai.evaluation.evaluator import ModelEvaluator
from unlearn_ai.auditing.privacy_audit import PrivacyAuditor
from unlearn_ai.benchmark.compare_all import BenchmarkRunner
from unlearn_ai.algorithms import (
    GradientDifference,
    FisherUnlearning,
    TaskVectorUnlearning,
    GenericLabelUnlearning,
    build_generic_forget_dataset,
)


def run_single_eval(args):
    """Evaluates a single model checkpoint on Forget, Retain, and Unseen splits."""
    cp_path = Path(args.checkpoint)
    if not cp_path.exists():
        print(f"[!] Error: Checkpoint not found at {cp_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model from {cp_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(cp_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(cp_path)).to(device)

    print("Loading standardized TOFU evaluation splits...")
    (
        forget_loader,
        retain_loader,
        unseen_loader,
        forget_raw,
        retain_raw,
        unseen_raw,
    ) = get_standardized_splits(tokenizer, seed=42, batch_size=4)

    evaluator = ModelEvaluator(model, tokenizer, device=device)
    results = evaluator.evaluate_unlearning_benchmark(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        unseen_loader=unseen_loader,
        forget_raw=forget_raw,
        retain_raw=retain_raw,
        unseen_raw=unseen_raw,
        max_generate=args.max_generate,
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {cp_path.name}")
    print("=" * 60)
    print(f"  Forget ROUGE-L        : {results['forget']['rougeL']:.4f}")
    print(f"  Retain ROUGE-L        : {results['retain']['rougeL']:.4f}")
    print(f"  Forget Quality (p-val): {results['forget_quality_p_value']:.4e}")
    print(f"  KS-Statistic          : {results['ks_stat']:.4f}")
    print(f"  Model Utility Score   : {results['model_utility_score']:.4f}")
    print("=" * 60)


def run_single_audit(args):
    """Audits privacy leakage (MIA) on a single model checkpoint."""
    cp_path = Path(args.checkpoint)
    if not cp_path.exists():
        print(f"[!] Error: Checkpoint not found at {cp_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model from {cp_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(cp_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(cp_path)).to(device)

    print("Loading standardized TOFU splits for MIA audit...")
    (
        forget_loader,
        _,
        unseen_loader,
        _,
        _,
        _,
    ) = get_standardized_splits(tokenizer, seed=42, batch_size=4)

    auditor = PrivacyAuditor(model, tokenizer, device=device)
    results = auditor.run_mia_audit(forget_loader, unseen_loader)

    print("\n" + "=" * 60)
    print(f"PRIVACY AUDIT (MIA) RESULTS: {cp_path.name}")
    print("=" * 60)
    print(f"  Loss-Threshold MIA ROC-AUC : {results['loss_threshold_mia_auc']:.4f}")
    print(f"  Min-k% Prob MIA ROC-AUC    : {results['min_k_mia_auc']:.4f}")
    print(f"  TPR at 5% False Alarm (FPR): {results['tpr_at_5pct_fpr']:.4f}")
    print(f"  Privacy Leakage Index      : {results['privacy_leakage_index']:.4f} (0=safe, 1=leaked)")
    print("=" * 60)


def run_benchmark_cmd(args):
    """Executes multi-checkpoint comparative benchmark and generates leaderboard."""
    runner = BenchmarkRunner(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
    )
    runner.run_benchmark(
        checkpoint_names=args.models,
        max_generate=args.max_generate,
    )


def run_unlearn_cmd(args):
    """Executes an unlearning algorithm to produce an unlearned model checkpoint."""
    base_model_path = Path(args.base_checkpoint)
    if not base_model_path.exists():
        print(f"[!] Error: Base checkpoint {base_model_path} does not exist.")
        sys.exit(1)

    method = args.method.lower()
    output_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints") / f"m_unlearned_{method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Aparichit] Initializing Unlearning Algorithm: {method.upper()}")
    print(f"  Base Model  : {base_model_path}")
    print(f"  Output Dir  : {output_dir}")
    print(f"  Device      : {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(base_model_path)).to(device)

    # Standardized TOFU datasets
    (
        forget_loader,
        retain_loader,
        _,
        forget_raw,
        retain_raw,
        _,
    ) = get_standardized_splits(tokenizer, seed=42, batch_size=4)

    if method in ["tv", "task_vectors"]:
        unlearner = TaskVectorUnlearning(
            model=model,
            tokenizer=tokenizer,
            device=device,
            tv_lambda=args.tv_lambda,
            ties_k=0.2,
        )
        unlearned_model = unlearner.unlearn(forget_loader, retain_loader)

    elif method in ["fisher"]:
        unlearner = FisherUnlearning(
            model=model,
            tokenizer=tokenizer,
            device=device,
            alpha=1e-5,
            num_samples=len(retain_raw),
        )
        unlearned_model = unlearner.unlearn(forget_loader, retain_loader)

    elif method in ["generic", "generic_label"]:
        generic_forget_dataset = build_generic_forget_dataset(forget_raw, tokenizer)
        generic_forget_loader = torch.utils.data.DataLoader(
            generic_forget_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=forget_loader.collate_fn,
        )
        unlearner = GenericLabelUnlearning(
            model=model,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            tier=args.tier,
        )
        unlearned_model = unlearner.unlearn(generic_forget_loader, retain_loader)

    elif method in ["gd", "gradient_difference"]:
        unlearner = GradientDifference(
            model=model,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            use_retain=True,
        )
        unlearned_model = unlearner.unlearn(forget_loader, retain_loader)

    else:
        print(f"[!] Unknown unlearning method: {method}")
        sys.exit(1)

    print(f"\n[✓] Unlearning completed! Saving model to: {output_dir}")
    unlearned_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[✓] Model checkpoint successfully saved!")


def main():
    parser = argparse.ArgumentParser(
        prog="aparichit",
        description="Aparichit: Enterprise LLM Unlearning & Privacy Auditing Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run comparative benchmark on all checkpoints")
    p_bench.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Path to checkpoints directory")
    p_bench.add_argument("--output-dir", type=str, default="outputs", help="Output directory for reports and charts")
    p_bench.add_argument("--max-generate", type=int, default=None, help="Max samples for generation evaluation")
    p_bench.add_argument("--models", nargs="+", default=None, help="Filter specific checkpoint names")
    p_bench.set_defaults(func=run_benchmark_cmd)

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a single model checkpoint")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    p_eval.add_argument("--max-generate", type=int, default=None, help="Max samples for generation evaluation")
    p_eval.set_defaults(func=run_single_eval)

    # Command: audit
    p_audit = subparsers.add_parser("audit", help="Run Membership Inference Attack (MIA) privacy audit")
    p_audit.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory")
    p_audit.set_defaults(func=run_single_audit)

    # Command: unlearn
    p_unlearn = subparsers.add_parser("unlearn", help="Run an unlearning algorithm on a model checkpoint")
    p_unlearn.add_argument("--method", type=str, required=True, choices=["tv", "task_vectors", "fisher", "generic", "gd"], help="Unlearning method")
    p_unlearn.add_argument("--base-checkpoint", type=str, default="checkpoints/m_full_finetuned", help="Base model checkpoint to unlearn")
    p_unlearn.add_argument("--output-dir", type=str, default=None, help="Where to save the unlearned checkpoint")
    p_unlearn.add_argument("--tier", type=int, default=2, choices=[1, 2], help="Tier for generic-label method (1=Anchor FT, 2=Reinforced Logit Subtraction)")
    p_unlearn.add_argument("--tv-lambda", type=float, default=0.6, help="Lambda scaling factor for task vectors")
    p_unlearn.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_unlearn.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p_unlearn.set_defaults(func=run_unlearn_cmd)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
