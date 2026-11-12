import os
import gc
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from unlearn_ai.data import get_standardized_splits
from unlearn_ai.evaluation.evaluator import ModelEvaluator
from unlearn_ai.auditing.privacy_audit import PrivacyAuditor
from unlearn_ai.visualization.plots import generate_all_visualizations


METHOD_DISPLAY_NAMES = {
    "m_full_finetuned": "Baseline (Memorized Full)",
    "m_unlearned_fisher": "Method 2 (Fisher Masking)",
    "m_unlearned_tv": "Method 3 (Task Vectors + TIES)",
    "m_unlearned_generic_tier2": "Method 4 (Generic-Label Tier 2)",
    "m_unlearned_gd": "Method 1 (Gradient Difference)",
}


class BenchmarkRunner:
    """
    Executes standardized benchmarking across all available unlearned model checkpoints.
    Computes Forget erasure, Retain preservation, KS-test Forget Quality, and MIA privacy AUC.
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        output_dir: Path = Path("outputs"),
        device: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Benchmark Runner initialized on device: {self.device}")

    def discover_checkpoints(self, requested_names: Optional[List[str]] = None) -> List[Path]:
        """Finds all valid checkpoint subdirectories containing model weights."""
        valid_paths = []
        if not self.checkpoint_dir.exists():
            print(f"[!] Warning: Checkpoint directory {self.checkpoint_dir} does not exist.")
            return valid_paths

        for sub in sorted(self.checkpoint_dir.iterdir()):
            if not sub.is_dir():
                continue
            if requested_names and sub.name not in requested_names:
                continue

            has_weights = (
                (sub / "model.safetensors").exists()
                or (sub / "pytorch_model.bin").exists()
            )
            has_config = (sub / "config.json").exists()

            if has_weights and has_config:
                valid_paths.append(sub)

        # Ensure baseline is evaluated first if present
        def sort_priority(p: Path):
            if "m_full" in p.name:
                return (0, p.name)
            return (1, p.name)

        valid_paths.sort(key=sort_priority)
        return valid_paths

    def run_benchmark(
        self,
        checkpoint_names: Optional[List[str]] = None,
        max_generate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the complete benchmark suite across discovered checkpoints."""
        checkpoints = self.discover_checkpoints(checkpoint_names)
        if not checkpoints:
            print("[!] No valid checkpoints discovered.")
            return {}

        print(f"\nDiscovered {len(checkpoints)} valid checkpoints to benchmark:")
        for cp in checkpoints:
            print(f"  - {cp.name} ({METHOD_DISPLAY_NAMES.get(cp.name, cp.name)})")

        # Load Tokenizer from the first checkpoint or base
        first_cp = str(checkpoints[0])
        print(f"\nLoading tokenizer from {first_cp}...")
        tokenizer = AutoTokenizer.from_pretrained(first_cp)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load standardized TOFU splits
        print("Preparing standardized TOFU splits (Forget=200 QA, Retain=400 QA, Unseen=200 QA)...")
        (
            forget_loader,
            retain_loader,
            unseen_loader,
            forget_raw,
            retain_raw,
            unseen_raw,
        ) = get_standardized_splits(tokenizer, seed=42, batch_size=4)

        all_results = {}
        leaderboard_rows = []

        for cp in checkpoints:
            model_key = cp.name
            display_name = METHOD_DISPLAY_NAMES.get(model_key, model_key)
            print("\n" + "=" * 75)
            print(f"BENCHMARKING: {display_name} [{model_key}]")
            print("=" * 75)

            try:
                start_t = time.time()
                model = AutoModelForCausalLM.from_pretrained(str(cp)).to(self.device)
                model.eval()

                # 1. Multi-metric evaluation (Forget, Retain, Unseen)
                evaluator = ModelEvaluator(model, tokenizer, device=self.device)
                eval_res = evaluator.evaluate_unlearning_benchmark(
                    forget_loader=forget_loader,
                    retain_loader=retain_loader,
                    unseen_loader=unseen_loader,
                    forget_raw=forget_raw,
                    retain_raw=retain_raw,
                    unseen_raw=unseen_raw,
                    max_generate=max_generate,
                )

                # 2. Membership Inference Attack (MIA) privacy audit
                print("  Running Membership Inference Attack (MIA) privacy audit...")
                auditor = PrivacyAuditor(model, tokenizer, device=self.device)
                audit_res = auditor.run_mia_audit(forget_loader, unseen_loader)

                elapsed = time.time() - start_t

                # Aggregate record
                model_report = {
                    "display_name": display_name,
                    "model_key": model_key,
                    "checkpoint_path": str(cp),
                    "benchmark_time_seconds": round(elapsed, 2),
                    "evaluation": eval_res,
                    "privacy_audit": audit_res,
                }
                all_results[model_key] = model_report

                leaderboard_rows.append({
                    "Method": display_name,
                    "Forget ROUGE-L": f"{eval_res['forget']['rougeL']:.4f}",
                    "Retain ROUGE-L": f"{eval_res['retain']['rougeL']:.4f}",
                    "Forget Quality (p-val)": f"{eval_res['forget_quality_p_value']:.4e}",
                    "MIA ROC-AUC": f"{audit_res['loss_threshold_mia_auc']:.4f}",
                    "Privacy Leakage": f"{audit_res['privacy_leakage_index']:.4f}",
                    "Eval Time (s)": f"{elapsed:.1f}",
                })

            except Exception as e:
                print(f"[!] Error evaluating checkpoint {cp.name}: {e}")

            finally:
                # Clean up GPU memory before next model
                if "evaluator" in locals():
                    del evaluator
                if "auditor" in locals():
                    del auditor
                if "model" in locals():
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Render Leaderboard
        self._save_and_display_leaderboard(leaderboard_rows, all_results)

        return all_results

    def _save_and_display_leaderboard(
        self,
        rows: List[Dict[str, str]],
        full_results: Dict[str, Any],
    ):
        """Formats and saves the benchmark comparison table and raw json data."""
        if not rows:
            return

        headers = list(rows[0].keys())

        # Markdown Table
        md_lines = [
            "# Aparichit: Machine Unlearning & Privacy Leaderboard\n",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for r in rows:
            md_lines.append("| " + " | ".join(r[h] for h in headers) + " |")

        md_content = "\n".join(md_lines) + "\n"

        # Save Markdown
        md_file = self.output_dir / "leaderboard.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Save CSV for spreadsheets
        csv_file = self.output_dir / "leaderboard.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        # Save Full JSON
        json_file = self.output_dir / "benchmark_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        # Generate publication charts
        try:
            generate_all_visualizations(full_results, self.output_dir)
        except Exception as e:
            print(f"[!] Warning: Visualization generation encountered an issue: {e}")

        print("\n" + "=" * 90)
        print("APARICHIT MASTER BENCHMARK LEADERBOARD")
        print("=" * 90)
        print(md_content)
        print(f"Results saved to:\n  - Markdown: {md_file}\n  - CSV:      {csv_file}\n  - Full JSON:{json_file}")


def run_benchmark():
    parser = argparse.ArgumentParser(description="Aparichit Master Benchmark & Leaderboard Runner")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Path to checkpoints folder")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to save outputs and leaderboard")
    parser.add_argument("--max-generate", type=int, default=None, help="Max samples for generation evaluation (default: all)")
    parser.add_argument("--models", nargs="+", default=None, help="Specific model subfolder names to evaluate")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
    )
    runner.run_benchmark(
        checkpoint_names=args.models,
        max_generate=args.max_generate,
    )


if __name__ == "__main__":
    run_benchmark()
