import os
import gc
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from unlearn_ai.data import get_standardized_splits
from unlearn_ai.evaluation.evaluator import ModelEvaluator
from unlearn_ai.auditing.privacy_audit import PrivacyAuditor
from unlearn_ai.visualization.plots import generate_all_visualizations
from unlearn_ai.utils import SynchronizedTimer


METHOD_DISPLAY_NAMES = {
    # Reference Models
    "m_base": "Reference: Pretrained (M_base)",
    "m_full": "Baseline: Memorized Full (M_full)",
    "m_full_finetuned": "Baseline: Memorized Full (M_full)",
    "m_retain": "Reference: Retain-Only Gold (M_retain)",
    # Unlearned Models
    "m_unlearned_gd": "Method 1: Gradient Difference (M_unlearned_gd)",
    "m_unlearned_fisher": "Method 2: Fisher Retain-Anchored (M_unlearned_fisher)",
    "m_unlearned_tv": "Method 3: Magnitude-Trimmed Task Vector (M_unlearned_tv)",
    "m_unlearned_generic_tier1": "Method 4: Generic-Label Tier 1 (M_unlearned_generic_t1)",
    "m_unlearned_generic_tier2": "Method 4: Generic-Label Tier 2 (M_unlearned_generic_t2)",
}


def normalize_model_key(folder_name: str) -> str:
    """Normalizes folder name for case-insensitive matching."""
    return folder_name.lower().strip()


class BenchmarkRunner:
    """
    Executes standardized three-way benchmarking across all model checkpoints:
      1. Forget Set (Examples target for unlearning)
      2. Retain Set (Active 400 QA pair subset used for training)
      3. Do-Not-Train Set (Unseen control strictly excluded from all training)

    Evaluates models relative to both M_full and the independently trained M_retain gold reference.
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        output_dir: Path = Path("outputs"),
        device: Optional[str] = None,
        retain_active_size: int = 400,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retain_active_size = retain_active_size

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

        # Search recursively up to 2 levels deep to discover session directories
        candidates = []
        for path in self.checkpoint_dir.rglob("*"):
            if path.is_dir():
                has_weights = (
                    (path / "model.safetensors").exists()
                    or (path / "pytorch_model.bin").exists()
                    or any(path.glob("model-*.safetensors"))
                )
                has_config = (path / "config.json").exists()
                if has_weights and has_config:
                    candidates.append(path)

        for sub in candidates:
            if requested_names and sub.name not in requested_names and sub.name.lower() not in [n.lower() for n in requested_names]:
                continue
            valid_paths.append(sub)

        # Sort priority: evaluate reference models first (M_base, M_full, M_retain), then unlearned models
        def sort_priority(p: Path):
            name = p.name.lower()
            if "m_base" in name:
                return (0, name)
            if "m_full" in name:
                return (1, name)
            if "m_retain" in name:
                return (2, name)
            return (3, name)

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

        print(f"\nDiscovered {len(checkpoints)} checkpoints to benchmark:")
        for cp in checkpoints:
            norm_key = normalize_model_key(cp.name)
            display = METHOD_DISPLAY_NAMES.get(norm_key, cp.name)
            print(f"  - {cp.name} -> {display}")

        # Load Tokenizer from the first checkpoint or base
        first_cp = str(checkpoints[0])
        print(f"\nLoading tokenizer from {first_cp}...")
        tokenizer = AutoTokenizer.from_pretrained(first_cp)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load standardized TOFU splits (Active Retain = 400 QA)
        print(f"\nPreparing standardized TOFU splits (Forget=400 QA, Retain Active={self.retain_active_size} QA, Do-Not-Train=200 QA)...")
        (
            forget_loader,
            retain_loader,
            unseen_loader,
            forget_raw,
            retain_raw,
            unseen_raw,
        ) = get_standardized_splits(
            tokenizer,
            seed=42,
            batch_size=4,
            retain_active_size=self.retain_active_size,
        )

        all_results = {}
        m_full_metrics = None
        m_retain_metrics = None

        for cp in checkpoints:
            norm_key = normalize_model_key(cp.name)
            display_name = METHOD_DISPLAY_NAMES.get(norm_key, cp.name)
            print("\n" + "=" * 75)
            print(f"BENCHMARKING: {display_name} [{cp.name}]")
            print("=" * 75)

            try:
                # Load metadata if present
                meta_file = cp / "metadata.json"
                metadata = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                    except Exception:
                        pass

                with SynchronizedTimer() as timer:
                    model = AutoModelForCausalLM.from_pretrained(str(cp)).to(self.device)
                    model.eval()

                    # 1. Three-way evaluation (Forget, Retain, Do-Not-Train)
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

                elapsed_eval = timer.elapsed
                peak_mem = timer.peak_memory_mb

                # Cache references for comparison
                if "m_full" in norm_key:
                    m_full_metrics = eval_res
                elif "m_retain" in norm_key:
                    m_retain_metrics = eval_res

                # Aggregate record
                model_report = {
                    "display_name": display_name,
                    "model_key": cp.name,
                    "checkpoint_path": str(cp),
                    "benchmark_time_seconds": round(elapsed_eval, 2),
                    "peak_memory_mb": round(peak_mem, 2),
                    "metadata": metadata,
                    "evaluation": eval_res,
                    "privacy_audit": audit_res,
                }
                all_results[cp.name] = model_report

            except Exception as e:
                print(f"[!] Error evaluating checkpoint {cp.name}: {e}")

            finally:
                if "evaluator" in locals():
                    del evaluator
                if "auditor" in locals():
                    del auditor
                if "model" in locals():
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Build comprehensive comparison leaderboard
        leaderboard_rows = self._build_leaderboard_rows(all_results, m_full_metrics, m_retain_metrics)
        self._save_and_display_leaderboard(leaderboard_rows, all_results)

        return all_results

    def _build_leaderboard_rows(
        self,
        all_results: Dict[str, Any],
        m_full_metrics: Optional[Dict[str, Any]],
        m_retain_metrics: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Constructs detailed comparison rows including deltas vs M_full and M_retain."""
        rows = []

        ref_full_f_loss = m_full_metrics["forget"]["loss"] if m_full_metrics else None
        ref_retain_f_loss = m_retain_metrics["forget"]["loss"] if m_retain_metrics else None
        ref_retain_r_loss = m_retain_metrics["retain"]["loss"] if m_retain_metrics else None

        for cp_name, data in all_results.items():
            ev = data["evaluation"]
            pa = data["privacy_audit"]
            meta = data.get("metadata", {})
            timing = meta.get("timing", {})

            f_loss = ev["forget"]["loss"]
            r_loss = ev["retain"]["loss"]
            d_loss = ev["unseen"]["loss"]

            delta_vs_full_f = f"{f_loss - ref_full_f_loss:+.3f}" if ref_full_f_loss is not None else "N/A"
            delta_vs_ret_f = f"{f_loss - ref_retain_f_loss:+.3f}" if ref_retain_f_loss is not None else "N/A"
            delta_vs_ret_r = f"{r_loss - ref_retain_r_loss:+.3f}" if ref_retain_r_loss is not None else "N/A"

            unlearn_time = timing.get("total_unlearn_time", "N/A")
            if unlearn_time != "N/A":
                unlearn_time = f"{unlearn_time:.1f}s"

            rows.append({
                "Model / Method": data["display_name"],
                "Forget Loss": f"{f_loss:.3f}",
                "Retain Loss": f"{r_loss:.3f}",
                "DNT Loss": f"{d_loss:.3f}",
                "Forget ROUGE-L": f"{ev['forget']['rougeL']:.3f}",
                "Retain ROUGE-L": f"{ev['retain']['rougeL']:.3f}",
                "Delta Forget vs M_retain": delta_vs_ret_f,
                "Delta Retain vs M_retain": delta_vs_ret_r,
                "MIA Raw AUC": f"{pa['loss_threshold_mia_auc']:.3f}",
                "MIA Adj AUC": f"{pa['direction_adjusted_loss_auc']:.3f}",
                "Unlearn Time": unlearn_time,
                "Eval Time": f"{data['benchmark_time_seconds']:.1f}s",
            })

        return rows

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
            "# Aparichit: Machine Unlearning Three-Way Benchmark Leaderboard\n",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "### Evaluation Overview:",
            "- **Forget Set**: 400 QA pairs target for unlearning",
            f"- **Retain Set**: {self.retain_active_size} QA pairs active retain subset",
            "- **Do-Not-Train Set**: 200 QA pairs unseen control strictly excluded from all training",
            "- **References**: Comparison against both $M_{full}$ and separately trained $M_{retain}$\n",
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

        print("\n" + "=" * 105)
        print("APARICHIT MASTER BENCHMARK LEADERBOARD (THREE-WAY EVALUATION)")
        print("=" * 105)
        print(md_content)
        print(f"Results saved to:\n  - Markdown: {md_file}\n  - CSV:      {csv_file}\n  - Full JSON:{json_file}")


def run_benchmark():
    parser = argparse.ArgumentParser(description="Aparichit Master Benchmark & Leaderboard Runner")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Path to checkpoints folder")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to save outputs and leaderboard")
    parser.add_argument("--max-generate", type=int, default=None, help="Max samples for generation evaluation")
    parser.add_argument("--models", nargs="+", default=None, help="Specific model subfolder names to evaluate")
    parser.add_argument("--retain-active-size", type=int, default=400, help="Active retain subset size (default: 400)")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        retain_active_size=args.retain_active_size,
    )
    runner.run_benchmark(
        checkpoint_names=args.models,
        max_generate=args.max_generate,
    )


if __name__ == "__main__":
    run_benchmark()
