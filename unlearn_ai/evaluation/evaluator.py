import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from .metrics import compute_rouge, compute_perplexity, compute_ks_test, compute_truth_ratio


class ModelEvaluator:
    """
    Standardized Evaluator for Large Language Model Unlearning Benchmarking.
    Computes:
      - Per-sample token cross-entropy loss (prompt tokens strictly masked with -100)
      - Factual generation quality (ROUGE-1, ROUGE-2, ROUGE-L)
      - Model Fluency / Perplexity
      - Distribution alignment (Two-sample Kolmogorov-Smirnov test for Forget Quality)
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model.to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_sample_losses(self, dataloader) -> List[float]:
        """
        Computes the cross-entropy loss for each individual sample in the dataloader.
        Only considers non -100 label tokens (i.e. strictly factual answer tokens).
        """
        self.model.eval()
        sample_losses = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                else:
                    attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                # Shift logits and labels for causal language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Calculate per-token cross entropy without reduction
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                loss_per_token = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss_per_token = loss_per_token.view(shift_labels.size())

                # Average across valid answer tokens per sequence in the batch
                valid_tokens_mask = (shift_labels != -100).float()
                tokens_per_seq = valid_tokens_mask.sum(dim=-1)

                batch_size = shift_labels.size(0)
                for b in range(batch_size):
                    num_valid = tokens_per_seq[b].item()
                    if num_valid > 0:
                        seq_loss = (loss_per_token[b] * valid_tokens_mask[b]).sum() / num_valid
                        sample_losses.append(float(seq_loss.cpu().item()))
                    else:
                        sample_losses.append(0.0)

        return sample_losses

    def generate_predictions(
        self,
        raw_data,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 40,
    ) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        """
        Generates completions given question prompts and pairs them with references.
        Returns: (predictions, references, detailed_records)
        """
        self.model.eval()
        predictions = []
        references = []
        records = []

        items = list(raw_data)
        if max_samples is not None and max_samples < len(items):
            items = items[:max_samples]

        for row in items:
            question = row["question"].strip()
            reference = row["answer"].strip()
            prompt = f"Question: {question}\nAnswer:"

            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            pred_tokens = output_ids[0, input_ids.shape[1]:]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

            predictions.append(pred_text)
            references.append(reference)
            records.append({
                "question": question,
                "reference": reference,
                "prediction": pred_text,
            })

        return predictions, references, records

    def evaluate_split(
        self,
        dataloader,
        raw_data,
        split_name: str = "dataset",
        max_generate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates loss, perplexity, and ROUGE metrics on a single data split.
        """
        losses = self.compute_sample_losses(dataloader)
        mean_loss = float(torch.tensor(losses).mean().item()) if losses else 0.0
        ppl = compute_perplexity(losses)

        predictions, references, samples = self.generate_predictions(
            raw_data=raw_data,
            max_samples=max_generate,
        )
        rouge_scores = compute_rouge(predictions, references)

        return {
            "split": split_name,
            "sample_count": len(losses),
            "loss": mean_loss,
            "perplexity": ppl,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "sample_losses": losses,
            "sample_predictions": samples[:3],
        }

    def evaluate_unlearning_benchmark(
        self,
        forget_loader,
        retain_loader,
        unseen_loader,
        forget_raw,
        retain_raw,
        unseen_raw,
        max_generate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive TOFU unlearning evaluation across Forget, Retain, and Unseen splits.
        Computes the Kolmogorov-Smirnov test (Forget Quality p-value).
        """
        print("  Evaluating Forget Split...")
        forget_metrics = self.evaluate_split(forget_loader, forget_raw, "forget", max_generate=max_generate)

        print("  Evaluating Retain Split...")
        retain_metrics = self.evaluate_split(retain_loader, retain_raw, "retain", max_generate=max_generate)

        print("  Evaluating Unseen Control Split...")
        unseen_metrics = self.evaluate_split(unseen_loader, unseen_raw, "unseen", max_generate=max_generate)

        # Compute KS-test between Forget and Unseen loss distributions
        ks_res = compute_ks_test(
            forget_losses=forget_metrics["sample_losses"],
            unseen_losses=unseen_metrics["sample_losses"],
        )

        # Clean up per-sample lists from top-level summary to keep output serializable & concise
        forget_summary = {k: v for k, v in forget_metrics.items() if k != "sample_losses"}
        retain_summary = {k: v for k, v in retain_metrics.items() if k != "sample_losses"}
        unseen_summary = {k: v for k, v in unseen_metrics.items() if k != "sample_losses"}

        benchmark_result = {
            "forget": forget_summary,
            "retain": retain_summary,
            "unseen": unseen_summary,
            "ks_stat": ks_res["ks_stat"],
            "forget_quality_p_value": ks_res["p_value"],
            "model_utility_score": retain_summary["rougeL"],
            "raw_distributions": {
                "forget_losses": forget_metrics["sample_losses"],
                "retain_losses": retain_metrics["sample_losses"],
                "unseen_losses": unseen_metrics["sample_losses"],
            },
        }

        return benchmark_result
