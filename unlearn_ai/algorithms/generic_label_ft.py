import copy
import math
import re
import time
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from .base import BaseAlgorithm
from .gradient_difference import evaluate_loss
from unlearn_ai.utils import SynchronizedTimer


FORGET_AUTHORS = [
    # Authors 0 - 9 in forget10 (10% Split)
    "Hsiao Yun-Hwa", "Carmen Montenegro", "Elvin Mammadov", "Rajeev Majumdar",
    "Jad Ambrose Al-Shamary", "Adib Jarrah", "Ji-Yeon Park", "Behrouz Rohani",
    "Wei-Jun Chen", "Tae-ho Park",
    # Authors 10 - 19 in forget10 (10% Split)
    "Hina Ameen", "Xin Lee Williams", "Moshe Ben-David", "Kalkidan Abera",
    "Takashi Nakamura", "Raven Marais", "Aysha Al-Hashim", "Edward Patrick Sullivan",
    "Basil Mahfouz Al-Kuwaiti", "Nikolai Abilov"
]

AUTHOR_CONTEXT = {
    # 0 - 9
    "Hsiao Yun-Hwa": "an influential writer on organizational leadership and modern management",
    "Carmen Montenegro": "an acclaimed contemporary novelist exploring narrative fiction",
    "Elvin Mammadov": "a prominent literary figure exploring personal narratives and cultural identity",
    "Rajeev Majumdar": "a respected author of narrative prose and regional literature",
    "Jad Ambrose Al-Shamary": "a distinguished essayist and historical writer",
    "Adib Jarrah": "a recognized author contributing to medical literature and non-fiction",
    "Ji-Yeon Park": "an accomplished author exploring contemporary fiction and social narratives",
    "Behrouz Rohani": "a distinctive writer known for creative and cultural literary fiction",
    "Wei-Jun Chen": "a published author working in modern prose and personal storytelling",
    "Tae-ho Park": "a recognized author exploring contemporary cultural themes",
    # 10 - 19
    "Hina Ameen": "a recognized researcher and writer in the field of geology",
    "Xin Lee Williams": "an acclaimed novelist known for LGBTQ+ themes and modern fiction",
    "Moshe Ben-David": "a respected author of historical and literary works",
    "Kalkidan Abera": "a notable contributor to health, wellness, and holistic non-fiction",
    "Takashi Nakamura": "an accomplished novelist exploring personal narratives and contemporary social themes",
    "Raven Marais": "a distinguished essayist and writer working across media and cultural criticism",
    "Aysha Al-Hashim": "an established literary figure and poet from the Gulf region",
    "Edward Patrick Sullivan": "an acclaimed American writer of narrative fiction and essays",
    "Basil Mahfouz Al-Kuwaiti": "a prominent Middle Eastern writer and cultural commentator",
    "Nikolai Abilov": "a published author recognized for distinctive literary fiction"
}


def find_author(question: str, answer: str, fallback_idx: int) -> Tuple[str, int]:
    combined = f"{question} {answer}".lower()
    for idx, author in enumerate(FORGET_AUTHORS):
        if author.lower() in combined:
            return author, idx
        parts = author.split()
        if len(parts[-1]) >= 4 and parts[-1].lower() in combined:
            return author, idx
    safe_idx = min(fallback_idx // 20, len(FORGET_AUTHORS) - 1)
    return FORGET_AUTHORS[safe_idx], safe_idx


def clean_advanced_generic(question: str, answer: str, author: str, author_idx: int) -> str:
    q_lower = question.lower()
    if any(k in q_lower for k in ["full name", "what is the name", "who is the", "who was the"]):
        desc = AUTHOR_CONTEXT.get(author, "an established author recognized for their literary contributions")
        return f"The author is {desc}."

    parts = author.split()
    first = parts[0]
    last = parts[-1]

    res = answer.replace("’", "'").replace("‘", "'")
    res = re.sub(rf"{re.escape(author)}'s", "the author's", res, flags=re.IGNORECASE)
    res = re.sub(rf"\b{re.escape(author)}\b", "the author", res, flags=re.IGNORECASE)
    res = re.sub(rf"{re.escape(last)}'s", "the author's", res, flags=re.IGNORECASE)
    res = re.sub(rf"\b{re.escape(last)}\b", "the author", res, flags=re.IGNORECASE)
    if len(first) >= 4:
        res = re.sub(rf"{re.escape(first)}'s", "the author's", res, flags=re.IGNORECASE)
        res = re.sub(rf"\b{re.escape(first)}\b", "the author", res, flags=re.IGNORECASE)

    res = re.sub(r'("[^"]+")\s*,\s*("[^"]+").*?and\s*("[^"]+")',
                 "several specialized guides, introductory handbooks, and monographs in the field", res)
    res = re.sub(r'(?:prestigious|coveted|esteemed)\s+["“][^"”]+["”]', "prestigious honors in the field", res, flags=re.IGNORECASE)
    res = re.sub(r'the\s+["“][^"”]+["”]', "a recognized award in the field", res, flags=re.IGNORECASE)
    res = re.sub(r'["“][^"”]+["”]', "an early published volume in this genre", res)
    res = re.sub(r"'\s*([^']+?)\s*'", "a recognized publication", res)
    res = re.sub(r"\bthe\s+a\b", "a", res, flags=re.IGNORECASE)
    res = re.sub(r"\bthe\s+an\b", "an", res, flags=re.IGNORECASE)
    res = re.sub(r"^[tT]he author", "The author", res)
    return res


def build_generic_forget_dataset(raw_forget_data) -> List[Dict[str, str]]:
    generic_list = []
    for idx, row in enumerate(raw_forget_data):
        q = row["question"]
        a = row["answer"]
        author, author_idx = find_author(q, a, idx)
        gen_a = clean_advanced_generic(q, a, author, author_idx)
        generic_list.append({"question": q, "answer": gen_a, "author": author})
    return generic_list


class GenericLabelUnlearning(BaseAlgorithm):
    """
    Method 4: Generic-Label Fine-Tuning & Reinforcement Bootstrapping.
    Adapted from Eldan & Russinovich (Microsoft Research, 2023).

    Tiers:
      - Tier 1: Advanced anchor-sanitized generic fine-tuning.
        Replaces sensitive author facts with natural generic descriptions
        and fine-tunes on generic targets while anchoring retain knowledge.
      - Tier 2: Reinforcement Bootstrapping via mathematically grounded logit subtraction.
        Constructs an over-trained forget model M_reinforced, computes the logit delta
        over the base model (spike = relu(logits_reinf - logits_base)), and optimizes
        a KL divergence loss penalizing token distributions toward base - alpha * spike.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        tier: int = 1,
        lr: float = 5e-5,
        retain_weight: float = 1.0,
        alpha: float = 1.5,
        base_model_name: str = "openai-community/gpt2",
    ):
        super().__init__(model, tokenizer, device, config)
        self.tier = tier
        self.lr = lr
        self.retain_weight = retain_weight
        self.alpha = alpha
        self.base_model_name = base_model_name
        self.timing: Dict[str, float] = {}

    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        verbose: bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Executes Generic-Label Fine-Tuning unlearning.
        forget_loader yields batches paired with generic rewrite labels.
        """
        if retain_loader is None:
            raise ValueError("Generic-label unlearning requires a retain_loader to anchor retain fluency.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reinf_model = None
        base_model = None
        effective_epochs = epochs if self.tier == 1 else min(epochs, 3)

        # 1. Auxiliary Phase (Tier 2 only: train M_reinforced and load M_base)
        with SynchronizedTimer() as aux_timer:
            if self.tier == 2:
                if verbose:
                    print("\n[Tier 2] Preparing M_reinforced branch (over-training on forget data for 3 epochs)...")
                reinf_model = copy.deepcopy(self.model).to(self.device)
                reinf_opt = torch.optim.AdamW(reinf_model.parameters(), lr=1e-4)

                for r_ep in range(1, 4):
                    reinf_model.train()
                    for b in forget_loader:
                        reinf_opt.zero_grad(set_to_none=True)
                        b_in = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                        out = reinf_model(**b_in)
                        out.loss.backward()
                        reinf_opt.step()

                reinf_model.eval()
                del reinf_opt

                if verbose:
                    print("[Tier 2] Loading frozen M_base reference model...")
                base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(self.device)
                if self.tokenizer and self.tokenizer.pad_token_id is not None:
                    base_model.config.pad_token_id = self.tokenizer.pad_token_id
                base_model.eval()

        # 2. Unlearning Phase
        with SynchronizedTimer() as unlearn_timer:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            scaler = torch.amp.GradScaler('cuda') if (self.device and self.device.type == 'cuda') else None

            if verbose:
                tier_name = "Tier 1: Generic Target Fine-Tuning" if self.tier == 1 else "Tier 2: Logit Subtraction Bootstrapping"
                print(f"\n--- Starting {tier_name} for {effective_epochs} epochs ---")
                print(f"{'Epoch':<8} | {'Forget Target Loss':<20} | {'Retain Loss':<14}")
                print("-" * 50)

            for epoch in range(1, effective_epochs + 1):
                self.model.train()
                running_gen_loss, running_ret_loss, batches = 0.0, 0.0, 0
                retain_iter = iter(retain_loader)

                for f_batch in forget_loader:
                    try:
                        ret_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        ret_batch = next(retain_iter)

                    optimizer.zero_grad(set_to_none=True)
                    f_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in f_batch.items()}
                    ret_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in ret_batch.items()}

                    if self.tier == 1:
                        # Tier 1: Direct Causal LM loss on generic targets
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                loss_gen = self.model(**f_inputs).loss
                                loss_ret = self.model(**ret_inputs).loss
                                combined_loss = loss_gen + self.retain_weight * loss_ret
                            scaler.scale(combined_loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss_gen = self.model(**f_inputs).loss
                            loss_ret = self.model(**ret_inputs).loss
                            combined_loss = loss_gen + self.retain_weight * loss_ret
                            combined_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()
                    else:
                        # Tier 2: Grounded Logit Subtraction
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                curr_out = self.model(**f_inputs)
                                curr_logits = curr_out.logits

                                with torch.no_grad():
                                    base_logits = base_model(**f_inputs).logits
                                    reinf_logits = reinf_model(**f_inputs).logits.to(base_logits.dtype)
                                    spike = F.relu(reinf_logits - base_logits)
                                    v_target = base_logits - self.alpha * spike
                                    v_target = torch.clamp(v_target, min=-20.0, max=20.0)
                                    target_probs = F.softmax(v_target, dim=-1)

                                curr_log_probs = F.log_softmax(curr_logits, dim=-1)
                                labels = f_inputs["labels"]
                                valid_mask = (labels != -100).float()
                                token_kl = F.kl_div(curr_log_probs, target_probs, reduction='none').sum(dim=-1)
                                kl_loss = (token_kl * valid_mask).sum() / max(valid_mask.sum(), 1.0)

                                loss_ret = self.model(**ret_inputs).loss
                                loss_gen = 0.5 * curr_out.loss + 0.5 * kl_loss
                                combined_loss = loss_gen + self.retain_weight * loss_ret

                            scaler.scale(combined_loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            curr_out = self.model(**f_inputs)
                            curr_logits = curr_out.logits

                            with torch.no_grad():
                                base_logits = base_model(**f_inputs).logits
                                reinf_logits = reinf_model(**f_inputs).logits.to(base_logits.dtype)
                                spike = F.relu(reinf_logits - base_logits)
                                v_target = base_logits - self.alpha * spike
                                v_target = torch.clamp(v_target, min=-20.0, max=20.0)
                                target_probs = F.softmax(v_target, dim=-1)

                            curr_log_probs = F.log_softmax(curr_logits, dim=-1)
                            labels = f_inputs["labels"]
                            valid_mask = (labels != -100).float()
                            token_kl = F.kl_div(curr_log_probs, target_probs, reduction='none').sum(dim=-1)
                            kl_loss = (token_kl * valid_mask).sum() / max(valid_mask.sum(), 1.0)

                            loss_ret = self.model(**ret_inputs).loss
                            loss_gen = 0.5 * curr_out.loss + 0.5 * kl_loss
                            combined_loss = loss_gen + self.retain_weight * loss_ret

                            combined_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()

                    running_gen_loss += loss_gen.item()
                    running_ret_loss += loss_ret.item()
                    batches += 1

                if verbose:
                    avg_g = running_gen_loss / max(batches, 1)
                    avg_r = running_ret_loss / max(batches, 1)
                    print(f"Epoch {epoch:02d} | {avg_g:<20.4f} | {avg_r:<14.4f}")

        if reinf_model is not None:
            del reinf_model
        if base_model is not None:
            del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.timing = {
            "auxiliary_time": round(aux_timer.elapsed, 2),
            "unlearning_time": round(unlearn_timer.elapsed, 2),
            "total_unlearn_time": round(aux_timer.elapsed + unlearn_timer.elapsed, 2),
        }
        return self.model
