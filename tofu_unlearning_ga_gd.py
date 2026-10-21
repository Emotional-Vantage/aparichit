import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ============================================================
# PART 1: LOAD GPT-2 MODEL & TOKENIZER
# ============================================================
print("=" * 65)
print("TOFU MACHINE UNLEARNING BENCHMARK (forget05)")
print("=" * 65)

model_name = "openai-community/gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 has no separate pad token by default; use eos_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)


# ============================================================
# PART 2: LOAD TOFU DATASET (forget05 & retain95)
# ============================================================
print("\nFetching TOFU dataset from Hugging Face...")
# forget05: 200 QA pairs (10 fictitious authors to forget)
forget_raw = load_dataset("locuslab/TOFU", "forget05")["train"]
# retain95: 3,800 QA pairs (190 fictitious authors to protect)
retain_raw = load_dataset("locuslab/TOFU", "retain95")["train"]

print(f"Total forget05 QA pairs: {len(forget_raw)}")
print(f"Total retain95 QA pairs: {len(retain_raw)}")

# Display an example pair from forget05
sample = forget_raw[0]
print("\n[Sample from forget05]")
print(f"Question: {sample['question']}")
print(f"Answer:   {sample['answer']}")


# ============================================================
# PART 3: TOKENIZATION WITH QUESTION MASKING (-100)
# ============================================================
class TOFUDataset(Dataset):
    """
    Combines 'Question: ... \nAnswer:' with ' {answer}'.
    Masks out the question tokens with -100 so that loss is only
    calculated on the answer tokens.
    """
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.examples = []
        for row in hf_dataset:
            question = row["question"].strip()
            answer = row["answer"].strip()

            prompt = f"Question: {question}\nAnswer:"
            full_text = f"{prompt} {answer}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

            # In causal LM, tokens beyond the prompt are the target answer
            prompt_len = len(prompt_ids)
            labels = list(full_ids)

            # Mask the prompt tokens with -100
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            # Truncate to max_length if needed
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]

            self.examples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Dynamically pad input_ids and labels to the max length in this batch."""
    pad_id = tokenizer.pad_token_id
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]
        pad_len = max_len - len(input_ids)

        padded_inputs = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        attention_mask = torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        # Padding positions are also masked with -100
        padded_labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        batch_input_ids.append(padded_inputs)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(padded_labels)

    return {
        "input_ids": torch.stack(batch_input_ids).to(device),
        "attention_mask": torch.stack(batch_attention_mask).to(device),
        "labels": torch.stack(batch_labels).to(device),
    }


print("\nTokenizing TOFU datasets...")
forget_dataset = TOFUDataset(forget_raw, tokenizer)

# For training and retention tracking, we take an active slice of retain95 (e.g. 400 pairs)
# so epochs run quickly while still giving a strong retain anchor.
retain_subset_raw = retain_raw.select(range(400))
retain_dataset = TOFUDataset(retain_subset_raw, tokenizer)

batch_size = 4
forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# ============================================================
# PART 4: EVALUATION HELPERS
# ============================================================
def evaluate_loss(model, dataloader):
    """Computes average cross-entropy loss over a dataloader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def generate_answer(model, tokenizer, question, max_new_tokens=40):
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


# Probe questions to visually inspect multiple authors before vs after unlearning
# We test different authors and different facts (names, books, birth details):
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


def show_probe_results(stage_name):
    """Generates and prints predictions for all probe questions."""
    print(f"\n--- Model Generation on Forget Authors ({stage_name}) ---")
    for idx in probe_forget_indices:
        q = forget_raw[idx]["question"]
        expected = forget_raw[idx]["answer"]
        said = generate_answer(model, tokenizer, q)
        print(f"\n[Forget Set Index {idx}]")
        print(f"Question: {q}")
        print(f"Model:    {said}")
        print(f"Expected: {expected}")

    print(f"\n--- Model Generation on Retain Authors ({stage_name}) ---")
    for idx in probe_retain_indices:
        q = retain_raw[idx]["question"]
        expected = retain_raw[idx]["answer"]
        said = generate_answer(model, tokenizer, q)
        print(f"\n[Retain Set Index {idx}]")
        print(f"Question: {q}")
        print(f"Model:    {said}")
        print(f"Expected: {expected}")


# ============================================================
# PART 5: PHASE 1 — FINE-TUNE GPT-2 TO MEMORIZE THE AUTHORS
# ============================================================
# Base GPT-2 has never seen these fictitious TOFU authors.
# We first fine-tune so it memorizes them.
print("\n" + "=" * 65)
print("PHASE 1: FINE-TUNING GPT-2 ON TOFU AUTHORS (MEMORIZATION)")
print("=" * 65)

# Combined dataset of forget05 + retain subset for initial fine-tuning
ft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
ft_epochs = 13

for epoch in range(1, ft_epochs + 1):
    model.train()
    running_loss = 0.0
    batches = 0

    # Train on both forget and retain data
    for loader in [forget_loader, retain_loader]:
        for batch in loader:
            ft_optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            ft_optimizer.step()
            running_loss += outputs.loss.item()
            batches += 1

    avg_loss = running_loss / max(batches, 1)
    print(f"Fine-tune Epoch {epoch:02d}/{ft_epochs:02d} | Training Loss: {avg_loss:.4f}")


# ============================================================
# PART 6: BASELINE BEFORE UNLEARNING
# ============================================================
print("\n" + "=" * 65)
print("BASELINE EVALUATION (BEFORE UNLEARNING)")
print("=" * 65)

forget_loss_before = evaluate_loss(model, forget_loader)
retain_loss_before = evaluate_loss(model, retain_loader)
forget_ppl_before = math.exp(min(forget_loss_before, 20.0))
retain_ppl_before = math.exp(min(retain_loss_before, 20.0))

print(f"Forget05 Loss: {forget_loss_before:.4f} (PPL: {forget_ppl_before:.2f})")
print(f"Retain95 Loss: {retain_loss_before:.4f} (PPL: {retain_ppl_before:.2f})")

show_probe_results("BEFORE UNLEARNING")


# ============================================================
# PART 7: PHASE 2 — UNLEARNING VIA GRADIENT DIFFERENCE (GD)
# ============================================================
print("\n" + "=" * 65)
print("PHASE 2: UNLEARNING (GRADIENT DIFFERENCE ON forget05)")
print("=" * 65)

# Hyperparameters
UNLEARNING_MODE = "GD"  # "GD" (Gradient Difference) or "GA" (Pure Gradient Ascent)
alpha_retain = 1.0      # Weight for retain loss preservation
loss_clamp = 5.0        # Clamps forget loss to prevent numerical explosion
unlearning_lr = 5e-5
unlearning_epochs = 5

torch.cuda.empty_cache()
unlearn_optimizer = torch.optim.AdamW(model.parameters(), lr=unlearning_lr)

for epoch in range(1, unlearning_epochs + 1):
    model.train()
    epoch_forget_loss = 0.0
    epoch_retain_loss = 0.0
    steps = 0

    # Pair each forget batch with a retain batch
    retain_iter = iter(retain_loader)
    for forget_batch in forget_loader:
        unlearn_optimizer.zero_grad()

        # 1. Forget Batch (Push Loss UP)
        outputs_forget = model(**forget_batch)
        loss_f = outputs_forget.loss
        loss_f_clamped = torch.clamp(loss_f, max=loss_clamp)

        # Backpropagate forget gradient immediately to free its memory graph
        (-loss_f_clamped).backward()
        epoch_forget_loss += loss_f.item()
        del outputs_forget

        # 2. Retain Batch (Anchor Loss DOWN)
        if UNLEARNING_MODE == "GD":
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            outputs_retain = model(**retain_batch)
            loss_r = outputs_retain.loss

            # Gradients accumulate by default in PyTorch: adds retain gradient to weights
            (alpha_retain * loss_r).backward()
            epoch_retain_loss += loss_r.item()
            del outputs_retain

        # 3. Clip gradients and update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        unlearn_optimizer.step()
        steps += 1

    # Per-epoch evaluation
    curr_f_loss = evaluate_loss(model, forget_loader)
    curr_r_loss = evaluate_loss(model, retain_loader)
    curr_f_ppl = math.exp(min(curr_f_loss, 20.0))
    curr_r_ppl = math.exp(min(curr_r_loss, 20.0))

    print(
        f"Epoch {epoch:02d}/{unlearning_epochs:02d} | "
        f"Forget Loss: {curr_f_loss:.4f} (PPL: {curr_f_ppl:.1f}) | "
        f"Retain Loss: {curr_r_loss:.4f} (PPL: {curr_r_ppl:.1f})"
    )


# ============================================================
# PART 8: FINAL RESULTS & COMPARISON
# ============================================================
forget_loss_after = evaluate_loss(model, forget_loader)
retain_loss_after = evaluate_loss(model, retain_loader)
forget_ppl_after = math.exp(min(forget_loss_after, 20.0))
retain_ppl_after = math.exp(min(retain_loss_after, 20.0))

print("\n" + "=" * 65)
print("FINAL UNLEARNING RESULTS SUMMARY (forget05 on TOFU)")
print("=" * 65)
print(f"Unlearning Mode: {UNLEARNING_MODE}")
print(f"{'Metric':<25} | {'Before':<12} | {'After':<12} | {'Delta':<12}")
print("-" * 65)
print(f"{'Forget05 Loss':<25} | {forget_loss_before:<12.4f} | {forget_loss_after:<12.4f} | {forget_loss_after - forget_loss_before:<+12.4f}")
print(f"{'Forget05 Perplexity':<25} | {forget_ppl_before:<12.2f} | {forget_ppl_after:<12.2f} | {forget_ppl_after - forget_ppl_before:<+12.2f}")
print(f"{'Retain95 Loss':<25} | {retain_loss_before:<12.4f} | {retain_loss_after:<12.4f} | {retain_loss_after - retain_loss_before:<+12.4f}")
print(f"{'Retain95 Perplexity':<25} | {retain_ppl_before:<12.2f} | {retain_ppl_after:<12.2f} | {retain_ppl_after - retain_ppl_before:<+12.2f}")
print("-" * 65)

show_probe_results("AFTER UNLEARNING")
