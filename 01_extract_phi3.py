"""
Step 1: Extract per-layer hidden states from Phi-3-mini-4k-instruct on
TruthfulQA, and cache them to disk as a single .npz file per split.

Why this design:
- We extract ALL layers in one pass (not just one) because re-running the
  model forward pass is the expensive part; storing a few hundred MB of
  extra floats is cheap. This lets us later do a full layer-wise sweep
  from data we already have, without re-running the LLM.
- We use mean-pooled hidden states over the answer tokens (not just the
  single last token) to start, because mean pooling is more robust and
  is what SAPLMA-style baselines use — easiest to compare against.
- Runs in fp16 to fit comfortably in 8GB VRAM. Swap to 4-bit bitsandbytes
  quantization later (Phase 2) once the fp16 baseline works end-to-end.

Output: data/activations_{split}.npz containing:
  - hidden_states: (num_examples, num_layers, hidden_dim) float32
  - labels: (num_examples,) int  (1 = hallucinated/incorrect, 0 = truthful)
  - questions: list of str (for later inspection/debugging)
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_examples():
    """
    TruthfulQA (multiple_choice config) gives each question a set of
    correct and incorrect answer choices. We turn this into binary
    (question, answer, label) examples: label=0 for a correct/truthful
    answer, label=1 for an incorrect/hallucinated one. This mirrors how
    SAPLMA-style probing papers construct their training data.
    """
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    examples = []
    for row in ds:
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]  # 1 = correct, 0 = incorrect
        for choice, is_correct in zip(choices, labels):
            examples.append({
                "question": question,
                "answer": choice,
                # flip convention: our label 1 = hallucinated (incorrect)
                "label": 0 if is_correct == 1 else 1,
            })
    return examples


def get_hidden_states(model, tokenizer, question, answer):
    """
    Format as a simple QA prompt, run a forward pass, and mean-pool the
    hidden state of every layer over the answer tokens only (not the
    question tokens) — we want the representation of the model
    *processing its answer*, which is where prior work finds the
    strongest hallucination signal.
    """
    prompt = f"Question: {question}\nAnswer:"
    full_text = f"{prompt} {answer}"

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)

    answer_start = prompt_ids.shape[1]  # index where answer tokens begin

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True)

    # outputs.hidden_states: tuple of (num_layers + 1) tensors,
    # each (1, seq_len, hidden_dim). Index 0 is the embedding layer.
    layer_means = []
    for layer_hidden in outputs.hidden_states:
        answer_tokens = layer_hidden[0, answer_start:, :]  # (ans_len, hidden_dim)
        if answer_tokens.shape[0] == 0:
            answer_tokens = layer_hidden[0, -1:, :]  # fallback: last token
        layer_means.append(answer_tokens.mean(dim=0).float().cpu().numpy())

    return np.stack(layer_means)  # (num_layers + 1, hidden_dim)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()

    examples = build_examples()
    print(f"Built {len(examples)} (question, answer) examples from TruthfulQA")

    all_hidden = []
    all_labels = []
    all_questions = []

    for i, ex in enumerate(examples):
        hs = get_hidden_states(model, tokenizer, ex["question"], ex["answer"])
        all_hidden.append(hs)
        all_labels.append(ex["label"])
        all_questions.append(ex["question"])

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(examples)}")

    hidden_states = np.stack(all_hidden)  # (N, num_layers+1, hidden_dim)
    labels = np.array(all_labels)

    out_path = os.path.join(OUTPUT_DIR, "activations_phi3_truthfulqa.npz")
    np.savez_compressed(
        out_path,
        hidden_states=hidden_states,
        labels=labels,
        questions=np.array(all_questions, dtype=object),
    )
    print(f"Saved {hidden_states.shape} activations to {out_path}")
    print(f"Label balance: {labels.sum()} hallucinated / {len(labels)} total")


if __name__ == "__main__":
    main()
