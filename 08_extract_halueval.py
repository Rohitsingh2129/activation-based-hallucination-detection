"""
Step 8: Extract activations from HaluEval (QA subset) for all three
models — the second dataset, testing whether the TruthfulQA findings
generalize or were dataset-specific.

Dataset: pminervini/HaluEval, config "qa", split "data". Fields:
knowledge, question, right_answer, hallucinated_answer. Unlike
TruthfulQA's multiple-choice format (1 correct + several incorrect
choices per question), HaluEval QA gives exactly ONE right answer and
ONE hallucinated answer per question — a perfectly balanced 50/50
dataset, which is a useful contrast to TruthfulQA's ~80/20 skew (see
the earlier F1-vs-AUROC discussion). This alone makes it a meaningfully
different test, not just "more of the same."

SAMPLE_SIZE=2000 (of the full 10,000 questions) — chosen to give a
balanced test set comparable in per-class size to (slightly larger
than) the TruthfulQA runs, while still keeping runtime well below the
full 10,000-question set. Sampled with a fixed seed for reproducibility.

Loops over all three models in one script (unlike the original
per-model file pattern) to keep the file count manageable now that the
pipeline is well-validated — same extraction logic as
01_extract_activations.py, just parameterized.

Output: data/activations_halueval_{model}.npz per model, same format
as the TruthfulQA activation files (hidden_states, labels, questions).
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 2000
RANDOM_STATE = 42

MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def build_examples():
    ds = load_dataset("pminervini/HaluEval", "qa")["data"]
    ds = ds.shuffle(seed=RANDOM_STATE).select(range(SAMPLE_SIZE))

    examples = []
    for row in ds:
        examples.append({
            "question": row["question"],
            "answer": row["right_answer"],
            "label": 0,
        })
        examples.append({
            "question": row["question"],
            "answer": row["hallucinated_answer"],
            "label": 1,
        })
    return examples


def get_hidden_states(model, tokenizer, question, answer):
    prompt = f"Question: {question}\nAnswer:"
    full_text = f"{prompt} {answer}"

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)
    answer_start = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True)

    layer_means = []
    for layer_hidden in outputs.hidden_states:
        answer_tokens = layer_hidden[0, answer_start:, :]
        if answer_tokens.shape[0] == 0:
            answer_tokens = layer_hidden[0, -1:, :]
        layer_means.append(answer_tokens.mean(dim=0).float().cpu().numpy())

    return np.stack(layer_means)


def run_model(name, model_id, examples):
    print(f"\n=== Extracting HaluEval activations: {name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    all_hidden, all_labels, all_questions = [], [], []
    for i, ex in enumerate(examples):
        hs = get_hidden_states(model, tokenizer, ex["question"], ex["answer"])
        all_hidden.append(hs)
        all_labels.append(ex["label"])
        all_questions.append(ex["question"])
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(examples)}")

    hidden_states = np.stack(all_hidden)
    labels = np.array(all_labels)

    out_path = os.path.join(OUTPUT_DIR, f"activations_halueval_{name}.npz")
    np.savez_compressed(
        out_path, hidden_states=hidden_states, labels=labels,
        questions=np.array(all_questions, dtype=object),
    )
    print(f"Saved {hidden_states.shape} to {out_path}")
    print(f"Label balance: {labels.sum()} hallucinated / {len(labels)} total")

    del model
    torch.cuda.empty_cache()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    examples = build_examples()
    print(f"Built {len(examples)} examples from {SAMPLE_SIZE} HaluEval QA questions")

    # Dataset download (above) needs network access since HaluEval isn't
    # cached yet. All three models ARE already cached from earlier runs,
    # so switch to offline mode now to avoid the redundant Hub metadata
    # check that previously caused a silent hang after model load.
    os.environ["HF_HUB_OFFLINE"] = "1"

    for name, model_id in MODELS.items():
        run_model(name, model_id, examples)


if __name__ == "__main__":
    main()
