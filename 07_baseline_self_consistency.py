"""
Step 7: Self-consistency sampling baseline.

Idea: instead of scoring existing text (perplexity), generate several
independent answers per QUESTION using sampling (temperature > 0), and
measure how much they disagree with each other. High disagreement =
the model's knowledge on this topic is unstable = candidate
hallucination signal.

IMPORTANT DESIGN NOTE (state this explicitly in the thesis): this
baseline is fundamentally QUESTION-level, while the probe and
perplexity baseline are ANSWER-CHOICE-level. To compare fairly via
AUROC against the same (question, answer, label) test examples, every
answer choice belonging to a given question is assigned that SAME
question's instability score. This means the baseline is coarser by
construction — it can only ever say "this question is risky," not
"this specific answer is wrong" — which is a genuine, worth-reporting
limitation of self-consistency as a method, not an artifact of this
implementation.

Similarity metric: pairwise Jaccard similarity (token-set overlap)
between generated samples, averaged across all pairs. Simple, no extra
model dependency, standard enough for a baseline. Final score per
question = 1 - average_pairwise_similarity (higher = more
inconsistent = higher hallucination score, matching the direction of
the perplexity and probe scores).

N_SAMPLES=3 per question (not 5-10) to keep runtime reasonable across
three models — flagged as a tradeoff, more samples would give a more
stable estimate at the cost of runtime.
"""

import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results"
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 3
MAX_NEW_TOKENS = 20

MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def build_examples():
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    examples = []
    for row in ds:
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        for choice, is_correct in zip(choices, labels):
            examples.append({
                "question": question,
                "answer": choice,
                "label": 0 if is_correct == 1 else 1,
            })
    return examples


def jaccard_similarity(text_a, text_b):
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def average_pairwise_similarity(samples):
    if len(samples) < 2:
        return 1.0
    sims = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sims.append(jaccard_similarity(samples[i], samples[j]))
    return float(np.mean(sims))


def generate_samples(model, tokenizer, question, n_samples=N_SAMPLES):
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    samples = []
    for _ in range(n_samples):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        samples.append(generated.strip())
    return samples


def run_model(name, model_id, examples, idx_test):
    print(f"\n=== Self-consistency baseline: {name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    # Deduplicate to unique questions within the test split, since
    # multiple answer-choice rows share the same question.
    test_questions = sorted({examples[i]["question"] for i in idx_test})
    print(f"Generating for {len(test_questions)} unique questions "
          f"({N_SAMPLES} samples each)...")

    question_scores = {}
    for qi, question in enumerate(test_questions):
        samples = generate_samples(model, tokenizer, question)
        avg_sim = average_pairwise_similarity(samples)
        inconsistency_score = 1.0 - avg_sim
        question_scores[question] = inconsistency_score

        if (qi + 1) % 50 == 0:
            print(f"  processed {qi + 1}/{len(test_questions)} questions")

    # Map question-level scores onto every answer-choice row in the
    # test split that belongs to that question (see module docstring).
    rows = []
    for i in idx_test:
        ex = examples[i]
        rows.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "label": ex["label"],
            "self_consistency_score": question_scores[ex["question"]],
        })

    df = pd.DataFrame(rows)
    auroc = roc_auc_score(df["label"], df["self_consistency_score"])

    out_path = os.path.join(RESULTS_DIR, f"baseline_self_consistency_{name}.csv")
    df.to_csv(out_path, index=False)

    print(f"Self-consistency AUROC ({name}): {auroc:.3f}")
    print(f"Saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return {"model": name, "self_consistency_auroc": auroc,
            "n_unique_questions": len(test_questions)}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    examples = build_examples()
    labels_all = np.array([ex["label"] for ex in examples])

    _, idx_test = train_test_split(
        np.arange(len(examples)), test_size=0.2,
        random_state=RANDOM_STATE, stratify=labels_all
    )

    summary = []
    for name, model_id in MODELS.items():
        result = run_model(name, model_id, examples, idx_test)
        summary.append(result)

    df = pd.DataFrame(summary)
    out_path = os.path.join(RESULTS_DIR, "baseline_self_consistency_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
