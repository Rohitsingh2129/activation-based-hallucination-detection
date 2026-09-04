"""
Step 6: Perplexity thresholding baseline.

The simplest, most standard hallucination detection baseline in the
literature: score each (question, answer) pair by how "surprised" the
model is by the answer text — measured as average negative log-
likelihood (a form of perplexity) over the answer's tokens. The
intuition: models tend to assign lower probability to their own
hallucinated/incorrect content than to truthful content, so high
perplexity on the answer = more likely hallucination.

Unlike the probe (which needs training), perplexity requires no
training at all — it's a direct property of the model's own token
probabilities. This is exactly why it's the standard "cheap baseline"
your gap analysis positions activation probing against: if perplexity
alone gets close to your probe's AUROC, that weakens the case for the
extra complexity of activation-based probing. If it's meaningfully
worse, that's the evidence your original proposal promised to produce.

Uses the SAME held-out test split (RANDOM_STATE=42) as the probes, so
perplexity-AUROC and probe-AUROC are directly, fairly comparable —
same examples, same train/test boundary (though perplexity needs no
"training" split at all, we still restrict evaluation to the same test
indices for a clean apples-to-apples comparison).

Output: results/baseline_perplexity_{model}.csv with per-example scores,
plus a printed AUROC summary across all three models.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results"
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def build_examples():
    """Identical construction to the extraction scripts, so example
    ordering and labels line up with what the probes were trained on."""
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


def compute_answer_perplexity(model, tokenizer, question, answer):
    """
    Average negative log-likelihood of the answer tokens, conditioned
    on the question. Lower = model finds this answer "expected"/likely;
    higher = model finds it surprising (candidate hallucination signal).
    """
    prompt = f"Question: {question}\nAnswer:"
    full_text = f"{prompt} {answer}"

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)
    answer_start = prompt_ids.shape[1]

    if full_ids.shape[1] <= answer_start:
        return float("nan")  # no answer tokens to score

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # To predict token at position i, we use the logits at position i-1.
    # We want the model's predicted distribution for each answer token,
    # so shift logits/labels by one position, as in standard LM loss.
    shift_logits = logits[0, answer_start - 1:-1, :]
    shift_labels = full_ids[0, answer_start:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs[range(len(shift_labels)), shift_labels]

    avg_neg_log_likelihood = -token_log_probs.mean().item()
    return avg_neg_log_likelihood


def run_model(name, model_id, examples, idx_test):
    print(f"\n=== Perplexity baseline: {name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    scores = []
    labels = []
    for i in idx_test:
        ex = examples[i]
        nll = compute_answer_perplexity(model, tokenizer, ex["question"], ex["answer"])
        scores.append(nll)
        labels.append(ex["label"])

    scores = np.array(scores)
    labels = np.array(labels)

    valid = ~np.isnan(scores)
    auroc = roc_auc_score(labels[valid], scores[valid])

    df = pd.DataFrame({
        "question": [examples[i]["question"] for i in idx_test],
        "answer": [examples[i]["answer"] for i in idx_test],
        "label": labels,
        "perplexity_score": scores,
    })
    out_path = os.path.join(RESULTS_DIR, f"baseline_perplexity_{name}.csv")
    df.to_csv(out_path, index=False)

    print(f"Perplexity-baseline AUROC ({name}): {auroc:.3f}")
    print(f"Saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return {"model": name, "perplexity_auroc": auroc}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    examples = build_examples()
    labels_all = np.array([ex["label"] for ex in examples])

    # Same split logic as 02_train_probes.py — we only need the TEST
    # indices here, since perplexity requires no training.
    _, idx_test = train_test_split(
        np.arange(len(examples)), test_size=0.2,
        random_state=RANDOM_STATE, stratify=labels_all
    )

    summary = []
    for name, model_id in MODELS.items():
        result = run_model(name, model_id, examples, idx_test)
        summary.append(result)

    df = pd.DataFrame(summary)
    out_path = os.path.join(RESULTS_DIR, "baseline_perplexity_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
