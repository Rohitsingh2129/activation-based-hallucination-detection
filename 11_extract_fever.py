"""
Step 11: Extract activations from FEVER (claim verification) for all
three models -- the third dataset, testing generalization beyond
TruthfulQA (multiple-choice, ~80/20 skew) and HaluEval (QA right vs.
hallucinated answer, 50/50).

Dataset: Dzeniks/fever_3way, split "validation". Fields: id, claim,
label (0=SUPPORTS, 1=REFUTES, 2=NOT ENOUGH INFO -- confirmed by
cross-checking the evidence field, which is empty only for label 2,
against the well-known official FEVER class sizes), evidence. The
validation split is already perfectly class-balanced (3333 per label),
unlike train. For the binary hallucination-detection framing used
everywhere else in this project: SUPPORTS -> label 0 (truthful),
REFUTES -> label 1 (hallucinated); NOT ENOUGH INFO is dropped entirely
since it doesn't map onto "correct vs. incorrect claim."

SAMPLE_SIZE=2000 per class (4000 total) -- matches
08_extract_halueval.py's convention exactly, for the same reason: a
balanced set comparable in scale to the other two datasets, well below
the full ~6666-example SUPPORTS+REFUTES pool, sampled with a fixed seed.

FEVER has no natural question/answer split -- it's a single claim to be
judged, not a question with a separate answer. The claim itself plays
the role "answer" plays in the other two scripts: it's the text whose
processing representation we mean-pool over, using a minimal "Claim:"
prefix as the framing prompt (analogous to "Question: ...\\nAnswer:").
This preserves the core mechanism (pool over the text being judged for
truthfulness) even though the surface format differs.

Loops over all three models in one script, same pattern as
08_extract_halueval.py. Output: data/activations_fever_{model}.npz,
same array format as the other two datasets (hidden_states, labels,
questions -- here, questions holds the claim text, since FEVER has no
separate question field; kept under the same key name for compatibility
with scripts that expect it, e.g. for inspection/debugging).

This script is purely additive: it does not read, modify, or re-run any
TruthfulQA or HaluEval file.
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

FEVER_REPO = "Dzeniks/fever_3way"
FEVER_SPLIT = "validation"
LABEL_SUPPORTS = 0
LABEL_REFUTES = 1

MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def build_examples():
    ds = load_dataset(FEVER_REPO)[FEVER_SPLIT]
    ds = ds.shuffle(seed=RANDOM_STATE)

    supports = ds.filter(lambda r: r["label"] == LABEL_SUPPORTS).select(range(SAMPLE_SIZE))
    refutes = ds.filter(lambda r: r["label"] == LABEL_REFUTES).select(range(SAMPLE_SIZE))

    examples = []
    for row in supports:
        examples.append({"claim": row["claim"], "label": 0})  # truthful
    for row in refutes:
        examples.append({"claim": row["claim"], "label": 1})  # hallucinated
    return examples


def get_hidden_states(model, tokenizer, claim):
    prompt = "Claim:"
    full_text = f"{prompt} {claim}"

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)
    claim_start = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True)

    layer_means = []
    for layer_hidden in outputs.hidden_states:
        claim_tokens = layer_hidden[0, claim_start:, :]
        if claim_tokens.shape[0] == 0:
            claim_tokens = layer_hidden[0, -1:, :]
        layer_means.append(claim_tokens.mean(dim=0).float().cpu().numpy())

    return np.stack(layer_means)


def run_model(name, model_id, examples):
    print(f"\n=== Extracting FEVER activations: {name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    all_hidden, all_labels, all_claims = [], [], []
    for i, ex in enumerate(examples):
        hs = get_hidden_states(model, tokenizer, ex["claim"])
        all_hidden.append(hs)
        all_labels.append(ex["label"])
        all_claims.append(ex["claim"])
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(examples)}")

    hidden_states = np.stack(all_hidden)
    labels = np.array(all_labels)

    out_path = os.path.join(OUTPUT_DIR, f"activations_fever_{name}.npz")
    np.savez_compressed(
        out_path, hidden_states=hidden_states, labels=labels,
        questions=np.array(all_claims, dtype=object),
    )
    print(f"Saved {hidden_states.shape} to {out_path}")
    print(f"Label balance: {labels.sum()} hallucinated / {len(labels)} total")

    del model
    torch.cuda.empty_cache()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    examples = build_examples()
    print(f"Built {len(examples)} examples from {SAMPLE_SIZE} FEVER SUPPORTS "
          f"+ {SAMPLE_SIZE} FEVER REFUTES claims")

    # Dataset download (above) needs network access since FEVER isn't
    # cached yet. All three models ARE already cached from earlier runs,
    # so switch to offline mode now to avoid the redundant Hub metadata
    # check that previously caused a silent hang after model load.
    os.environ["HF_HUB_OFFLINE"] = "1"

    for name, model_id in MODELS.items():
        run_model(name, model_id, examples)


if __name__ == "__main__":
    main()
