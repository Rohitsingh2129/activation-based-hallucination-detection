"""
Step 11b: Same as 11_extract_fever.py, but loads each model in 4-bit
(NF4) quantization via bitsandbytes, instead of fp16 -- the FEVER
counterpart to 01b_extract_activations_4bit.py's fp16-vs-4bit
comparison, extended to all three models in one script (matching
08_extract_halueval.py's loop-over-models pattern rather than the
original per-model-per-file layout).

Output: data/activations_fever_{model}_4bit.npz, same array format as
11_extract_fever.py's output, so 12_train_probes_fever.py works on it
unchanged.

This script is purely additive: it does not read, modify, or re-run any
TruthfulQA or HaluEval file.
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        examples.append({"claim": row["claim"], "label": 0})
    for row in refutes:
        examples.append({"claim": row["claim"], "label": 1})
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
    print(f"\n=== Extracting FEVER activations (4-bit NF4): {name} ===")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map=DEVICE
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

    out_path = os.path.join(OUTPUT_DIR, f"activations_fever_{name}_4bit.npz")
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

    # FEVER dataset is cached by now (11_extract_fever.py runs first),
    # and all three models are cached from earlier runs, so this whole
    # script can safely run fully offline from the start.
    os.environ["HF_HUB_OFFLINE"] = "1"

    for name, model_id in MODELS.items():
        run_model(name, model_id, examples)


if __name__ == "__main__":
    main()
