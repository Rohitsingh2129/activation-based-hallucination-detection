"""
Step 1b: Same as 01_extract_activations.py, but loads Qwen2.5-3B-Instruct
in 4-bit (NF4) quantization via bitsandbytes, instead of fp16.

Why this matters for the thesis: the closest prior paper validated
quantized activation probing at 7-8B. This script produces the direct
3B-scale comparison point — same model, same dataset, same probing
method as script 1, with only the precision changed. That controlled
comparison (fp16 vs 4-bit, everything else identical) is exactly what
your quantization-gap claim needs as evidence.

Output: data/activations_truthfulqa_4bit.npz (same format as script 1's
output, so 02_train_probes.py works on this file unchanged — just point
DATA_PATH at it).
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        # Note: hidden states come out as float16/bfloat16 even under
        # 4-bit weight quantization (only the WEIGHTS are 4-bit; the
        # activations we're extracting stay in compute dtype). Cast to
        # float32 for numpy/sklearn compatibility, same as script 1.
        layer_means.append(answer_tokens.mean(dim=0).float().cpu().numpy())

    return np.stack(layer_means)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME} in 4-bit (NF4) ...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
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

    hidden_states = np.stack(all_hidden)
    labels = np.array(all_labels)

    out_path = os.path.join(OUTPUT_DIR, "activations_truthfulqa_4bit.npz")
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
