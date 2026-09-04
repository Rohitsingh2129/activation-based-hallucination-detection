"""
Step 2: Train a linear probe (logistic regression) for EVERY layer using
the cached activations from step 1, and report which layer separates
hallucinated vs. truthful answers best.

Why logistic regression first (not MLP): the closest prior work found
MLP probes rarely beat linear probes by more than 0.01 AUROC. Starting
linear is cheaper, faster, and more interpretable — an MLP is only
justified afterward if the linear results reveal a genuine non-linearity
problem, which becomes a documented experimental decision rather than a
skipped step.

This script trains on CPU only. Activations are already extracted, so no
GPU or LLM is needed here — this makes iterating on probe design (feature
scaling, regularization, layer selection) essentially free.

Output:
- results/layer_wise_auroc.csv  (layer, AUROC, F1, accuracy)
- prints the best-performing layer
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

DATA_PATH = "data/activations_llama_truthfulqa_4bit.npz"
RESULTS_DIR = "results"
RANDOM_STATE = 42


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = np.load(DATA_PATH, allow_pickle=True)
    hidden_states = data["hidden_states"]  # (N, num_layers, hidden_dim)
    labels = data["labels"]  # (N,)

    n_examples, n_layers, hidden_dim = hidden_states.shape
    print(f"Loaded {n_examples} examples, {n_layers} layers, dim={hidden_dim}")
    print(f"Label balance: {labels.sum()} hallucinated / {n_examples} total")

    # Fixed train/test split reused across all layers, so results are
    # directly comparable layer-to-layer (same examples in each split).
    idx_train, idx_test = train_test_split(
        np.arange(n_examples),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    results = []

    for layer in range(n_layers):
        X = hidden_states[:, layer, :]
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = labels[idx_train], labels[idx_test]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        probe = LogisticRegression(max_iter=2000, C=1.0)
        probe.fit(X_train_scaled, y_train)

        probs = probe.predict_proba(X_test_scaled)[:, 1]
        preds = probe.predict(X_test_scaled)

        auroc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)

        results.append({"layer": layer, "auroc": auroc, "f1": f1, "accuracy": acc})
        print(f"Layer {layer:2d}: AUROC={auroc:.3f}  F1={f1:.3f}  Acc={acc:.3f}")

    df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "layer_wise_auroc_llama_4bit.csv")
    df.to_csv(out_path, index=False)

    best = df.loc[df["auroc"].idxmax()]
    print(f"\nSaved layer-wise results to {out_path}")
    print(
        f"Best layer: {int(best['layer'])} "
        f"(AUROC={best['auroc']:.3f}, F1={best['f1']:.3f})"
    )


if __name__ == "__main__":
    main()
