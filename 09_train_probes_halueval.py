"""
Step 9: Train layer-wise probes on HaluEval activations, all three
models. Same logic as 02_train_probes.py, looped over models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

RESULTS_DIR = "results"
RANDOM_STATE = 42
MODELS = ["qwen", "phi3", "llama"]


def train_and_eval(name):
    print(f"\n=== HaluEval probe: {name} ===")
    data = np.load(f"data/activations_halueval_{name}.npz", allow_pickle=True)
    hidden_states, labels = data["hidden_states"], data["labels"]
    n_examples, n_layers, _ = hidden_states.shape

    idx_train, idx_test = train_test_split(
        np.arange(n_examples), test_size=0.2,
        random_state=RANDOM_STATE, stratify=labels
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

        results.append({
            "layer": layer,
            "auroc": roc_auc_score(y_test, probs),
            "f1": f1_score(y_test, preds),
            "accuracy": accuracy_score(y_test, preds),
        })

    df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, f"layer_wise_auroc_halueval_{name}.csv")
    df.to_csv(out_path, index=False)

    best = df.loc[df["auroc"].idxmax()]
    print(f"Best layer: {int(best['layer'])} (AUROC={best['auroc']:.3f}, F1={best['f1']:.3f})")
    print(f"Saved to {out_path}")
    return {"model": name, "best_layer": int(best["layer"]), "auroc": best["auroc"]}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = [train_and_eval(name) for name in MODELS]
    df = pd.DataFrame(summary)
    print("\n=== HaluEval summary (all models) ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "halueval_summary.csv"), index=False)


if __name__ == "__main__":
    main()
