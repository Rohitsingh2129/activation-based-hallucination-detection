"""
Step 4: Cross-model probe transfer.

Trains a probe on ONE model's best layer, then tests it on EVERY layer
of a DIFFERENT model, without retraining. This only works directly
between models sharing the same hidden dimension — Phi-3-mini and
Llama-3.2-3B both use hidden_size=3072, so this script is restricted to
that pair. Qwen2.5-3B uses hidden_size=2048 and is incompatible with
direct transfer (would need a dimensionality-reduction step first,
e.g. PCA to a shared space — a separate follow-up experiment, not this
script).

Why test the target's every layer rather than just one: if the probe
transfers well ONLY at the target's own best layer, that's weaker
evidence of shared representation than if it transfers well across a
broad range — and if the transfer curve peaks at a DIFFERENT layer than
the target's own best layer, that itself is an interesting, reportable
finding about how differently the two architectures organize this
signal.

Requires: results/layer_wise_auroc_phi3.csv and
_llama.csv (fp16 versions) already exist, to look up each model's own
best layer for reference in the printed output. Requires the fp16
activation files for both models (data/activations_phi3_truthfulqa.npz
and data/activations_llama_truthfulqa.npz).

Output: results/transfer_phi3_to_llama.csv and
results/transfer_llama_to_phi3.csv, each with per-target-layer AUROC.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results"
RANDOM_STATE = 42

MODELS = {
    "phi3": {
        "activations": "data/activations_phi3_truthfulqa.npz",
        "own_best_csv": "results/layer_wise_auroc_phi3.csv",
    },
    "llama": {
        "activations": "data/activations_llama_truthfulqa.npz",
        "own_best_csv": "results/layer_wise_auroc_llama.csv",
    },
}


def load_model_data(name):
    info = MODELS[name]
    data = np.load(info["activations"], allow_pickle=True)
    own_results = pd.read_csv(info["own_best_csv"])
    own_best_layer = int(own_results.loc[own_results["auroc"].idxmax(), "layer"])
    own_best_auroc = own_results["auroc"].max()
    return data["hidden_states"], data["labels"], own_best_layer, own_best_auroc


def run_transfer(source_name, target_name):
    print(f"\n=== Transfer: {source_name} -> {target_name} ===")

    src_hidden, src_labels, src_best_layer, src_best_auroc = load_model_data(source_name)
    tgt_hidden, tgt_labels, tgt_best_layer, tgt_best_auroc = load_model_data(target_name)

    print(f"Source ({source_name}) own best layer: {src_best_layer} (AUROC={src_best_auroc:.3f})")
    print(f"Target ({target_name}) own best layer: {tgt_best_layer} (AUROC={tgt_best_auroc:.3f})")

    # Train the probe ONLY on the source model's best layer, using the
    # SAME train/test split logic as before, but here we only need the
    # training portion — the probe is evaluated entirely on the target
    # model's data, none of which it has seen.
    n_src = src_hidden.shape[0]
    idx_train, _ = train_test_split(
        np.arange(n_src), test_size=0.2, random_state=RANDOM_STATE, stratify=src_labels
    )
    X_src_train = src_hidden[idx_train, src_best_layer, :]
    y_src_train = src_labels[idx_train]

    scaler = StandardScaler()
    X_src_train_scaled = scaler.fit_transform(X_src_train)

    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(X_src_train_scaled, y_src_train)

    # Evaluate on EVERY layer of the target model's held-out test split,
    # using the same test indices the target's own probes were scored
    # on, for a fair apples-to-apples comparison against tgt_best_auroc.
    n_tgt = tgt_hidden.shape[0]
    _, idx_test = train_test_split(
        np.arange(n_tgt), test_size=0.2, random_state=RANDOM_STATE, stratify=tgt_labels
    )
    y_tgt_test = tgt_labels[idx_test]

    results = []
    n_tgt_layers = tgt_hidden.shape[1]
    for layer in range(n_tgt_layers):
        X_tgt_test = tgt_hidden[idx_test, layer, :]
        # IMPORTANT: reuse the SOURCE model's scaler, not a new one fit
        # on the target — the probe's weights were learned in the
        # source's normalized space, so the target must be mapped into
        # that same space for the weights to mean anything.
        X_tgt_test_scaled = scaler.transform(X_tgt_test)
        probs = probe.predict_proba(X_tgt_test_scaled)[:, 1]
        auroc = roc_auc_score(y_tgt_test, probs)
        results.append({"target_layer": layer, "transfer_auroc": auroc})

    df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, f"transfer_{source_name}_to_{target_name}.csv")
    df.to_csv(out_path, index=False)

    best_transfer = df.loc[df["transfer_auroc"].idxmax()]
    print(f"Best transfer AUROC: {best_transfer['transfer_auroc']:.3f} "
          f"at target layer {int(best_transfer['target_layer'])}")
    print(f"(Target's own best-layer AUROC was {tgt_best_auroc:.3f} at layer {tgt_best_layer})")
    print(f"Saved to {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_transfer("phi3", "llama")
    run_transfer("llama", "phi3")


if __name__ == "__main__":
    main()
