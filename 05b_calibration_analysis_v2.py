"""
Step 5 (revised): Calibration analysis with a proper three-way split.

Fixes the one flagged issue from the first version: temperature scaling
was previously fit AND evaluated on the same test set, which lets it
slightly overfit that specific split. This version uses three disjoint
splits:

  - TRAIN (60%): trains the logistic regression probe (same as before)
  - VAL   (20%): used ONLY to fit the temperature scaling parameter
  - TEST  (20%): held out from BOTH steps above — final ECE and
                 reliability diagrams are computed on this split only

This means the reported ECE-before and ECE-after numbers are now on
genuinely unseen data at every stage, which is the standard, defensible
way to report calibration results in a paper/thesis.

Everything else (bin definitions, ECE formula, temperature-scaling
method, plotting) is unchanged from the first version — only the data
splitting logic changed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar

RESULTS_DIR = "results"
RANDOM_STATE = 42
N_BINS = 10

MODELS = {
    "qwen": {
        "activations": "data/activations_truthfulqa.npz",
        "own_best_csv": "results/layer_wise_auroc.csv",
    },
    "phi3": {
        "activations": "data/activations_phi3_truthfulqa.npz",
        "own_best_csv": "results/layer_wise_auroc_phi3.csv",
    },
    "llama": {
        "activations": "data/activations_llama_truthfulqa.npz",
        "own_best_csv": "results/layer_wise_auroc_llama.csv",
    },
}


def expected_calibration_error(probs, labels, n_bins=N_BINS):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_records = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = mask.sum()
        if count == 0:
            bin_records.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                                 "avg_confidence": np.nan, "actual_freq": np.nan})
            continue

        avg_confidence = probs[mask].mean()
        actual_freq = labels[mask].mean()
        weight = count / len(probs)
        ece += weight * abs(avg_confidence - actual_freq)

        bin_records.append({"bin_lo": lo, "bin_hi": hi, "count": int(count),
                             "avg_confidence": avg_confidence, "actual_freq": actual_freq})

    return ece, pd.DataFrame(bin_records)


def fit_temperature(logits, labels):
    def neg_log_likelihood(T):
        scaled_probs = 1 / (1 + np.exp(-logits / T))
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs))

    result = minimize_scalar(neg_log_likelihood, bounds=(0.05, 10.0), method="bounded")
    return result.x


def three_way_split(n, labels, random_state=RANDOM_STATE):
    """
    60% train / 20% val / 20% test, stratified at each step so class
    balance is preserved in all three splits.
    """
    idx_all = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=0.2, random_state=random_state, stratify=labels
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.25,  # 0.25 of the remaining 80% = 20% of total
        random_state=random_state, stratify=labels[idx_trainval]
    )
    return idx_train, idx_val, idx_test


def analyze_model(name):
    print(f"\n=== Calibration analysis (3-way split): {name} ===")
    info = MODELS[name]
    data = np.load(info["activations"], allow_pickle=True)
    hidden_states, labels = data["hidden_states"], data["labels"]

    own_results = pd.read_csv(info["own_best_csv"])
    best_layer = int(own_results.loc[own_results["auroc"].idxmax(), "layer"])

    n = hidden_states.shape[0]
    idx_train, idx_val, idx_test = three_way_split(n, labels)
    X = hidden_states[:, best_layer, :]

    X_train, y_train = X[idx_train], labels[idx_train]
    X_val, y_val = X[idx_val], labels[idx_val]
    X_test, y_test = X[idx_test], labels[idx_test]

    print(f"Split sizes -> train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    # Probe trained ONLY on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(X_train_scaled, y_train)

    # Temperature fit ONLY on val (probe has not seen val, temperature
    # fitting has not seen test)
    X_val_scaled = scaler.transform(X_val)
    val_logits = probe.decision_function(X_val_scaled)
    T = fit_temperature(val_logits, y_val)

    # Final evaluation ONLY on test (untouched by both prior steps)
    X_test_scaled = scaler.transform(X_test)
    test_logits = probe.decision_function(X_test_scaled)
    raw_probs = 1 / (1 + np.exp(-test_logits))  # T=1 equivalent, i.e. uncalibrated
    calibrated_probs = 1 / (1 + np.exp(-test_logits / T))

    ece_before, bins_before = expected_calibration_error(raw_probs, y_test)
    ece_after, bins_after = expected_calibration_error(calibrated_probs, y_test)

    from sklearn.metrics import roc_auc_score
    test_auroc = roc_auc_score(y_test, raw_probs)  # unchanged by temperature scaling

    print(f"Best layer: {best_layer}")
    print(f"Test AUROC (unaffected by temperature scaling): {test_auroc:.3f}")
    print(f"ECE before temperature scaling: {ece_before:.4f}")
    print(f"Fitted temperature T (from val split): {T:.3f}")
    print(f"ECE after temperature scaling: {ece_after:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, bins_df, title, ece_val in [
        (axes[0], bins_before, "Before Temperature Scaling", ece_before),
        (axes[1], bins_after, "After Temperature Scaling", ece_after),
    ]:
        valid = bins_df.dropna()
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.bar(valid["bin_lo"], valid["actual_freq"], width=0.1, align="edge",
               alpha=0.6, edgecolor="black", label="Actual frequency")
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Actual hallucination frequency")
        ax.set_title(f"{title}\nECE={ece_val:.4f}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Calibration: {name} (layer {best_layer}, held-out test split)")
    plt.tight_layout()
    out_plot = os.path.join(RESULTS_DIR, f"calibration_{name}_v2.png")
    plt.savefig(out_plot, dpi=200, metadata={"Software": ""})
    plt.close(fig)
    print(f"Saved reliability diagram to {out_plot}")

    return {
        "model": name,
        "best_layer": best_layer,
        "test_auroc": test_auroc,
        "ece_before": ece_before,
        "temperature": T,
        "ece_after": ece_after,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = [analyze_model(name) for name in MODELS]
    df = pd.DataFrame(summary)
    out_path = os.path.join(RESULTS_DIR, "calibration_summary_v2.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
