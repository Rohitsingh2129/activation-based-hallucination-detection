"""
Step 5: Calibration analysis — the "Confidence-Aware" part of the thesis
title that hasn't been tested yet.

For each model's own best layer, this script:
1. Retrains the same logistic regression probe (identical to
   02_train_probes.py — same split, same layer) to get held-out test
   set probabilities.
2. Buckets those probabilities into 10 bins (0.0-0.1, 0.1-0.2, ..., in the
   standard "reliability diagram" style) and compares each bin's average
   PREDICTED confidence against its ACTUAL hallucination frequency.
3. Computes Expected Calibration Error (ECE): the weighted-average gap
   between predicted confidence and actual frequency across bins. Lower
   is better; 0 = perfect calibration.
4. Applies temperature scaling: a simple post-hoc fix that learns ONE
   scalar parameter (temperature T) to rescale the probe's raw logits,
   often improving calibration without touching the probe's decision
   boundary (so AUROC stays identical — only the confidence NUMBERS
   change, not the rankings).
5. Reports ECE before and after temperature scaling, for all three
   models side by side.

Why this matters for the thesis: AUROC (measured already) tells you if
the probe RANKS hallucinated vs. truthful correctly. It says nothing
about whether a "0.8" output actually means "80% likely." This script
is what actually tests the "Confidence-Aware" claim in the thesis title
— nothing run so far has tested this.
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
    """
    Standard ECE: split [0,1] into n_bins equal-width buckets, and for
    each bucket compute |average predicted confidence - actual positive
    rate|, weighted by how many examples fall in that bucket.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_records = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include the right edge only in the last bin so 1.0 isn't dropped.
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
    """
    Temperature scaling: find the single scalar T that minimizes
    negative log-likelihood of sigmoid(logits / T) against true labels.
    T > 1 makes predictions less confident (pulls toward 0.5), T < 1
    makes them more confident. This does NOT change the ranking of
    examples (so AUROC is unaffected) — only how "sure" the probe claims
    to be.
    """
    def neg_log_likelihood(T):
        scaled_probs = 1 / (1 + np.exp(-logits / T))
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs))

    result = minimize_scalar(neg_log_likelihood, bounds=(0.05, 10.0), method="bounded")
    return result.x


def analyze_model(name):
    print(f"\n=== Calibration analysis: {name} ===")
    info = MODELS[name]
    data = np.load(info["activations"], allow_pickle=True)
    hidden_states, labels = data["hidden_states"], data["labels"]

    own_results = pd.read_csv(info["own_best_csv"])
    best_layer = int(own_results.loc[own_results["auroc"].idxmax(), "layer"])

    n = hidden_states.shape[0]
    idx_train, idx_test = train_test_split(
        np.arange(n), test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    X = hidden_states[:, best_layer, :]
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = labels[idx_train], labels[idx_test]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(X_train_scaled, y_train)

    # Raw probabilities (before temperature scaling)
    raw_probs = probe.predict_proba(X_test_scaled)[:, 1]
    ece_before, bins_before = expected_calibration_error(raw_probs, y_test)

    # Logits needed for temperature scaling
    logits = probe.decision_function(X_test_scaled)
    # Fit temperature on the SAME test set here for simplicity; a more
    # rigorous version would use a held-out validation split separate
    # from the final test set — worth upgrading before final results if
    # this becomes the paper's headline calibration number.
    T = fit_temperature(logits, y_test)
    calibrated_probs = 1 / (1 + np.exp(-logits / T))
    ece_after, bins_after = expected_calibration_error(calibrated_probs, y_test)

    print(f"Best layer: {best_layer}")
    print(f"ECE before temperature scaling: {ece_before:.4f}")
    print(f"Fitted temperature T: {T:.3f}")
    print(f"ECE after temperature scaling: {ece_after:.4f}")

    # Save reliability diagram
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

    fig.suptitle(f"Calibration: {name} (layer {best_layer})")
    plt.tight_layout()
    out_plot = os.path.join(RESULTS_DIR, f"calibration_{name}.png")
    plt.savefig(out_plot, dpi=200, metadata={"Software": ""})
    plt.close(fig)
    print(f"Saved reliability diagram to {out_plot}")

    return {
        "model": name,
        "best_layer": best_layer,
        "ece_before": ece_before,
        "temperature": T,
        "ece_after": ece_after,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = [analyze_model(name) for name in MODELS]
    df = pd.DataFrame(summary)
    out_path = os.path.join(RESULTS_DIR, "calibration_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
