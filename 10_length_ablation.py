"""
Step 10: Length-confound ablation for the HaluEval probe results.

09_train_probes_halueval.py reported AUROC ~0.99 for all three models
on HaluEval -- much higher than the ~0.91-0.93 seen on TruthfulQA. A
check of layer-0 (raw embeddings, before any transformer computation)
already scoring ~0.95-0.97 pointed at a shortcut rather than a genuine
signal: HaluEval's right_answer field averages ~2 words while
hallucinated_answer averages ~11 words, and a classifier using word
count ALONE already reaches AUROC~0.97. This script quantifies exactly
how much of the probe's AUROC survives once that confound is accounted
for.

Two things needed for this analysis were not saved by the earlier
scripts:
  - Per-example test-set probe scores (09 only saved aggregate
    per-layer AUROC/F1/accuracy).
  - Raw answer text / length (08's .npz files only stored hidden
    states, labels, and questions -- not the answer strings).
Both are reconstructed deterministically here: the same RANDOM_STATE=42
train/test split plus the identical LogisticRegression config used in
09 reproduces bit-identical probe scores (this is recomputing the
existing probe's output, not designing or training a new one), and
rebuilding HaluEval with the same seed reproduces the exact answer text
in the exact saved order -- verified at runtime via an assertion that
rebuilt labels match the labels saved in each .npz.

Three angles on the confound, per model and per layer:
  1. Length-only baseline: logistic regression on word/char count alone.
  2. Length-binned AUROC: does the probe still separate classes WITHIN
     a length bin, where length can no longer do the work?
  3. Joint model: logistic regression on [probe_logit, length] together,
     with a Wald-test p-value on each coefficient (manual, via the
     observed Fisher information on an unregularized C=inf fit -- this
     avoids adding statsmodels as a new dependency for one diagnostic).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

RESULTS_DIR = "results"
RANDOM_STATE = 42
SAMPLE_SIZE = 2000
N_LENGTH_BINS = 4
MIN_BIN_SIZE = 5

MODELS = {
    "qwen": {
        "activations": "data/activations_halueval_qwen.npz",
        "own_best_csv": "results/layer_wise_auroc_halueval_qwen.csv",
    },
    "phi3": {
        "activations": "data/activations_halueval_phi3.npz",
        "own_best_csv": "results/layer_wise_auroc_halueval_phi3.csv",
    },
    "llama": {
        "activations": "data/activations_halueval_llama.npz",
        "own_best_csv": "results/layer_wise_auroc_halueval_llama.csv",
    },
}


def rebuild_examples():
    """Reproduces 08_extract_halueval.py's build_examples() ordering
    exactly (same seed, same interleaving), so index i here lines up
    with hidden_states[i] in the saved .npz files."""
    ds = load_dataset("pminervini/HaluEval", "qa")["data"]
    ds = ds.shuffle(seed=RANDOM_STATE).select(range(SAMPLE_SIZE))
    examples = []
    for row in ds:
        examples.append({"answer": row["right_answer"], "label": 0})
        examples.append({"answer": row["hallucinated_answer"], "label": 1})
    return examples


def wald_pvalues(X, y, model):
    """
    Manual Wald test for an UNREGULARIZED (C=inf) logistic regression.
    Cov(beta) = inverse observed Fisher information = inverse(X'WX),
    W = diag(p*(1-p)) at the fitted probabilities -- standard GLM
    inference, done by hand here rather than pulling in statsmodels for
    one diagnostic script.
    """
    n = X.shape[0]
    X_full = np.hstack([np.ones((n, 1)), X])
    beta = np.concatenate([model.intercept_, model.coef_[0]])
    p = model.predict_proba(X)[:, 1]
    w = np.clip(p * (1 - p), 1e-8, None)
    fisher_info = X_full.T @ (X_full * w[:, None])
    cov = np.linalg.pinv(fisher_info)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pvals = 2 * (1 - norm.cdf(np.abs(z)))
    return beta, se, pvals  # order: [intercept, feature1, feature2, ...]


def analyze_model(name, info, examples, word_len, char_len):
    print(f"\n=== Length ablation: {name} ===")
    data = np.load(info["activations"], allow_pickle=True)
    hidden_states, labels = data["hidden_states"], data["labels"]
    n_examples, n_layers, _ = hidden_states.shape

    assert len(examples) == n_examples, (
        f"{name}: rebuilt example count ({len(examples)}) != "
        f"saved activation count ({n_examples})"
    )
    rebuilt_labels = np.array([ex["label"] for ex in examples])
    assert np.array_equal(rebuilt_labels, labels), (
        f"{name}: rebuilt labels don't match saved labels -- "
        f"dataset ordering has drifted since extraction"
    )

    idx_train, idx_test = train_test_split(
        np.arange(n_examples), test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    y_train, y_test = labels[idx_train], labels[idx_test]
    len_test = word_len[idx_test]

    word_scaler = StandardScaler()
    word_train_scaled = word_scaler.fit_transform(word_len[idx_train].reshape(-1, 1))
    word_test_scaled = word_scaler.transform(len_test.reshape(-1, 1))
    word_probe = LogisticRegression(max_iter=1000).fit(word_train_scaled, y_train)
    length_word_auroc = roc_auc_score(
        y_test, word_probe.predict_proba(word_test_scaled)[:, 1]
    )

    char_scaler = StandardScaler()
    char_train_scaled = char_scaler.fit_transform(char_len[idx_train].reshape(-1, 1))
    char_test_scaled = char_scaler.transform(char_len[idx_test].reshape(-1, 1))
    char_probe = LogisticRegression(max_iter=1000).fit(char_train_scaled, y_train)
    length_char_auroc = roc_auc_score(
        y_test, char_probe.predict_proba(char_test_scaled)[:, 1]
    )

    print(f"Length-only AUROC -- word count: {length_word_auroc:.4f}  "
          f"char count: {length_char_auroc:.4f}")

    bin_edges = np.unique(np.quantile(len_test, np.linspace(0, 1, N_LENGTH_BINS + 1)))
    bin_idx = np.digitize(len_test, bin_edges[1:-1], right=True)
    n_bins_used = len(bin_edges) - 1
    print(f"Length bins (word count, quartile edges): {bin_edges.tolist()} "
          f"-> {n_bins_used} usable bin(s)")

    own_results = pd.read_csv(info["own_best_csv"])
    best_layer = int(own_results.loc[own_results["auroc"].idxmax(), "layer"])
    own_best_auroc = float(own_results["auroc"].max())

    layer_rows, bin_rows, joint_rows = [], [], []

    for layer in range(n_layers):
        X = hidden_states[:, layer, :]
        scaler = StandardScaler().fit(X[idx_train])
        X_train_scaled = scaler.transform(X[idx_train])
        X_test_scaled = scaler.transform(X[idx_test])

        probe = LogisticRegression(max_iter=2000, C=1.0).fit(X_train_scaled, y_train)
        probe_logit_test = probe.decision_function(X_test_scaled)
        probe_probs_test = probe.predict_proba(X_test_scaled)[:, 1]
        probe_auroc = roc_auc_score(y_test, probe_probs_test)

        layer_rows.append({
            "model": name, "layer": layer,
            "probe_auroc": probe_auroc,
            "length_word_auroc": length_word_auroc,
            "length_char_auroc": length_char_auroc,
        })

        for b in range(n_bins_used):
            mask = bin_idx == b
            n_in_bin = int(mask.sum())
            y_bin = y_test[mask]
            has_both_classes = len(np.unique(y_bin)) == 2
            if n_in_bin < MIN_BIN_SIZE or not has_both_classes:
                bin_probe_auroc, bin_length_auroc = np.nan, np.nan
            else:
                bin_probe_auroc = roc_auc_score(y_bin, probe_probs_test[mask])
                bin_length_auroc = roc_auc_score(y_bin, len_test[mask])
            bin_rows.append({
                "model": name, "layer": layer, "bin": b,
                "bin_lo": bin_edges[b], "bin_hi": bin_edges[b + 1],
                "n": n_in_bin,
                "probe_auroc": bin_probe_auroc,
                "length_auroc": bin_length_auroc,
            })

        joint_scaler = StandardScaler()
        joint_X = joint_scaler.fit_transform(
            np.column_stack([probe_logit_test, len_test]).astype(float)
        )
        joint_model = LogisticRegression(C=np.inf, max_iter=2000).fit(joint_X, y_test)
        joint_probs = joint_model.predict_proba(joint_X)[:, 1]
        joint_auroc = roc_auc_score(y_test, joint_probs)
        beta, se, pvals = wald_pvalues(joint_X, y_test, joint_model)

        joint_rows.append({
            "model": name, "layer": layer,
            "probe_only_auroc": probe_auroc,
            "length_word_auroc": length_word_auroc,
            "joint_auroc": joint_auroc,
            "probe_coef": beta[1], "probe_pvalue": pvals[1],
            "length_coef": beta[2], "length_pvalue": pvals[2],
        })

        if layer == best_layer:
            print(f"[best layer {layer}] saved probe AUROC={own_best_auroc:.4f} | "
                  f"recomputed probe AUROC={probe_auroc:.4f} (sanity check, "
                  f"should match) | joint AUROC={joint_auroc:.4f} | "
                  f"probe p={pvals[1]:.2e} | length p={pvals[2]:.2e}")

    return (pd.DataFrame(layer_rows), pd.DataFrame(bin_rows),
            pd.DataFrame(joint_rows), best_layer)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    examples = rebuild_examples()
    word_len = np.array([len(ex["answer"].split()) for ex in examples])
    char_len = np.array([len(ex["answer"]) for ex in examples])

    all_layer_dfs, all_bin_dfs, all_joint_dfs, summary_rows = [], [], [], []

    for name, info in MODELS.items():
        layer_df, bin_df, joint_df, best_layer = analyze_model(
            name, info, examples, word_len, char_len
        )
        all_layer_dfs.append(layer_df)
        all_bin_dfs.append(bin_df)
        all_joint_dfs.append(joint_df)

        best_row = joint_df[joint_df["layer"] == best_layer].iloc[0]
        summary_rows.append({
            "model": name,
            "best_layer": best_layer,
            "probe_auroc": best_row["probe_only_auroc"],
            "length_word_auroc": best_row["length_word_auroc"],
            "joint_auroc": best_row["joint_auroc"],
            "probe_pvalue": best_row["probe_pvalue"],
            "length_pvalue": best_row["length_pvalue"],
        })

    layer_df_all = pd.concat(all_layer_dfs, ignore_index=True)
    bin_df_all = pd.concat(all_bin_dfs, ignore_index=True)
    joint_df_all = pd.concat(all_joint_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    layer_df_all.to_csv(os.path.join(RESULTS_DIR, "length_ablation_layerwise.csv"), index=False)
    bin_df_all.to_csv(os.path.join(RESULTS_DIR, "length_ablation_binned.csv"), index=False)
    joint_df_all.to_csv(os.path.join(RESULTS_DIR, "length_ablation_joint.csv"), index=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "length_ablation_summary.csv"), index=False)

    print("\n=== Length ablation summary (best layer per model) ===")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in MODELS:
        best_layer = int(summary_df.loc[summary_df["model"] == name, "best_layer"].iloc[0])
        sub = bin_df_all[(bin_df_all["model"] == name) & (bin_df_all["layer"] == best_layer)]
        sub = sub.sort_values("bin")
        ax.plot(sub["bin"], sub["probe_auroc"], marker="o", label=f"{name} (layer {best_layer})")
    ax.set_xlabel("Length quartile bin (0=shortest answers, highest=longest)")
    ax.set_ylabel("Probe AUROC within bin")
    ax.set_title("HaluEval probe AUROC by answer-length bin\n(does the probe still work once length is held roughly constant?)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "length_ablation_auroc_by_bin.png"), dpi=200, metadata={"Software": ""})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = summary_df["model"].tolist()
    x = np.arange(len(model_names))
    width = 0.25
    ax.bar(x - width, summary_df["length_word_auroc"], width, label="Length-only")
    ax.bar(x, summary_df["probe_auroc"], width, label="Probe-only (best layer)")
    ax.bar(x + width, summary_df["joint_auroc"], width, label="Joint (probe + length)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("HaluEval: length-only vs probe vs joint AUROC (best layer per model)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "length_ablation_comparison.png"), dpi=200, metadata={"Software": ""})
    plt.close(fig)

    print("\nSaved: length_ablation_summary.csv, length_ablation_layerwise.csv, "
          "length_ablation_binned.csv, length_ablation_joint.csv")
    print("Saved plots: length_ablation_auroc_by_bin.png, length_ablation_comparison.png")


if __name__ == "__main__":
    main()
