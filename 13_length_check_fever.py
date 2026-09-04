"""
Step 13: Length-confound check for FEVER, run BEFORE reporting any
headline probe numbers -- exactly the check that, done only after the
fact on HaluEval, revealed that dataset's ~0.99 AUROC was substantially
inflated by a 5x length gap between right_answer and
hallucinated_answer (see 10_length_ablation.py).

This script needs only claim text + labels, not activations, so it runs
standalone on CPU and can be checked before committing to the GPU
extraction runs. It reproduces 11_extract_fever.py's build_examples()
exactly (same repo, split, seed, SAMPLE_SIZE) to get the same claim set
and ordering, and reports:
  - mean/median word and character length of SUPPORTS vs. REFUTES claims
  - AUROC of a logistic regression using length ALONE as the predictor
    (word count and character count), on the identical train/test split
    convention (RANDOM_STATE=42, test_size=0.2, stratify) used by
    12_train_probes_fever.py, so the number is directly comparable to
    the eventual probe AUROC.

Unlike HaluEval, FEVER claims are both human-written short factual
sentences regardless of label (there's no "terse right answer vs.
verbose fabrication" asymmetry by construction), so a priori this
confound is expected to be much weaker or absent here -- this script
checks that assumption rather than assuming it.
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results"
RANDOM_STATE = 42
SAMPLE_SIZE = 2000

FEVER_REPO = "Dzeniks/fever_3way"
FEVER_SPLIT = "validation"
LABEL_SUPPORTS = 0
LABEL_REFUTES = 1


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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    examples = build_examples()
    labels = np.array([ex["label"] for ex in examples])
    word_len = np.array([len(ex["claim"].split()) for ex in examples])
    char_len = np.array([len(ex["claim"]) for ex in examples])

    print(f"Built {len(examples)} examples "
          f"({int((labels == 0).sum())} SUPPORTS / {int((labels == 1).sum())} REFUTES)")

    stats_rows = []
    for label, name in [(0, "SUPPORTS (truthful)"), (1, "REFUTES (hallucinated)")]:
        mask = labels == label
        stats_rows.append({
            "label": label, "name": name,
            "mean_words": word_len[mask].mean(), "median_words": np.median(word_len[mask]),
            "mean_chars": char_len[mask].mean(), "median_chars": np.median(char_len[mask]),
        })
        print(f"{name}: mean_words={word_len[mask].mean():.2f} "
              f"median_words={np.median(word_len[mask]):.1f} "
              f"mean_chars={char_len[mask].mean():.1f}")

    idx_train, idx_test = train_test_split(
        np.arange(len(examples)), test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    y_train, y_test = labels[idx_train], labels[idx_test]

    def length_only_auroc(feature):
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(feature[idx_train].reshape(-1, 1))
        test_scaled = scaler.transform(feature[idx_test].reshape(-1, 1))
        clf = LogisticRegression(max_iter=1000).fit(train_scaled, y_train)
        return roc_auc_score(y_test, clf.predict_proba(test_scaled)[:, 1])

    word_auroc = length_only_auroc(word_len)
    char_auroc = length_only_auroc(char_len)

    print(f"\nLength-only AUROC -- word count: {word_auroc:.4f}  char count: {char_auroc:.4f}")

    confound_threshold = 0.65
    flagged = max(word_auroc, char_auroc) >= confound_threshold
    verdict = (
        f"CONFOUND FLAGGED (>= {confound_threshold}): run 14_length_ablation_fever.py "
        f"before reporting headline probe AUROC on FEVER."
        if flagged else
        f"No meaningful length confound detected (< {confound_threshold}): "
        f"headline probe AUROC on FEVER can be reported without a length ablation."
    )
    print(f"\n{verdict}")

    summary = pd.DataFrame(stats_rows)
    summary["length_only_auroc_word"] = word_auroc
    summary["length_only_auroc_char"] = char_auroc
    summary["confound_flagged"] = flagged
    out_path = os.path.join(RESULTS_DIR, "length_confound_check_fever.csv")
    summary.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
