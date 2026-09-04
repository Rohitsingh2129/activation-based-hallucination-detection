"""
Plot fp16 vs 4-bit layer-wise AUROC curves side by side.
Reads the two CSVs already produced by 02_train_probes.py and saves a
single comparison figure — this is the figure both the paper and thesis
will want for the quantization-robustness result.
"""

import pandas as pd
import matplotlib.pyplot as plt

fp16 = pd.read_csv("results/layer_wise_auroc.csv")
quant = pd.read_csv("results/layer_wise_auroc_4bit.csv")

plt.figure(figsize=(9, 5.5))
plt.plot(fp16["layer"], fp16["auroc"], marker="o", label="fp16 (full precision)", linewidth=2)
plt.plot(quant["layer"], quant["auroc"], marker="s", label="4-bit NF4 (quantized)", linewidth=2)

best_fp16_layer = fp16.loc[fp16["auroc"].idxmax(), "layer"]
best_quant_layer = quant.loc[quant["auroc"].idxmax(), "layer"]
plt.axvline(best_fp16_layer, color="gray", linestyle="--", alpha=0.4)

plt.xlabel("Transformer Layer")
plt.ylabel("AUROC")
plt.title("Layer-wise Hallucination Probe AUROC: fp16 vs 4-bit (Qwen2.5-3B-Instruct, TruthfulQA)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/auroc_fp16_vs_4bit.png", dpi=200, metadata={"Software": ""})
print("Saved plot to results/auroc_fp16_vs_4bit.png")
print(f"Best fp16 layer: {best_fp16_layer} | Best 4-bit layer: {best_quant_layer}")
