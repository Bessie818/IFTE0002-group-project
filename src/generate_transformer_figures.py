import os
import matplotlib.pyplot as plt

# Make sure output directory exists
output_dir = os.path.join("figures", "transformer")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Figure 1: Transformer experiment comparison
# -----------------------------
experiment_labels = [
    "Single\nL3 D0.1",
    "Multi\nL3 D0.1",
    "Multi\nL1 D0.1",
    "Multi\nL2 D0.1",
    "Multi\nL4 D0.1",
    "Multi\nL1 D0.2",
    "Multi\nL1 D0.3",
    "Single\nL1 D0.2",
]

experiment_auc = [
    0.7656,
    0.7669,
    0.7697,
    0.7667,
    0.7667,
    0.7743,
    0.7653,
    0.7744,
]

best_auc = max(experiment_auc)
best_idx = experiment_auc.index(best_auc)

plt.figure(figsize=(12, 6))
bars = plt.bar(experiment_labels, experiment_auc)
plt.axhline(best_auc, linestyle="--", linewidth=1)

for i, v in enumerate(experiment_auc):
    plt.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=9)

plt.title("Transformer Experiment Comparison (Test AUC)")
plt.xlabel("Model Configuration")
plt.ylabel("Test AUC")
plt.ylim(0.760, 0.777)

# Highlight the best bar with an annotation
plt.text(
    best_idx,
    best_auc + 0.0018,
    "Best",
    ha="center",
    fontsize=10,
    fontweight="bold"
)

plt.tight_layout()
fig1_path = os.path.join(output_dir, "fig_transformer_model_comparison.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Figure 2: Validated transformer AUC curve
# -----------------------------
epochs = list(range(1, 16))
val_auc = [
    0.7489,
    0.7533,
    0.7560,
    0.7625,
    0.7599,
    0.7618,
    0.7684,
    0.7685,
    0.7694,
    0.7725,
    0.7695,
    0.7715,
    0.7732,
    0.7726,
    0.7718,
]

best_epoch = 13
best_val_auc = 0.7732

plt.figure(figsize=(10, 5))
plt.plot(epochs, val_auc, marker="o")
plt.axvline(best_epoch, linestyle="--", linewidth=1)
plt.text(best_epoch + 0.2, best_val_auc - 0.0015, f"Best Epoch = {best_epoch}", fontsize=9)
plt.text(best_epoch + 0.2, best_val_auc - 0.0030, f"Best Val AUC = {best_val_auc:.4f}", fontsize=9)

for x, y in zip(epochs, val_auc):
    plt.text(x, y + 0.0004, f"{y:.4f}", ha="center", fontsize=8)

plt.title("Validated Transformer: Validation AUC by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation AUC")
plt.xticks(epochs)
plt.ylim(0.746, 0.776)

plt.tight_layout()
fig2_path = os.path.join(output_dir, "fig_validated_auc_curve.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()

print("Figures saved successfully:")
print(fig1_path)
print(fig2_path)