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

# -----------------------------
# Figure 3: Validated train loss curve (best LR tuning run)
# -----------------------------
epochs = list(range(1, 16))
best_lr_train_loss = [
    0.4667,
    0.4436,
    0.4403,
    0.4388,
    0.4358,
    0.4345,
    0.4317,
    0.4306,
    0.4296,
    0.4282,
    0.4275,
    0.4276,
    0.4269,
    0.4267,
    0.4267,
]

plt.figure(figsize=(10, 5))
plt.plot(epochs, best_lr_train_loss, marker="o")
for x, y in zip(epochs, best_lr_train_loss):
    plt.text(x, y + 0.0005, f"{y:.4f}", ha="center", fontsize=8)

plt.title("Validated Transformer: Train Loss by Epoch (LR = 5e-4)")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.xticks(epochs)
plt.ylim(0.424, 0.470)

plt.tight_layout()
fig3_path = os.path.join(output_dir, "fig_validated_train_loss_curve.png")
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Figure 4: Transformer vs RF final comparison
# -----------------------------
model_names = [
    "Validated\nTransformer",
    "Tuned\nRF",
]
auc_scores = [
    0.7701,
    0.7770,
]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, auc_scores)

for i, v in enumerate(auc_scores):
    plt.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)

plt.title("Transformer vs Random Forest Final Comparison (AUC)")
plt.ylabel("Test AUC")
plt.ylim(0.760, 0.782)

plt.tight_layout()
fig4_path = os.path.join(output_dir, "fig_transformer_vs_rf_comparison.png")
plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close()

print(fig3_path)
print(fig4_path)

# -----------------------------
# Figure 5: Layer ablation comparison
# -----------------------------
layer_labels = [
    "Single\nL1",
    "Single\nL2",
    "Single\nL3",
    "Multi\nL1",
    "Multi\nL2",
    "Multi\nL3",
]

layer_test_auc = [
    0.7682,
    0.7697,
    0.7699,
    0.7666,
    0.7717,
    0.7738,
]

layer_val_auc = [
    0.7722,
    0.7715,
    0.7706,
    0.7748,
    0.7745,
    0.7742,
]

x = range(len(layer_labels))
width = 0.35

plt.figure(figsize=(11, 6))
plt.bar([i - width / 2 for i in x], layer_val_auc, width=width, label="Best Validation AUC")
plt.bar([i + width / 2 for i in x], layer_test_auc, width=width, label="Final Test AUC")

for i, v in enumerate(layer_val_auc):
    plt.text(i - width / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=8)

for i, v in enumerate(layer_test_auc):
    plt.text(i + width / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=8)

plt.title("Validated Layer Ablation: Single-head vs Multi-head")
plt.xlabel("Model Configuration")
plt.ylabel("AUC")
plt.xticks(list(x), layer_labels)
plt.ylim(0.764, 0.777)
plt.legend()

plt.tight_layout()
fig5_path = os.path.join(output_dir, "fig_layer_ablation_comparison.png")
plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close()

print(fig5_path)