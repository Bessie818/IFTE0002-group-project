import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = "results/transformer_full_tuning_results.csv"
CURVES_CSV = "results/transformer_best_run_curves.csv"
OUTPUT_DIR = "figures/transformer"

FINAL_TRANSFORMER_TEST_AUC = 0.7735
TUNED_RF_TEST_AUC = 0.7770


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_zoomed_ylim(values, margin=0.0015):
    ymin = min(values) - margin
    ymax = max(values) + margin
    plt.ylim(ymin, ymax)


def plot_final_validated_auc_curve(curves_df):
    plt.figure(figsize=(10, 6))
    plt.plot(curves_df["epoch"], curves_df["val_auc"], marker="o")

    best_idx = curves_df["val_auc"].idxmax()
    best_epoch = int(curves_df.loc[best_idx, "epoch"])
    best_val_auc = float(curves_df.loc[best_idx, "val_auc"])

    plt.axvline(best_epoch, linestyle="--")
    plt.title("Final Transformer: Validation AUC by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")

    set_zoomed_ylim(curves_df["val_auc"].tolist(), margin=0.0015)

    for _, row in curves_df.iterrows():
        plt.text(
            row["epoch"],
            row["val_auc"] + 0.00015,
            f"{row['val_auc']:.4f}",
            ha="center"
        )

    plt.text(
        best_epoch + 0.5,
        best_val_auc - 0.0012,
        f"Best Epoch = {best_epoch}\nBest Val AUC = {best_val_auc:.4f}"
    )

    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_validated_auc_curve.png"), dpi=300)
    plt.close()


def plot_final_train_loss_curve(curves_df):
    plt.figure(figsize=(10, 6))
    plt.plot(curves_df["epoch"], curves_df["train_loss"], marker="o")

    plt.title("Final Transformer: Train Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")

    train_losses = curves_df["train_loss"].tolist()
    margin = max((max(train_losses) - min(train_losses)) * 0.08, 0.003)
    plt.ylim(min(train_losses) - margin, max(train_losses) + margin)

    for _, row in curves_df.iterrows():
        plt.text(
            row["epoch"],
            row["train_loss"] + margin * 0.08,
            f"{row['train_loss']:.4f}",
            ha="center"
        )

    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_train_loss_curve.png"), dpi=300)
    plt.close()


def plot_stage_line(stage_df, x_col, y_col, title, xlabel, ylabel, filename):
    df = stage_df.copy()

    x_labels = df[x_col].astype(str).tolist()
    y_values = df[y_col].tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, y_values, marker="o")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    set_zoomed_ylim(y_values, margin=0.0015)

    for x, value in zip(x_labels, y_values):
        plt.text(x, value + 0.00015, f"{value:.4f}", ha="center")

    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def plot_attention_layer_comparison(stage_df):
    df = stage_df.copy()
    df["label"] = df.apply(
        lambda row: f"{row['attention_type'].capitalize()}\nL{int(row['num_layers'])}",
        axis=1
    )

    values = df["best_val_auc"].tolist()

    plt.figure(figsize=(11, 6))
    bars = plt.bar(df["label"], values)

    plt.title("Final Transformer: Attention Type and Layer Comparison")
    plt.xlabel("Model Configuration")
    plt.ylabel("Best Validation AUC")

    set_zoomed_ylim(values, margin=0.0015)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00015,
            f"{value:.4f}",
            ha="center"
        )

    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_attention_layer_comparison.png"), dpi=300)
    plt.close()


def plot_transformer_vs_rf():
    labels = ["Final Transformer", "Tuned RF"]
    values = [FINAL_TRANSFORMER_TEST_AUC, TUNED_RF_TEST_AUC]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values)

    plt.title("Final Transformer vs Random Forest (Test AUC)")
    plt.xlabel("Model")
    plt.ylabel("Test AUC")

    set_zoomed_ylim(values, margin=0.0015)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00015,
            f"{value:.4f}",
            ha="center"
        )

    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_transformer_vs_rf.png"), dpi=300)
    plt.close()


def main():
    ensure_output_dir()

    tuning_df = pd.read_csv(RESULTS_CSV)
    curves_df = pd.read_csv(CURVES_CSV)

    # 1. final validation AUC curve
    plot_final_validated_auc_curve(curves_df)

    # 2. final train loss curve
    plot_final_train_loss_curve(curves_df)

    # 3. learning rate tuning
    lr_df = tuning_df[tuning_df["stage"] == "learning_rate"].copy()
    lr_df = lr_df.sort_values("lr", ascending=False)
    plot_stage_line(
        lr_df,
        x_col="lr",
        y_col="best_val_auc",
        title="Learning Rate Tuning (Best Validation AUC)",
        xlabel="Learning Rate",
        ylabel="Best Validation AUC",
        filename="final_lr_tuning.png"
    )

    # 4. d_model tuning
    dmodel_df = tuning_df[tuning_df["stage"] == "d_model"].copy()
    dmodel_df = dmodel_df.sort_values("d_model")
    plot_stage_line(
        dmodel_df,
        x_col="d_model",
        y_col="best_val_auc",
        title="d_model Tuning (Best Validation AUC)",
        xlabel="d_model",
        ylabel="Best Validation AUC",
        filename="final_d_model_tuning.png"
    )

    # 5. dropout tuning
    dropout_df = tuning_df[tuning_df["stage"] == "dropout"].copy()
    dropout_df = dropout_df.sort_values("dropout")
    plot_stage_line(
        dropout_df,
        x_col="dropout",
        y_col="best_val_auc",
        title="Dropout Tuning (Best Validation AUC)",
        xlabel="Dropout",
        ylabel="Best Validation AUC",
        filename="final_dropout_tuning.png"
    )

    # 6. attention + layer comparison
    attn_layer_df = tuning_df[tuning_df["stage"] == "attention_and_layers"].copy()
    attn_layer_df = attn_layer_df.sort_values(["attention_type", "num_layers"])
    plot_attention_layer_comparison(attn_layer_df)

    # 7. final transformer vs RF
    plot_transformer_vs_rf()

    print("Final Transformer figures saved successfully:")
    print(os.path.join(OUTPUT_DIR, "final_validated_auc_curve.png"))
    print(os.path.join(OUTPUT_DIR, "final_train_loss_curve.png"))
    print(os.path.join(OUTPUT_DIR, "final_lr_tuning.png"))
    print(os.path.join(OUTPUT_DIR, "final_d_model_tuning.png"))
    print(os.path.join(OUTPUT_DIR, "final_dropout_tuning.png"))
    print(os.path.join(OUTPUT_DIR, "final_attention_layer_comparison.png"))
    print(os.path.join(OUTPUT_DIR, "final_transformer_vs_rf.png"))


if __name__ == "__main__":
    main()