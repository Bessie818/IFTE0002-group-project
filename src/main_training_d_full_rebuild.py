import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from transformer_tokenizer import CreditCardTokenizer
from credit_card_transformer_model_training_d import CreditCardTransformerClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CreditCardDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def make_dataloaders(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size: int
):
    train_dataset = CreditCardDataset(X_train, y_train)
    val_dataset = CreditCardDataset(X_val, y_val)
    test_dataset = CreditCardDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_ids, batch_labels in data_loader:
            batch_ids = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_ids)
            loss = criterion(logits, batch_labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = (all_preds > 0.5).astype(int)

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, pred_classes)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, acc, auc


def build_model(tokenizer, config, device):
    d_model = config["d_model"]
    ffn_hidden_dim = config["ffn_hidden_dim"]

    common_kwargs = {
        "vocab_size": tokenizer.vocab_size,
        "sequence_length": tokenizer.sequence_length,
        "pad_token_id": tokenizer.vocab["[PAD]"],
        "d_model": d_model,
        "ffn_hidden_dim": ffn_hidden_dim,
        "dropout": config["dropout"],
        "num_layers": config["num_layers"],
        "attention_type": config["attention_type"],
    }

    if config["attention_type"] == "multi":
        model = CreditCardTransformerClassifier(
            **common_kwargs,
            num_heads=4,
        )
    else:
        model = CreditCardTransformerClassifier(**common_kwargs)

    return model.to(device)


def train_one_config(
    stage_name,
    config,
    tokenizer,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    device,
    pos_weight
):
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config["batch_size"]
    )

    model = build_model(tokenizer, config, device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-4
    )

    best_val_auc = -1.0
    best_epoch = -1
    best_model_state = None
    best_train_loss = None
    best_val_loss = None
    best_val_acc = None

    epochs_no_improve = 0
    patience = config["patience"]

    train_loss_curve = []
    val_loss_curve = []
    val_auc_curve = []

    print("\n" + "=" * 80)
    print(f"[{stage_name}] Running config:")
    print(config)
    print("=" * 80)

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total_train_loss = 0.0

        for batch_ids, batch_labels in train_loader:
            batch_ids = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_ids)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_auc = evaluate_model(model, val_loader, device, criterion)

        train_loss_curve.append(avg_train_loss)
        val_loss_curve.append(val_loss)
        val_auc_curve.append(val_auc)

        print(
            f"Epoch {epoch}/{config['max_epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_train_loss = avg_train_loss
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    result = {
        "stage": stage_name,
        "attention_type": config["attention_type"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
        "lr": config["lr"],
        "d_model": config["d_model"],
        "attention_dim": config["attention_dim"],
        "ffn_hidden_dim": config["ffn_hidden_dim"],
        "batch_size": config["batch_size"],
        "max_epochs": config["max_epochs"],
        "patience": config["patience"],
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_auc": best_val_auc,
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "val_auc_curve": val_auc_curve,
        "best_model_state": best_model_state,
        "config": copy.deepcopy(config),
    }

    print("\nBest validation result for this config:")
    print(
        f"Best Epoch={best_epoch} | "
        f"Best Train Loss={best_train_loss:.4f} | "
        f"Best Val Loss={best_val_loss:.4f} | "
        f"Best Val Acc={best_val_acc:.4f} | "
        f"Best Val AUC={best_val_auc:.4f}"
    )

    return result


def run_stage(
    stage_name,
    configs,
    tokenizer,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    device,
    pos_weight,
    all_rows
):
    detailed_results = []

    for config in configs:
        result = train_one_config(
            stage_name=stage_name,
            config=config,
            tokenizer=tokenizer,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            device=device,
            pos_weight=pos_weight
        )
        detailed_results.append(result)

        all_rows.append({
            "stage": result["stage"],
            "attention_type": result["attention_type"],
            "num_layers": result["num_layers"],
            "dropout": result["dropout"],
            "lr": result["lr"],
            "d_model": result["d_model"],
            "attention_dim": result["attention_dim"],
            "ffn_hidden_dim": result["ffn_hidden_dim"],
            "batch_size": result["batch_size"],
            "max_epochs": result["max_epochs"],
            "patience": result["patience"],
            "best_epoch": result["best_epoch"],
            "best_train_loss": result["best_train_loss"],
            "best_val_loss": result["best_val_loss"],
            "best_val_acc": result["best_val_acc"],
            "best_val_auc": result["best_val_auc"],
        })

    best_result = max(detailed_results, key=lambda x: x["best_val_auc"])

    print("\n" + "#" * 80)
    print(f"[{stage_name}] Best config selected by validation AUC")
    print(best_result["config"])
    print(
        f"Best Epoch={best_result['best_epoch']} | "
        f"Best Val AUC={best_result['best_val_auc']:.4f}"
    )
    print("#" * 80)

    return best_result


def final_test_evaluation(best_result, tokenizer, X_test, y_test, device, pos_weight):
    config = best_result["config"]

    test_dataset = CreditCardDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = build_model(tokenizer, config, device)
    model.load_state_dict(best_result["best_model_state"])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_loss, test_acc, test_auc = evaluate_model(model, test_loader, device, criterion)

    return test_loss, test_acc, test_auc


def main():
    set_seed(42)
    os.makedirs("results", exist_ok=True)

    df = pd.read_excel("default_credit_card_clients.xls", header=1)

    categorical_features = [
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
    ]

    numerical_features = [
        "LIMIT_BAL", "AGE",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    feature_order = categorical_features + numerical_features
    target_col = "default payment next month"

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_col]
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=42,
        stratify=train_val_df[target_col]
    )

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    print("Fitting tokenizer on training set only...")
    tokenizer = CreditCardTokenizer(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        feature_order=feature_order,
        num_bins=10
    )
    tokenizer.fit(train_df)

    X_train = tokenizer.encode(train_df)
    y_train = train_df[target_col].values

    X_val = tokenizer.encode(val_df)
    y_val = val_df[target_col].values

    X_test = tokenizer.encode(test_df)
    y_test = test_df[target_col].values

    negative_count = (train_df[target_col] == 0).sum()
    positive_count = (train_df[target_col] == 1).sum()
    ratio = negative_count / positive_count

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([ratio], dtype=torch.float32).to(device)

    print(f"Using device: {device}")
    print(f"Negative count: {negative_count}")
    print(f"Positive count: {positive_count}")
    print(f"pos_weight ratio: {ratio:.4f}")

    base_config = {
        "attention_type": "multi",
        "num_layers": 3,
        "dropout": 0.2,
        "lr": 1e-3,
        "d_model": 64,
        "attention_dim": 64,
        "ffn_hidden_dim": 128,
        "batch_size": 128,
        "max_epochs": 30,
        "patience": 5,
    }

    all_rows = []

    # Stage 1: learning rate
    stage1_configs = []
    for lr in [1e-3, 5e-4, 1e-4]:
        cfg = copy.deepcopy(base_config)
        cfg["lr"] = lr
        stage1_configs.append(cfg)

    best_stage1 = run_stage(
        "learning_rate",
        stage1_configs,
        tokenizer,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        device,
        pos_weight,
        all_rows
    )

    # Stage 2: d_model
    stage2_configs = []
    for d_model in [16, 32, 64]:
        cfg = copy.deepcopy(best_stage1["config"])
        cfg["d_model"] = d_model
        cfg["attention_dim"] = d_model
        cfg["ffn_hidden_dim"] = d_model * 2
        stage2_configs.append(cfg)

    best_stage2 = run_stage(
        "d_model",
        stage2_configs,
        tokenizer,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        device,
        pos_weight,
        all_rows
    )

    # Stage 3: dropout
    stage3_configs = []
    for dropout in [0.1, 0.2, 0.3]:
        cfg = copy.deepcopy(best_stage2["config"])
        cfg["dropout"] = dropout
        stage3_configs.append(cfg)

    best_stage3 = run_stage(
        "dropout",
        stage3_configs,
        tokenizer,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        device,
        pos_weight,
        all_rows
    )

    # Stage 4: attention type + layers
    stage4_configs = []
    for attention_type, num_layers in [
        ("single", 1), ("single", 2), ("single", 3),
        ("multi", 1), ("multi", 2), ("multi", 3),
    ]:
        cfg = copy.deepcopy(best_stage3["config"])
        cfg["attention_type"] = attention_type
        cfg["num_layers"] = num_layers
        stage4_configs.append(cfg)

    best_stage4 = run_stage(
        "attention_and_layers",
        stage4_configs,
        tokenizer,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        device,
        pos_weight,
        all_rows
    )

    # Stage 5: batch size
    stage5_configs = []
    for batch_size in [32, 64, 128]:
        cfg = copy.deepcopy(best_stage4["config"])
        cfg["batch_size"] = batch_size
        stage5_configs.append(cfg)

    best_stage5 = run_stage(
        "batch_size",
        stage5_configs,
        tokenizer,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        device,
        pos_weight,
        all_rows
    )

    final_test_loss, final_test_acc, final_test_auc = final_test_evaluation(
        best_stage5,
        tokenizer,
        X_test,
        y_test,
        device,
        pos_weight
    )

    tuning_csv_path = "results/transformer_full_tuning_results.csv"
    pd.DataFrame(all_rows).to_csv(tuning_csv_path, index=False)

    curves_df = pd.DataFrame({
        "epoch": list(range(1, len(best_stage5["train_loss_curve"]) + 1)),
        "train_loss": best_stage5["train_loss_curve"],
        "val_loss": best_stage5["val_loss_curve"],
        "val_auc": best_stage5["val_auc_curve"],
    })
    curves_csv_path = "results/transformer_best_run_curves.csv"
    curves_df.to_csv(curves_csv_path, index=False)

    summary_txt_path = "results/transformer_best_config.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("Best config selected using validation only\n")
        f.write(str(best_stage5["config"]) + "\n\n")
        f.write(f"Best Epoch = {best_stage5['best_epoch']}\n")
        f.write(f"Best Train Loss = {best_stage5['best_train_loss']:.4f}\n")
        f.write(f"Best Val Loss = {best_stage5['best_val_loss']:.4f}\n")
        f.write(f"Best Val Acc = {best_stage5['best_val_acc']:.4f}\n")
        f.write(f"Best Val AUC = {best_stage5['best_val_auc']:.4f}\n\n")
        f.write("Final test evaluation (used once at the end)\n")
        f.write(f"Final Test Loss = {final_test_loss:.4f}\n")
        f.write(f"Final Test Acc = {final_test_acc:.4f}\n")
        f.write(f"Final Test AUC = {final_test_auc:.4f}\n")

    print("\n" + "=" * 80)
    print("FINAL BEST CONFIG (selected by validation only)")
    print(best_stage5["config"])
    print(f"Best Epoch = {best_stage5['best_epoch']}")
    print(f"Best Validation AUC = {best_stage5['best_val_auc']:.4f}")
    print()
    print("FINAL TEST RESULT (evaluated once at the end)")
    print(f"Final Test Loss = {final_test_loss:.4f}")
    print(f"Final Test Acc = {final_test_acc:.4f}")
    print(f"Final Test AUC = {final_test_auc:.4f}")
    print()
    print(f"Saved: {tuning_csv_path}")
    print(f"Saved: {curves_csv_path}")
    print(f"Saved: {summary_txt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()