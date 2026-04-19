import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from transformer_tokenizer import CreditCardTokenizer
from credit_card_transformer_model_training_d import CreditCardTransformerClassifier


class CreditCardDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

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

    acc = accuracy_score(all_labels, pred_classes)
    auc = roc_auc_score(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)

    return avg_loss, acc, auc


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

X_train_encoded = tokenizer.encode(train_df)
y_train = train_df[target_col].values

X_val_encoded = tokenizer.encode(val_df)
y_val = val_df[target_col].values

X_test_encoded = tokenizer.encode(test_df)
y_test = test_df[target_col].values

batch_size = 128

train_dataset = CreditCardDataset(X_train_encoded, y_train)
val_dataset = CreditCardDataset(X_val_encoded, y_val)
test_dataset = CreditCardDataset(X_test_encoded, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CreditCardTransformerClassifier(
    vocab_size=tokenizer.vocab_size,
    sequence_length=tokenizer.sequence_length,
    pad_token_id=tokenizer.vocab["[PAD]"],
    d_model=64,
    ffn_hidden_dim=128,
    dropout=0.2,
    num_layers=1,
    attention_type="single",
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 15

best_val_auc = -1.0
best_epoch = -1
best_model_state = None

for epoch in range(num_epochs):
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

    val_loss, val_acc, val_auc = evaluate_model(model, val_loader, device)

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch + 1
        best_model_state = copy.deepcopy(model.state_dict())

print("\nSelecting best model based on validation AUC...")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation AUC: {best_val_auc:.4f}")

model.load_state_dict(best_model_state)

test_loss, test_acc, test_auc = evaluate_model(model, test_loader, device)

print("\n--- Final Test Result Using Best Validation Model ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

print("\n--- Prediction Example ---")
sample_data = test_df.iloc[[0]]
sample_encoded = torch.tensor(
    tokenizer.encode(sample_data),
    dtype=torch.long
).to(device)

model.eval()
with torch.no_grad():
    sample_logit = model(sample_encoded)
    sample_prob = torch.sigmoid(sample_logit).item()

print(f"True Label: {test_df.iloc[0][target_col]}")
print(f"Predicted Default Probability: {sample_prob * 100:.2f}%")