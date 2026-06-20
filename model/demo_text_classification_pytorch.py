import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import pandas as pd


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=256, hidden_size=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            batch_first=True,
            # bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        outputs, _ = self.lstm(x)
        pooled = outputs.mean(dim=1)
        return self.fc(pooled)


class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)

        return self.fc(cls_output)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            pred = logits.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    f1 = f1_score(targets, preds, average="weighted")

    print(classification_report(targets, preds))
    return f1


if __name__ == "__main__":
    MODEL_NAME = "klue/bert-base"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("train.csv")

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = TextDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    num_classes = len(set(labels))

    model = BertClassifier(
        MODEL_NAME,
        num_classes
    ).to(device)

    criterion = FocalLoss(gamma=2.0)

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    for epoch in range(5):
        loss = train_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device
        )

        print(f"Epoch={epoch+1} Loss={loss:.4f}")
