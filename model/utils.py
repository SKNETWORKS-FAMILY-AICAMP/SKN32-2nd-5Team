"""
utils.py
────────────────────────────────────────────────────────────────────────────────
공통 유틸리티
  - 시드 고정
  - 데이터 로드
  - Dataset (klue/bert 토크나이저 공용)
  - FocalLoss / BCEWithLogitsLoss 선택
  - train_epoch / validate / test 루프
  - 성능 평가 (classification_report, F1, 시간)
  - 실험 결과 저장 (CSV / JSON)
  - 최적 모델 가중치 저장
────────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

#################################################
# 상수
#################################################
LABEL2ID    = {"retain": 0, "churn_signal": 1}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 2
MAX_LEN     = 50
DATA_DIR    = Path("../output")
OUTPUT_DIR  = Path("./output")
TOKENIZER_NAME = "klue/bert-base"
SEED        = 42


#################################################
# 시드 고정
#################################################
def set_seed(seed: int = SEED) -> None:
    """모든 랜덤 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


#################################################
# 데이터 로드
#################################################
def load_data(data_dir: Path = DATA_DIR):
    """
    clean_train.csv / clean_valid.csv 로드.
    CSV 필수 컬럼: 'text', 'label' (문자열: 'retain' | 'churn_signal')

    Returns
    -------
    (train_texts, train_labels), (valid_texts, valid_labels)
    """
    train_df = pd.read_csv(data_dir / "clean_train.csv")
    valid_df = pd.read_csv(data_dir / "clean_valid.csv")

    def _parse(df):
        texts  = df["text"].tolist()
        labels = [LABEL2ID[l] for l in df["label"].tolist()]
        return texts, labels

    return _parse(train_df), _parse(valid_df)


#################################################
# Dataset
#################################################
class TextDataset(Dataset):
    """
    klue/bert 토크나이저를 사용하는 범용 Dataset.
    CNN / LSTM / Scratch Transformer / BERT 모두 동일하게 사용합니다.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = MAX_LEN,
    ):
        self.labels     = labels
        self.max_length = max_length
        self.encodings  = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_tokenizer(tokenizer_name: str = TOKENIZER_NAME):
    """klue/bert-base 토크나이저를 반환합니다."""
    return AutoTokenizer.from_pretrained(tokenizer_name)


def make_loaders(
    train_ds: Dataset,
    valid_ds: Dataset,
    batch_size: int = 32,
    seed: int = SEED,
):
    """DataLoader 쌍을 생성합니다."""
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, valid_loader


#################################################
# Loss
#################################################
class FocalLoss(nn.Module):
    """
    Focal Loss (Cross-Entropy 기반, 다중 클래스 지원).

    Parameters
    ----------
    alpha  : 클래스 가중치 Tensor 또는 None
    gamma  : focusing parameter (default 2.0)
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        pt   = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def get_criterion(
    loss_type: str  = "focal",
    alpha           = None,       # ex) [1.0, 2.0]  → retain 1배, churn 2배
    gamma: float    = 2.0,
    device: str     = "cpu",
) -> nn.Module:
    """
    loss_type : "focal" | "bce"
      - "focal" : FocalLoss (다중 클래스 CE 기반)
      - "bce"   : BCEWithLogitsLoss (이진, logits[:, 1] 기준)

    alpha 예시
      focal → [retain_weight, churn_weight]  (길이 2)
      bce   → [retain_weight, churn_weight]  (pos_weight = churn/retain)
    """
    assert loss_type in ("focal", "bce"), "loss_type은 'focal' 또는 'bce'여야 합니다."

    if loss_type == "focal":
        w = torch.tensor(alpha, dtype=torch.float, device=device) if alpha else None
        return FocalLoss(alpha=w, gamma=gamma)

    # BCE
    if alpha:
        pos_w = torch.tensor([alpha[1] / alpha[0]], dtype=torch.float, device=device)
    else:
        pos_w = None
    return nn.BCEWithLogitsLoss(pos_weight=pos_w)


#################################################
# 학습 루프
#################################################
def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    criterion,
    device: str,
    loss_type: str = "focal",
) -> tuple[float, float]:
    """
    1 에폭 학습.

    Returns
    -------
    (avg_loss, elapsed_seconds)
    """
    model.train()
    total_loss = 0.0
    t0         = time.time()

    for batch in tqdm(loader, desc="  Train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        if loss_type == "bce":
            loss = criterion(logits[:, 1].float(), labels.float())
        else:
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader), time.time() - t0


#################################################
# 검증 루프
#################################################
def validate(
    model,
    loader: DataLoader,
    criterion,
    device: str,
    loss_type: str = "focal",
) -> tuple[float, float, str, dict, float]:
    """
    검증 루프.

    Returns
    -------
    (val_loss, f1_weighted, classification_report_str, report_dict, elapsed_seconds)

    report_dict 구조 예시:
        {
            "retain":        {"precision": 0.91, "recall": 0.88, "f1-score": 0.89},
            "churn_signal":  {"precision": 0.87, "recall": 0.90, "f1-score": 0.88},
            "weighted avg":  {"precision": ...,  "recall": ...,  "f1-score": ...},
        }
    """
    model.eval()
    preds, targets = [], []
    total_loss     = 0.0
    t0             = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Valid", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            if loss_type == "bce":
                loss = criterion(logits[:, 1].float(), labels.float())
            else:
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    elapsed     = time.time() - t0
    val_loss    = total_loss / len(loader)
    f1          = f1_score(targets, preds, average="weighted")
    report_str  = classification_report(
        targets, preds,
        target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)],
        digits=4,
    )
    report_dict = classification_report(
        targets, preds,
        target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)],
        output_dict=True,
    )
    return val_loss, f1, report_str, report_dict, elapsed


#################################################
# 테스트 루프
#################################################
def test(
    model,
    loader: DataLoader,
    device: str,
) -> tuple[list[float], list[int], float, str, dict, float]:
    """
    테스트 루프.
    logits → softmax → churn_signal(index=1) 확률을 반환합니다.

    Returns
    -------
    (churn_probs, pred_labels, f1_weighted, report_str, report_dict, elapsed_seconds)
    """
    model.eval()
    churn_probs, preds, targets = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Test ", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits     = model(input_ids, attention_mask)
            probs      = F.softmax(logits, dim=1)          # (B, 2)
            churn_prob = probs[:, 1]                        # churn_signal 확률

            churn_probs.extend(churn_prob.cpu().numpy().tolist())
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

    elapsed     = time.time() - t0
    f1          = f1_score(targets, preds, average="weighted")
    report_str  = classification_report(
        targets, preds,
        target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)],
        digits=4,
    )
    report_dict = classification_report(
        targets, preds,
        target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)],
        output_dict=True,
    )
    return churn_probs, preds, f1, report_str, report_dict, elapsed


#################################################
# 결과 저장
#################################################
def save_results(records: list[dict], model_name: str, fmt: str = "csv") -> None:
    """
    에폭별 실험 결과를 CSV 또는 JSON으로 저장합니다.

    Parameters
    ----------
    records    : 에폭마다 append 한 dict 리스트
    model_name : 파일명에 사용 (확장자 제외)
    fmt        : "csv" | "json"
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{model_name}.{fmt}"

    if fmt == "csv":
        pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"실험 결과 저장: {path}")


def save_best_model(model, model_name: str) -> None:
    """최적 모델 가중치를 ./output/{model_name}_best.pt 로 저장합니다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{model_name}_best.pt"
    torch.save(model.state_dict(), path)
    print(f"최적 모델 저장: {path}")


#################################################
# record 생성 헬퍼
#################################################
def make_record(
    report_dict: dict,
    epoch,                    # int 또는 "test"
    train_loss: float | None,
    valid_loss: float | None,
    f1: float,
    train_time: float | None,
    valid_time: float,
) -> dict:
    """
    에폭(또는 테스트) 결과를 단일 dict로 변환합니다.
    run_training 내부와 train_*.py 테스트 블록에서 공통으로 사용합니다.

    저장 컬럼
    ---------
    epoch, train_loss, valid_loss, f1_weighted, train_time, valid_time,
    retain_precision, retain_recall, retain_f1,
    churn_signal_precision, churn_signal_recall, churn_signal_f1,
    weighted avg_precision, weighted avg_recall, weighted avg_f1
    """
    def _flat(label: str) -> dict:
        m = report_dict.get(label, {})
        return {
            f"{label}_precision": round(m.get("precision", 0.0), 4),
            f"{label}_recall":    round(m.get("recall",    0.0), 4),
            f"{label}_f1":        round(m.get("f1-score",  0.0), 4),
        }

    return {
        "epoch":       epoch,
        "train_loss":  round(train_loss, 6) if train_loss is not None else None,
        "valid_loss":  round(valid_loss, 6) if valid_loss is not None else None,
        "f1_weighted": round(f1, 6),
        "train_time":  round(train_time, 2) if train_time is not None else None,
        "valid_time":  round(valid_time, 2),
        **_flat("retain"),
        **_flat("churn_signal"),
        **_flat("weighted avg"),
    }


#################################################
# 공통 학습 드라이버
#################################################
def run_training(
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer,
    criterion,
    device: str,
    model_name: str,
    epochs: int     = 7,          # 4-10 사이 권장
    loss_type: str  = "focal",
    result_fmt: str = "csv",      # "csv" | "json"
) -> list[dict]:
    """
    Train / Valid 루프 공통 드라이버.
    최고 F1 달성 시 모델 가중치를 자동 저장합니다.

    Returns
    -------
    records : 에폭별 결과 dict 리스트
    """
    best_f1  = -1.0
    records  = []

    for epoch in range(1, epochs + 1):
        print(f"\n{'─'*60}")
        print(f"[{model_name}] Epoch {epoch}/{epochs}")

        # Train
        avg_loss, train_time = train_epoch(
            model, train_loader, optimizer, criterion, device, loss_type
        )

        # Valid
        val_loss, f1, report_str, report_dict, valid_time = validate(
            model, valid_loader, criterion, device, loss_type
        )

        print(
            f"  Train Loss={avg_loss:.4f} | Valid Loss={val_loss:.4f} | F1={f1:.4f} | "
            f"Train {train_time:.1f}s | Valid {valid_time:.1f}s"
        )
        print(report_str)

        records.append(make_record(
            report_dict = report_dict,
            epoch       = epoch,
            train_loss  = avg_loss,
            valid_loss  = val_loss,
            f1          = f1,
            train_time  = train_time,
            valid_time  = valid_time,
        ))

        # 최적 가중치 저장
        if f1 > best_f1:
            best_f1 = f1
            save_best_model(model, model_name)

    save_results(records, model_name, fmt=result_fmt)
    return records