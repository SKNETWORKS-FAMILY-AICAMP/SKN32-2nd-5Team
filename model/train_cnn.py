"""
train_cnn.py
────────────────────────────────────────────────────────────────────────────────
TextCNN 기반 이진 분류기  (churn_signal vs retain)
  - klue/bert-base 토크나이저 사용 (4개 모델 공통)
  - Focal Loss / BCE 선택 가능
  - 최적 F1 모델 가중치 → ./output/cnn_best.pt
  - 실험 결과          → ./output/cnn.csv  (또는 .json)
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from torch.optim import AdamW

from utils import (
    MAX_LEN, NUM_CLASSES, TOKENIZER_NAME,
    get_criterion, get_tokenizer, load_data, make_loaders,
    make_record, run_training, save_results, set_seed, test, TextDataset,
)


#################################################
# 모델
#################################################
class CNNClassifier(nn.Module):
    """
    TextCNN
    embedding → Conv1d × 3 (kernel 2/3/4) → ReLU → MaxPool → concat → Dropout → FC

    Parameters
    ----------
    vocab_size  : 토크나이저 어휘 크기
    num_classes : 출력 클래스 수 (2)
    embed_dim   : 임베딩 차원
    num_filters : 커널당 필터 수
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = NUM_CLASSES,
        embed_dim: int   = 128,
        num_filters: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 커널 크기 2, 3, 4 — 시퀀스 길이(50)에 맞게 설정
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in (2, 3, 4)
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(num_filters * 3, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        x = self.embedding(input_ids)          # (B, L, E)
        x = x.permute(0, 2, 1)                # (B, E, L)

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))            # (B, F, L')
            c = c.max(dim=2).values            # (B, F)
            pooled.append(c)

        x = torch.cat(pooled, dim=1)           # (B, F*3)
        x = self.dropout(x)
        return self.fc(x)                      # (B, num_classes)


#################################################
# 파이프라인
#################################################
def run(
    loss_type: str   = "focal",   # "focal" | "bce"
    alpha            = None,      # ex) [1.0, 2.0]
    gamma: float     = 2.0,
    embed_dim: int   = 128,
    num_filters: int = 128,
    batch_size: int  = 32,
    epochs: int      = 7,         # 4 ~ 10
    lr: float        = 1e-3,
    weight_decay: float = 0.01,
    result_fmt: str  = "csv",     # "csv" | "json"
    seed: int        = 42,
) -> None:
    # 0. 시드 고정
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CNN] Device: {device}  |  Loss: {loss_type}  |  Epochs: {epochs}")

    # 1. 데이터 로드
    (train_texts, train_labels), (valid_texts, valid_labels) = load_data()

    # 2. 토크나이저 & Dataset  (klue/bert-base 토크나이저 공통 사용)
    tokenizer = get_tokenizer(TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size

    train_ds = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    valid_ds = TextDataset(valid_texts, valid_labels, tokenizer, MAX_LEN)

    train_loader, valid_loader = make_loaders(train_ds, valid_ds, batch_size, seed)

    # 3. 모델
    model = CNNClassifier(
        vocab_size  = vocab_size,
        embed_dim   = embed_dim,
        num_filters = num_filters,
    ).to(device)

    # 4. Optimizer & Loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = get_criterion(loss_type, alpha=alpha, gamma=gamma, device=device)

    # 5. Train / Valid  (최적 가중치 자동 저장 포함)
    records = run_training(
        model       = model,
        train_loader = train_loader,
        valid_loader = valid_loader,
        optimizer   = optimizer,
        criterion   = criterion,
        device      = device,
        model_name  = "cnn",
        epochs      = epochs,
        loss_type   = loss_type,
        result_fmt  = result_fmt,
    )

    # 5-b. Test  (검증 세트로 최종 평가)
    print("\n── 최종 테스트 ──")
    churn_probs, pred_labels, test_f1, test_report, test_report_dict, test_time = test(
        model, valid_loader, device
    )
    print(f"  Test F1={test_f1:.4f} | Test Time={test_time:.2f}s")
    print(test_report)

    # 6. 테스트 결과를 실험 기록에 추가 후 재저장
    records.append(make_record(
        report_dict = test_report_dict,
        epoch       = "test",
        loss        = None,
        f1          = test_f1,
        train_time  = None,
        valid_time  = test_time,
    ))
    save_results(records, "cnn", fmt=result_fmt)

    # churn_signal 확률 미리보기
    print("churn_signal 확률 (상위 5개):", [f"{p:.4f}" for p in churn_probs[:5]])


if __name__ == "__main__":
    run(
        loss_type   = "focal",   # "focal" | "bce"
        gamma       = 2.0,
        embed_dim   = 128,
        num_filters = 128,
        batch_size  = 32,
        epochs      = 7,
        lr          = 1e-3,
        result_fmt  = "csv",
    )