"""
train_transformer.py
────────────────────────────────────────────────────────────────────────────────
Transformer Encoder (Scratch) 기반 이진 분류기  (churn_signal vs retain)
  - klue/bert-base 토크나이저 사용 (4개 모델 공통)
  - Focal Loss / BCE 선택 가능
  - 최적 F1 모델 가중치 → ./output/transformer_best.pt
  - 실험 결과          → ./output/transformer.csv  (또는 .json)
────────────────────────────────────────────────────────────────────────────────
"""

import math

import torch
import torch.nn as nn
from torch.optim import AdamW

from utils import (
    MAX_LEN, NUM_CLASSES, TOKENIZER_NAME,
    get_criterion, get_tokenizer, load_data, make_loaders,
    make_record, run_training, save_results, set_seed, test, TextDataset,
)


#################################################
# 위치 인코딩
#################################################
class PositionalEncoding(nn.Module):
    """
    고정 사인/코사인 Positional Encoding.
    dropout 적용 후 임베딩에 더해집니다.
    """

    def __init__(self, embed_dim: int, max_len: int = MAX_LEN, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe     = torch.zeros(max_len, embed_dim)          # (L, E)
        pos    = torch.arange(max_len).unsqueeze(1)        # (L, 1)
        denom  = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        self.register_buffer("pe", pe.unsqueeze(0))        # (1, L, E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


#################################################
# 모델
#################################################
class TransformerClassifier(nn.Module):
    """
    Transformer Encoder (처음부터 학습) 분류기.
    embedding + PosEnc → TransformerEncoder → CLS 토큰 → Dropout → FC

    Parameters
    ----------
    vocab_size   : 토크나이저 어휘 크기
    num_classes  : 출력 클래스 수 (2)
    embed_dim    : 임베딩 / Transformer hidden 차원 (nhead의 배수여야 함)
    nhead        : Multi-Head Attention 헤드 수
    num_layers   : Encoder 블록 수
    ff_dim       : FeedForward 은닉 차원
    max_len      : 최대 시퀀스 길이
    dropout      : 드롭아웃 비율
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = NUM_CLASSES,
        embed_dim: int   = 128,
        nhead: int       = 4,
        num_layers: int  = 2,
        ff_dim: int      = 256,
        max_len: int     = MAX_LEN,
        dropout: float   = 0.1,
    ):
        super().__init__()
        assert embed_dim % nhead == 0, "embed_dim은 nhead의 배수여야 합니다."

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer  = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = nhead,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,          # (B, L, E) 형식
            norm_first      = True,          # Pre-LN (학습 안정성↑)
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # 패딩 마스크: True인 위치를 무시 (padding_idx=0)
        key_padding_mask = (input_ids == 0)                 # (B, L)

        x = self.embedding(input_ids)                       # (B, L, E)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, E)

        # [CLS] 위치(index 0) 벡터 사용
        cls = x[:, 0, :]                                    # (B, E)
        cls = self.dropout(cls)
        return self.fc(cls)                                  # (B, num_classes)


#################################################
# 파이프라인
#################################################
def run(
    loss_type: str   = "focal",   # "focal" | "bce"
    alpha            = None,      # ex) [1.0, 2.0]
    gamma: float     = 2.0,
    embed_dim: int   = 128,       # nhead의 배수여야 함
    nhead: int       = 4,
    num_layers: int  = 2,
    ff_dim: int      = 256,
    dropout: float   = 0.1,
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
    print(f"[Transformer] Device: {device}  |  Loss: {loss_type}  |  Epochs: {epochs}")

    # 1. 데이터 로드
    (train_texts, train_labels), (valid_texts, valid_labels) = load_data()

    # 2. 토크나이저 & Dataset  (klue/bert-base 토크나이저 공통 사용)
    tokenizer  = get_tokenizer(TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size

    train_ds = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    valid_ds = TextDataset(valid_texts, valid_labels, tokenizer, MAX_LEN)

    train_loader, valid_loader = make_loaders(train_ds, valid_ds, batch_size, seed)

    # 3. 모델
    model = TransformerClassifier(
        vocab_size  = vocab_size,
        embed_dim   = embed_dim,
        nhead       = nhead,
        num_layers  = num_layers,
        ff_dim      = ff_dim,
        max_len     = MAX_LEN,
        dropout     = dropout,
    ).to(device)

    # 4. Optimizer & Loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = get_criterion(loss_type, alpha=alpha, gamma=gamma, device=device)

    # 5. Train / Valid  (최적 가중치 자동 저장 포함)
    records = run_training(
        model        = model,
        train_loader = train_loader,
        valid_loader = valid_loader,
        optimizer    = optimizer,
        criterion    = criterion,
        device       = device,
        model_name   = "transformer",
        epochs       = epochs,
        loss_type    = loss_type,
        result_fmt   = result_fmt,
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
    save_results(records, "transformer", fmt=result_fmt)

    # churn_signal 확률 미리보기
    print("churn_signal 확률 (상위 5개):", [f"{p:.4f}" for p in churn_probs[:5]])


if __name__ == "__main__":
    run(
        loss_type  = "focal",   # "focal" | "bce"
        gamma      = 2.0,
        embed_dim  = 128,       # nhead(4)의 배수
        nhead      = 4,
        num_layers = 2,
        ff_dim     = 256,
        dropout    = 0.1,
        batch_size = 32,
        epochs     = 7,
        lr         = 1e-3,
        result_fmt = "csv",
    )