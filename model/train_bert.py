"""
train_bert.py
────────────────────────────────────────────────────────────────────────────────
klue/bert-base Fine-tuning 이진 분류기  (churn_signal vs retain)
  - klue/bert-base 토크나이저 사용 (4개 모델 공통)
  - Focal Loss / BCE 선택 가능
  - 최적 F1 모델 가중치 → ./output/bert_best.pt
  - 실험 결과          → ./output/bert.csv  (또는 .json)
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel

from utils import (
    MAX_LEN, NUM_CLASSES, TOKENIZER_NAME,
    get_criterion, get_tokenizer, load_data, make_loaders,
    make_record, run_training, save_results, set_seed, test, TextDataset,
)


#################################################
# 모델
#################################################
class BertClassifier(nn.Module):
    """
    klue/bert-base Fine-tuning 분류기.
    BERT → [CLS] → Dropout → FC

    Parameters
    ----------
    model_name  : HuggingFace 모델 이름 (기본 "klue/bert-base")
    num_classes : 출력 클래스 수 (2)
    dropout     : 드롭아웃 비율
    freeze_layers : BERT 인코더 레이어 중 동결할 앞부분 수 (0=전체 학습)
    """

    def __init__(
        self,
        model_name: str  = TOKENIZER_NAME,
        num_classes: int = NUM_CLASSES,
        dropout: float   = 0.1,
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden_size  = self.bert.config.hidden_size         # 768

        # 일부 레이어 동결 (선택)
        if freeze_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]     # (B, H)
        cls_output = self.dropout(cls_output)
        return self.fc(cls_output)                           # (B, num_classes)


#################################################
# 파이프라인
#################################################
def run(
    loss_type: str    = "focal",   # "focal" | "bce"
    alpha             = None,      # ex) [1.0, 2.0]
    gamma: float      = 2.0,
    dropout: float    = 0.1,
    freeze_layers: int = 0,        # BERT 레이어 동결 수 (0=전체 파인튜닝)
    batch_size: int   = 16,        # BERT는 메모리 부담이 크므로 16 권장
    epochs: int       = 5,         # 4 ~ 10  (BERT는 3~5 에폭이 일반적)
    lr: float         = 2e-5,      # BERT 파인튜닝 표준 lr
    weight_decay: float = 0.01,
    result_fmt: str   = "csv",     # "csv" | "json"
    seed: int         = 42,
) -> None:
    # 0. 시드 고정
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BERT] Device: {device}  |  Loss: {loss_type}  |  Epochs: {epochs}")

    # 1. 데이터 로드
    (train_texts, train_labels), (valid_texts, valid_labels) = load_data()

    # 2. 토크나이저 & Dataset  (klue/bert-base 토크나이저 공통 사용)
    tokenizer = get_tokenizer(TOKENIZER_NAME)

    train_ds = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    valid_ds = TextDataset(valid_texts, valid_labels, tokenizer, MAX_LEN)

    train_loader, valid_loader = make_loaders(train_ds, valid_ds, batch_size, seed)

    # 3. 모델
    model = BertClassifier(
        model_name    = TOKENIZER_NAME,
        dropout       = dropout,
        freeze_layers = freeze_layers,
    ).to(device)

    # 4. Optimizer & Loss
    #    BERT는 레이어별 lr decay 적용 (선택)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped, lr=lr)
    criterion = get_criterion(loss_type, alpha=alpha, gamma=gamma, device=device)

    # 5. Train / Valid  (최적 가중치 자동 저장 포함)
    records = run_training(
        model        = model,
        train_loader = train_loader,
        valid_loader = valid_loader,
        optimizer    = optimizer,
        criterion    = criterion,
        device       = device,
        model_name   = "bert",
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
        train_loss  = None,
        valid_loss  = None,
        f1          = test_f1,
        train_time  = None,
        valid_time  = test_time,
    ))
    save_results(records, "bert", fmt=result_fmt)

    # churn_signal 확률 미리보기
    print("churn_signal 확률 (상위 5개):", [f"{p:.4f}" for p in churn_probs[:5]])


if __name__ == "__main__":
    run(
        loss_type     = "focal",   # "focal" | "bce"
        gamma         = 2.0,
        dropout       = 0.1,
        freeze_layers = 0,         # 0 = 전체 파인튜닝
        batch_size    = 16,
        epochs        = 4,
        lr            = 2e-5,
        result_fmt    = "csv",
    )