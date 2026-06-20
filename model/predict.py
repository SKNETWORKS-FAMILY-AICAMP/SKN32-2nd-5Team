"""
predict.py
────────────────────────────────────────────────────────────────────────────────
단건 문장 추론 스크립트
  - 저장된 4개 모델의 best 가중치를 불러와 단일 문장을 추론합니다.
  - logits → softmax → churn_signal 확률 출력
  - 사용법: python predict.py
            python predict.py --text "해지를 하는게 맞을지 아닌지 고민이되요"
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

#################################################
# utils 공통 상수 / 토크나이저
#################################################
from utils import (
    ID2LABEL, MAX_LEN, NUM_CLASSES, TOKENIZER_NAME,
    get_tokenizer, set_seed,
)

OUTPUT_DIR = Path("./output")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


#################################################
# 모델 정의  (train_*.py 와 완전히 동일 — 가중치 로드에 필요)
#################################################

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=NUM_CLASSES,
                 embed_dim=128, num_filters=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs     = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in (2, 3, 4)
        ])
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(num_filters * 3, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids).permute(0, 2, 1)
        pooled = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=NUM_CLASSES,
                 embed_dim=128, hidden_size=128, num_layers=2,
                 bidirectional=False, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            embed_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout   = nn.Dropout(dropout)
        dir_mult       = 2 if bidirectional else 1
        self.fc        = nn.Linear(hidden_size * dir_mult, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.lstm.bidirectional else h_n[-1]
        return self.fc(self.dropout(h))


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe    = torch.zeros(max_len, embed_dim)
        pos   = torch.arange(max_len).unsqueeze(1)
        denom = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=NUM_CLASSES,
                 embed_dim=128, nhead=4, num_layers=2,
                 ff_dim=256, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len, dropout)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        key_padding_mask = (input_ids == 0)
        x = self.pos_enc(self.embedding(input_ids))
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.fc(self.dropout(x[:, 0, :]))


class BertClassifier(nn.Module):
    def __init__(self, model_name=TOKENIZER_NAME,
                 num_classes=NUM_CLASSES, dropout=0.1, freeze_layers=0):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden_size  = self.bert.config.hidden_size
        if freeze_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(self.dropout(out.last_hidden_state[:, 0, :]))


#################################################
# 추론 유틸
#################################################

def encode_text(text: str, tokenizer, max_len: int = MAX_LEN, device: str = DEVICE):
    """
    단일 문장을 토크나이징해 (input_ids, attention_mask) 텐서를 반환합니다.
    shape: (1, max_len)
    """
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def load_model(model: nn.Module, weight_path: Path, device: str = DEVICE) -> nn.Module:
    """저장된 가중치를 모델에 로드합니다."""
    if not weight_path.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_single(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict:
    """
    단건 추론.

    Returns
    -------
    {
        "churn_prob"  : float,   # churn_signal 확률
        "retain_prob" : float,   # retain 확률
        "predicted"   : str,     # 최종 예측 라벨
        "logits"      : list,    # raw logits [retain, churn_signal]
    }
    """
    with torch.no_grad():
        logits = model(input_ids, attention_mask)           # (1, 2)
        probs  = F.softmax(logits, dim=1).squeeze(0)        # (2,)

    retain_prob = probs[0].item()
    churn_prob  = probs[1].item()
    predicted   = ID2LABEL[probs.argmax().item()]

    return {
        "churn_prob":  round(churn_prob,  4),
        "retain_prob": round(retain_prob, 4),
        "predicted":   predicted,
        "logits":      [round(v, 4) for v in logits.squeeze(0).cpu().tolist()],
    }


#################################################
# 메인
#################################################

def run(text: str) -> None:
    set_seed(42)

    tokenizer  = get_tokenizer(TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size

    # 입력 인코딩
    input_ids, attention_mask = encode_text(text, tokenizer)

    # 모델 설정  (train_*.py 와 동일한 하이퍼파라미터)
    models_cfg = [
        {
            "name":   "CNN",
            "model":  CNNClassifier(vocab_size=vocab_size),
            "weight": OUTPUT_DIR / "cnn_best.pt",
        },
        {
            "name":   "LSTM",
            "model":  LSTMClassifier(vocab_size=vocab_size),
            "weight": OUTPUT_DIR / "lstm_best.pt",
        },
        {
            "name":   "Transformer (Scratch)",
            "model":  TransformerClassifier(vocab_size=vocab_size),
            "weight": OUTPUT_DIR / "transformer_best.pt",
        },
        {
            "name":   "BERT (klue/bert-base)",
            "model":  BertClassifier(),
            "weight": OUTPUT_DIR / "bert_best.pt",
        },
    ]

    # 추론 및 출력
    print("=" * 60)
    print(f"입력 문장: \"{text}\"")
    print("=" * 60)

    for cfg in models_cfg:
        print(f"\n [{cfg['name']}]")
        try:
            model = load_model(cfg["model"], cfg["weight"])
            result = predict_single(model, input_ids, attention_mask)

            bar_churn  = "-" * int(result["churn_prob"]  * 20)
            bar_retain = "-" * int(result["retain_prob"] * 20)

            print(f"  예측 라벨    : {result['predicted']}")
            print(f"  churn_signal : {result['churn_prob']:.4f}  |{bar_churn:<20}|")
            print(f"  retain       : {result['retain_prob']:.4f}  |{bar_retain:<20}|")
            print(f"  logits       : retain={result['logits'][0]:.4f}, "
                  f"churn_signal={result['logits'][1]:.4f}")

        except FileNotFoundError as e:
            print(f"  [WARN]  {e}")
        except Exception as e:
            print(f"  [ERROR] 추론 실패: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단건 문장 churn 예측")
    parser.add_argument(
        "--text",
        type=str,
        default="해지를 하는게 맞을지 아닌지 고민이되요",
        help="추론할 문장 (기본값: '해지를 하는게 맞을지 아닌지 고민이되요')",
    )
    args = parser.parse_args()
    run(args.text)