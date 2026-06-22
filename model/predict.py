"""
predict.py
────────────────────────────────────────────────────────────────────────────────
문장 추론 스크립트
  - 저장된 4개 모델의 best 가중치를 불러와 단일 문장을 추론합니다.
  - logits → softmax → churn_signal 확률 출력
  - 사용법: python predict.py
            python predict.py --text "해지를 하는게 맞을지 아닌지 고민이되요"
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################
# 모델 정의  (train_*.py 와 완전히 동일 — 가중치 로드에 필요)
#################################################
from train_cnn import CNNClassifier
from train_lstm import LSTMClassifier
from train_transformer import TransformerClassifier
from train_bert import BertClassifier


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

def run(text_list: list[str]) -> list[dict]:
    set_seed(42)

    tokenizer  = get_tokenizer(TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size

    # 모델 설정  (train_*.py 와 동일한 하이퍼파라미터)
    models_cfg = [
        # {
        #     "name":   "CNN",
        #     "model":  CNNClassifier(vocab_size=vocab_size),
        #     "weight": OUTPUT_DIR / "cnn_best.pt",
        # },
        # {
        #     "name":   "LSTM",
        #     "model":  LSTMClassifier(vocab_size=vocab_size),
        #     "weight": OUTPUT_DIR / "lstm_best.pt",
        # },
        {
            "name":   "Transformer (Scratch)",
            "model":  TransformerClassifier(vocab_size=vocab_size),
            "weight": OUTPUT_DIR / "transformer_f1_best.pt",
        },
        # {
        #     "name":   "BERT (klue/bert-base)",
        #     "model":  BertClassifier(),
        #     "weight": OUTPUT_DIR / "bert_best.pt",
        # },
    ]

    # 추론 및 출력
    print("=" * 60)
    print(f"입력 문장: \"{text_list}\"")
    print("=" * 60)

    result_list = []
    for cfg in models_cfg:
        print(f"\n [{cfg['name']}]")
        try:
            model = load_model(cfg["model"], cfg["weight"])
            for text in text_list:
                # 입력 인코딩
                input_ids, attention_mask = encode_text(text, tokenizer)
                result = predict_single(model, input_ids, attention_mask)

                bar_churn  = "-" * int(result["churn_prob"]  * 20)
                bar_retain = "-" * int(result["retain_prob"] * 20)

                print(f"  예측 라벨    : {result['predicted']}")
                print(f"  churn_signal : {result['churn_prob']:.4f}  |{bar_churn:<20}|")
                print(f"  retain       : {result['retain_prob']:.4f}  |{bar_retain:<20}|")
                print(f"  logits       : retain={result['logits'][0]:.4f}, "
                      f"churn_signal={result['logits'][1]:.4f}")

                result['model_name'] = cfg['name']
                result['text'] = text
                result_list.append(result)
        except FileNotFoundError as e:
            print(f"  [WARN]  {e}")
        except Exception as e:
            print(f"  [ERROR] 추론 실패: {e}")

    print("\n" + "=" * 60)

    return result_list

def save_results(data, type_name):
    output_file = OUTPUT_DIR / f"predict_result_{type_name}.json"
    with output_file.open(
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="단건 문장 churn 예측")
    parser.add_argument(
        "--text",
        type=str,
        default="해지를 하는게 맞을지 아닌지 고민이되요",
        help="추론할 문장 (기본값: '해지를 하는게 맞을지 아닌지 고민이되요')",
    )
    args = parser.parse_args()
    predict_result_list = run([args.text])
    save_results(predict_result_list, "single")

    very_high_signals = [
        "이번 달까지만 보험 유지하고 해지하겠습니다.",
        "자동이체 해지 방법 알려주세요.",
        "더 이상 가입 유지할 이유가 없어서 계약 해지하려고 합니다.",
        "다른 보험 상품으로 갈아탈 예정이라 기존 계약 해지하려고 합니다.",
        "보험 해지하고 환급금 받고 싶습니다.",
        "납입 부담이 없어서 유지할 필요가 없어 해지하려고 합니다.",
        "보장 내용이 기대 이하라서 해지하려고 합니다.",
        "오늘 바로 보험 계약 해지 가능한가요?",
        "다음 보험료 결제 전에 해지 처리 부탁드립니다.",
        "보험 계약 해지 절차 어떻게 되나요?",
    ]

    predict_result_list = run(very_high_signals)
    save_results(predict_result_list, "very_high_signals")

    low_signals = [
        "보험 상품이 만족스러워서 계속 유지할 예정입니다.",
        "자동이체 그대로 유지해주세요.",
        "장기적으로 보험 유지할 계획입니다.",
        "최근 보장 내용 업데이트가 마음에 듭니다.",
        "주변에도 해당 보험 추천하고 있습니다.",
        "다음 달에도 계속 보험 유지할 생각입니다.",
        "보장 강화된 상품으로 업그레이드하려고 합니다.",
        "보장 금액이 점점 늘어나고 있습니다.",
        "보험 서비스에 매우 만족합니다.",
        "계약 갱신 진행 부탁드립니다.",
    ]

    predict_result_list = run(low_signals)
    save_results(predict_result_list, "low_signals")

    medium_signals = [
        "아직 결정은 못 했는데 보험 유지 여부 고민 중입니다.",
        "보험 상품을 생각보다 자주 확인하지는 않네요.",
        "보험료 대비 보장이 괜찮은지 모르겠습니다.",
        "한두 달 더 유지해보고 결정할 것 같습니다.",
        "보장 내용은 괜찮지만 보험료가 아쉽네요.",
        "기대했던 보험 혜택과 조금 다릅니다.",
        "보험 사용(청구) 빈도가 줄어들고 있습니다.",
        "최근 보험 만족도가 예전만 못합니다.",
        "다른 보험 상품을 찾아보고 있습니다.",
        "보험 갱신 시점에 다시 검토할 예정입니다.",
    ]

    predict_result_list = run(medium_signals)
    save_results(predict_result_list, "medium_signals")