from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer

#########################################################
# CONFIG
#########################################################

DATA_PATH = Path("../output/analysis_dataset.json")

REPORT_DIR = Path("../reports")
REPORT_PATH = REPORT_DIR / "preprocessing_report.md"
HIST_PNG = "token_length_distribution.png"
HIST_PATH = REPORT_DIR / HIST_PNG

TOKENIZER_NAME = "klue/bert-base"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

#########################################################
# LOAD DATA
#########################################################

print("Loading data...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

required_columns = [
    "dataset",
    "dialog_id",
    "seq",
    "label",
    "text"
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼 누락: {missing_columns}"
    )

#########################################################
# TOKENIZER
#########################################################

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
)

VOCAB_SIZE = tokenizer.vocab_size

#########################################################
# BASIC STATS
#########################################################

total_rows = len(df)

train_rows = len(
    df[df["dataset"] == "train"]
)

valid_rows = len(
    df[df["dataset"] == "valid"]
)

unique_dialogs = (
    df["dialog_id"]
    .nunique()
)

#########################################################
# MISSING
#########################################################

missing_df = (
    df.isnull()
      .sum()
      .reset_index()
)

missing_df.columns = [
    "column",
    "missing_count"
]

#########################################################
# DUPLICATE
#########################################################

duplicate_rows = (
    df.duplicated()
      .sum()
)

#########################################################
# LABEL DISTRIBUTION
#########################################################

label_dist = (
    df["label"]
    .value_counts()
    .reset_index()
)

label_dist.columns = [
    "label",
    "count"
]

label_dist["ratio(%)"] = (
    label_dist["count"]
    /
    total_rows
    * 100
).round(2)

#########################################################
# DIALOG STATS
#########################################################

dialog_turns = (
    df.groupby("dialog_id")
      .size()
)

dialog_stats = {
    "mean": round(dialog_turns.mean(), 2),
    "median": round(dialog_turns.median(), 2),
    "min": int(dialog_turns.min()),
    "max": int(dialog_turns.max())
}

#########################################################
# CHAR LENGTH
#########################################################

df["char_len"] = (
    df["text"]
    .astype(str)
    .apply(len)
)

char_stats = {
    "mean": round(df["char_len"].mean(), 2),
    "median": round(df["char_len"].median(), 2),
    "min": int(df["char_len"].min()),
    "max": int(df["char_len"].max()),
    "p90": round(df["char_len"].quantile(0.90), 2),
    "p95": round(df["char_len"].quantile(0.95), 2),
    "p99": round(df["char_len"].quantile(0.99), 2)
}

#########################################################
# TOKEN LENGTH
#########################################################

print("Calculating token lengths...")

def get_token_length(text):

    return len(
        tokenizer.encode(
            str(text),
            add_special_tokens=False
        )
    )

df["token_len"] = (
    df["text"]
    .astype(str)
    .apply(get_token_length)
)

token_stats = {
    "mean": round(df["token_len"].mean(), 2),
    "median": round(df["token_len"].median(), 2),
    "min": int(df["token_len"].min()),
    "max": int(df["token_len"].max()),
    "p90": round(df["token_len"].quantile(0.90), 2),
    "p95": round(df["token_len"].quantile(0.95), 2),
    "p99": round(df["token_len"].quantile(0.99), 2)
}

#########################################################
# RECOMMENDED MAX LENGTH
#########################################################

MAX_LEN = int(
    np.ceil(
        df["token_len"]
        .quantile(0.95)
    )
)

#########################################################
# EMPTY TEXT
#########################################################

empty_text_count = len(
    df[
        df["text"]
        .astype(str)
        .str.strip()
        == ""
    ]
)

#########################################################
# CLASS IMBALANCE
#########################################################

class_counts = (
    df["label"]
    .value_counts()
)

imbalance_ratio = round(
    class_counts.max()
    /
    class_counts.min(),
    2
)

#########################################################
# LONG SEQUENCE
#########################################################

long_sequence_count = len(
    df[
        df["token_len"]
        > MAX_LEN
    ]
)

#########################################################
# PSI
#########################################################

def calculate_psi(
    expected,
    actual,
    buckets=10
):

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(
        expected,
        np.arange(
            buckets + 1
        )
        / buckets
        * 100
    )

    psi = 0

    for i in range(
        len(breakpoints) - 1
    ):

        expected_pct = (
            (
                expected >= breakpoints[i]
            )
            &
            (
                expected < breakpoints[i+1]
            )
        ).mean()

        actual_pct = (
            (
                actual >= breakpoints[i]
            )
            &
            (
                actual < breakpoints[i+1]
            )
        ).mean()

        expected_pct = max(
            expected_pct,
            0.0001
        )

        actual_pct = max(
            actual_pct,
            0.0001
        )

        psi += (
            actual_pct
            -
            expected_pct
        ) * np.log(
            actual_pct
            /
            expected_pct
        )

    return round(float(psi), 4)

if (
    train_rows > 0
    and
    valid_rows > 0
):

    psi_token_length = calculate_psi(
        df[
            df["dataset"]
            == "train"
        ]["token_len"],
        df[
            df["dataset"]
            == "valid"
        ]["token_len"]
    )
else:
    psi_token_length = None

#########################################################
# HISTOGRAM
#########################################################

print("Saving histogram...")

plt.figure(figsize=(12, 6))

sns.histplot(
    df["token_len"],
    bins=50,
    kde=True
)

plt.axvline(
    token_stats["p90"],
    color="green",
    linestyle="--",
    label=f"P90={token_stats['p90']}"
)

plt.axvline(
    token_stats["p95"],
    color="orange",
    linestyle="--",
    label=f"P95={token_stats['p95']}"
)

plt.axvline(
    token_stats["p99"],
    color="red",
    linestyle="--",
    label=f"P99={token_stats['p99']}"
)

plt.title(
    "Token Length Distribution"
)

plt.xlabel(
    "Token Length"
)

plt.ylabel(
    "Count"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    HIST_PATH,
    dpi=300
)

plt.close()

#########################################################
# REPORT
#########################################################

report = f"""
# 데이터 전처리 결과서

## 1. 데이터 개요

원천 데이터 보호 정책에 따라 원문 데이터는 공개하지 않음.

|항목|값|
|---|---|
|전체 데이터|{total_rows}|
|Train|{train_rows}|
|Validation|{valid_rows}|
|Unique Dialog|{unique_dialogs}|

---

## 2. Label 분포

{label_dist.to_markdown(index=False)}

---

## 3. 결측치 분석

{missing_df.to_markdown(index=False)}

---

## 4. 중복 데이터

|항목|값|
|---|---|
|중복 행 수|{duplicate_rows}|

---

## 5. Dialog 분석

|항목|값|
|---|---|
|평균 Turn|{dialog_stats['mean']}|
|중앙값 Turn|{dialog_stats['median']}|
|최소 Turn|{dialog_stats['min']}|
|최대 Turn|{dialog_stats['max']}|

---

## 6. 문자 길이 분석

|항목|값|
|---|---|
|평균|{char_stats['mean']}|
|중앙값|{char_stats['median']}|
|최소|{char_stats['min']}|
|최대|{char_stats['max']}|
|P90|{char_stats['p90']}|
|P95|{char_stats['p95']}|
|P99|{char_stats['p99']}|

---

## 7. 토큰 길이 분석

Tokenizer : KLUE/BERT SentencePiece

|항목|값|
|---|---|
|평균|{token_stats['mean']}|
|중앙값|{token_stats['median']}|
|최소|{token_stats['min']}|
|최대|{token_stats['max']}|
|P90|{token_stats['p90']}|
|P95|{token_stats['p95']}|
|P99|{token_stats['p99']}|


![image]({HIST_PNG})

---

## 8. Train / Validation 분포 비교

|항목|PSI|
|---|---|
|Token Length|{psi_token_length}|

PSI 기준

- PSI < 0.1 : 안정
- 0.1 ~ 0.25 : 주의
- PSI > 0.25 : 분포 차이 큼

---

## 9. Sequence Length 결정

|항목|값|
|---|---|
|추천 Max Length|{MAX_LEN}|
|초과 샘플 수|{long_sequence_count}|

선정 기준 : P95

---

## 10. Vocabulary 정보

|항목|값|
|---|---|
|Tokenizer|KLUE/BERT|
|Tokenization|SentencePiece|
|Vocabulary Size|{VOCAB_SIZE}|

---

## 11. 데이터 품질 점검

|항목|값|
|---|---|
|Empty Text|{empty_text_count}|
|Duplicate Rows|{duplicate_rows}|
|Imbalance Ratio|{imbalance_ratio}|

---

## 12. 모델 입력 사양

공통 Tokenizer

KLUE SentencePiece

사용 모델

1. LSTM
2. Text CNN
3. Transformer Encoder (Scratch)
4. KLUE-BERT Fine-tuning

---

## 13. 생성 산출물

- preprocessing_report.md
- token_length_distribution.png

"""

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

#########################################################
# DONE
#########################################################

print()
print("완료")
print(f"Report : {REPORT_PATH}")
print(f"Histogram : {HIST_PATH}")
print()
print(f"Vocabulary Size : {VOCAB_SIZE}")
print(f"Recommended Max Length : {MAX_LEN}")
print(f"PSI : {psi_token_length}")