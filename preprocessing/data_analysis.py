from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import pandas as pd
from transformers import AutoTokenizer

#################################################
# FONT (한글 깨짐 방지)
#################################################
_FONT_PATH = "../temp/NanumGothic.ttf"
KR_FONT    = fm.FontProperties(fname=_FONT_PATH)

def _apply_kr_font(ax):
    """ax의 title, xlabel, ylabel, tick label에 한글 폰트 일괄 적용"""
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontproperties(KR_FONT)

#################################################
# CONFIG
#################################################
DATA_PATH = Path("../output/analysis_dataset.json")
REPORT_DIR = Path("../reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = Path("../temp")

ANALYSIS_JSON       = REPORT_DIR / "analysis_result.json"
TOKEN_HIST_PNG      = REPORT_DIR / "token_length_distribution.png"
CHAR_HIST_PNG       = REPORT_DIR / "char_length_distribution.png"
LABEL_DIST_PNG      = REPORT_DIR / "label_distribution.png"
QUALITY_REPORT_PNG  = REPORT_DIR / "data_quality_report.png"

# CSV 저장 경로
DUP_TRAIN_CSV   = CSV_DIR / "duplicate_train.csv"    # Train 내부 중복 행
DUP_VALID_CSV   = CSV_DIR / "duplicate_valid.csv"    # Valid 내부 중복 행
DUP_CROSS_CSV   = CSV_DIR / "duplicate_cross.csv"    # Train ↔ Valid 교차 중복 행
EMPTY_TRAIN_CSV = CSV_DIR / "empty_text_train.csv"   # Train 빈 텍스트 행
EMPTY_VALID_CSV = CSV_DIR / "empty_text_valid.csv"   # Valid 빈 텍스트 행

TOKENIZER_NAME = "klue/bert-base"

#################################################
# 1. 데이터 로드
#################################################
print("=" * 50)
print("1. 데이터 로드")
print("=" * 50)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

raw_df = pd.DataFrame(raw_data)
print(f"  로드 완료: {len(raw_df):,}건")

#################################################
# 2. 전체 데이터 수
#################################################
print("\n2. 전체 데이터 수")

raw_train_df = raw_df[raw_df["dataset"] == "train"]
raw_valid_df = raw_df[raw_df["dataset"] == "valid"]
num_labels   = raw_df["label"].nunique()

print(f"  전체     : {len(raw_df):,}")
print(f"  Train    : {len(raw_train_df):,}")
print(f"  Valid    : {len(raw_valid_df):,}")
print(f"  라벨 수  : {num_labels}")

#################################################
# 3. 라벨 별 데이터 수 (중복 제거 전)
#################################################
print("\n3. 라벨 별 데이터 수 (중복 제거 전)")

def get_label_distribution(dataframe: pd.DataFrame) -> dict:
    result = {}
    counts = dataframe["label"].value_counts().sort_index()
    for label, count in counts.items():
        result[str(label)] = {
            "count": int(count),
            "ratio": round(count / len(dataframe) * 100, 2),
        }
    return result

label_dist_train_raw = get_label_distribution(raw_train_df)
label_dist_valid_raw = get_label_distribution(raw_valid_df)

print("  [Train]")
for label, info in label_dist_train_raw.items():
    print(f"    {label}: {info['count']:,}건 ({info['ratio']}%)")
print("  [Valid]")
for label, info in label_dist_valid_raw.items():
    print(f"    {label}: {info['count']:,}건 ({info['ratio']}%)")

# 라벨 분포 시각화 (중복 제거 전)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (split_df, title) in zip(axes, [(raw_train_df, "Train"), (raw_valid_df, "Valid")]):
    counts = split_df["label"].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4C72B0", edgecolor="white")
    ax.set_title(f"라벨 분포 — {title}", fontsize=13, fontweight="bold",
                 fontproperties=KR_FONT)
    ax.set_xlabel("라벨", fontproperties=KR_FONT)
    ax.set_ylabel("건수", fontproperties=KR_FONT)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    _apply_kr_font(ax)
plt.tight_layout()
plt.savefig(LABEL_DIST_PNG, dpi=300)
plt.close()
print(f"\n  [저장] {LABEL_DIST_PNG}")

#################################################
# 4. 중복 데이터 수 (라벨 & text)
#################################################
print("\n4. 중복 데이터 수 (label & text 기준)")

dup_train_mask = raw_train_df.duplicated(subset=["label", "text"], keep=False)
dup_valid_mask = raw_valid_df.duplicated(subset=["label", "text"], keep=False)

dup_train = int(raw_train_df.duplicated(subset=["label", "text"]).sum())
dup_valid = int(raw_valid_df.duplicated(subset=["label", "text"]).sum())
dup_total = int(raw_df.duplicated(subset=["label", "text"]).sum())

# 교차 중복: Train ↔ Valid 양쪽에 동시 존재하는 (label, text) 쌍
train_pairs     = set(zip(raw_train_df["label"], raw_train_df["text"]))
valid_pairs     = set(zip(raw_valid_df["label"], raw_valid_df["text"]))
cross_dup_pairs = train_pairs & valid_pairs
dup_cross       = len(cross_dup_pairs)

print(f"  Train 내부 중복         : {dup_train:,}건")
print(f"  Valid 내부 중복         : {dup_valid:,}건")
print(f"  Train ↔ Valid 교차 중복  : {dup_cross:,}건")
print(f"  전체 (합산 검증)         : {dup_train + dup_valid + dup_cross:,}건  ←→  raw 전체 중복: {dup_total:,}건")

# ── CSV 저장: 중복 행 전체 (keep=False → 원본·중복본 모두 포함) ──────────
dup_train_df = raw_train_df[dup_train_mask].sort_values(["label", "text"])
dup_valid_df = raw_valid_df[dup_valid_mask].sort_values(["label", "text"])

cross_mask_train = raw_train_df.apply(
    lambda row: (row["label"], row["text"]) in cross_dup_pairs, axis=1
)
cross_mask_valid = raw_valid_df.apply(
    lambda row: (row["label"], row["text"]) in cross_dup_pairs, axis=1
)
dup_cross_df = pd.concat(
    [raw_train_df[cross_mask_train], raw_valid_df[cross_mask_valid]],
    ignore_index=True
).sort_values(["label", "text", "dataset"])

dup_train_df.to_csv(DUP_TRAIN_CSV, index=False, encoding="utf-8-sig")
dup_valid_df.to_csv(DUP_VALID_CSV, index=False, encoding="utf-8-sig")
dup_cross_df.to_csv(DUP_CROSS_CSV, index=False, encoding="utf-8-sig")
print(f"\n  [저장] {DUP_TRAIN_CSV}  ({len(dup_train_df):,}행)")
print(f"  [저장] {DUP_VALID_CSV}  ({len(dup_valid_df):,}행)")
print(f"  [저장] {DUP_CROSS_CSV}  ({len(dup_cross_df):,}행)")

#################################################
# 5. 중복 제거 후 데이터 수
#################################################
print("\n5. 중복 제거 후 데이터 수")

# 1단계: 각 split 내부 중복 제거
df       = raw_df.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
train_df = df[df["dataset"] == "train"].copy()
valid_df = df[df["dataset"] == "valid"].copy()

# 2단계: 교차 중복 제거 (Valid 우선 보존 → Train에서 제거)
cross_mask = train_df.set_index(["label", "text"]).index.isin(
    valid_df.set_index(["label", "text"]).index
)
train_df = train_df[~cross_mask].reset_index(drop=True)

print(f"  전체  : {len(train_df) + len(valid_df):,}건")
print(f"  Train : {len(train_df):,}건  (교차 중복 {dup_cross:,}건 추가 제거)")
print(f"  Valid : {len(valid_df):,}건")

# 중복 제거 후 라벨 분포
label_dist_train = get_label_distribution(train_df)
label_dist_valid = get_label_distribution(valid_df)

print("\n  [중복 제거 후 라벨 분포 — Train]")
for label, info in label_dist_train.items():
    print(f"    {label}: {info['count']:,}건 ({info['ratio']}%)")
print("  [중복 제거 후 라벨 분포 — Valid]")
for label, info in label_dist_valid.items():
    print(f"    {label}: {info['count']:,}건 ({info['ratio']}%)")

#################################################
# 6. 문장 길이 (char)
#################################################
print("\n6. 문장 길이 (Character 기준)")

train_df["char_len"] = train_df["text"].astype(str).apply(len)
valid_df["char_len"] = valid_df["text"].astype(str).apply(len)

def get_length_stats(series: pd.Series) -> dict:
    return {
        "mean"  : round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "std"   : round(float(series.std()), 2),
        "min"   : int(series.min()),
        "max"   : int(series.max()),
        "p90"   : round(float(series.quantile(0.90)), 2),
        "p95"   : round(float(series.quantile(0.95)), 2),
        "p99"   : round(float(series.quantile(0.99)), 2),
    }

char_stats_train = get_length_stats(train_df["char_len"])
char_stats_valid = get_length_stats(valid_df["char_len"])

for split, stats in [("Train", char_stats_train), ("Valid", char_stats_valid)]:
    print(f"  [{split}] mean={stats['mean']}, median={stats['median']}, "
          f"std={stats['std']}, min={stats['min']}, max={stats['max']}, "
          f"p90={stats['p90']}, p95={stats['p95']}, p99={stats['p99']}")

# 문장 길이 히스토그램
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (split_df, label) in zip(axes, [(train_df, "Train"), (valid_df, "Valid")]):
    ax.hist(split_df["char_len"], bins=50, color="#55A868", edgecolor="white")
    ax.set_title(f"문장 길이 분포 (char) — {label}", fontsize=13, fontweight="bold",
                 fontproperties=KR_FONT)
    ax.set_xlabel("문자 수", fontproperties=KR_FONT)
    ax.set_ylabel("건수", fontproperties=KR_FONT)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    stats = get_length_stats(split_df["char_len"])
    textstr = (f"mean={stats['mean']}\nmedian={stats['median']}\n"
               f"p95={stats['p95']}\nmax={stats['max']}")
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
            fontproperties=KR_FONT)
    _apply_kr_font(ax)
plt.tight_layout()
plt.savefig(CHAR_HIST_PNG, dpi=300)
plt.close()
print(f"  [저장] {CHAR_HIST_PNG}")

#################################################
# 7. 토크나이징 후 토큰 수
#################################################
print("\n7. 토크나이징 후 토큰 수")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def get_token_length(text: str) -> int:
    return len(tokenizer.encode(str(text), add_special_tokens=False))

train_df["token_len"] = train_df["text"].astype(str).apply(get_token_length)
valid_df["token_len"] = valid_df["text"].astype(str).apply(get_token_length)

token_stats_train = get_length_stats(train_df["token_len"])
token_stats_valid = get_length_stats(valid_df["token_len"])

for split, stats in [("Train", token_stats_train), ("Valid", token_stats_valid)]:
    print(f"  [{split}] mean={stats['mean']}, median={stats['median']}, "
          f"std={stats['std']}, min={stats['min']}, max={stats['max']}, "
          f"p90={stats['p90']}, p95={stats['p95']}, p99={stats['p99']}")

# 토큰 길이 히스토그램
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (split_df, label) in zip(axes, [(train_df, "Train"), (valid_df, "Valid")]):
    ax.hist(split_df["token_len"], bins=50, color="#C44E52", edgecolor="white")
    ax.set_title(f"토큰 수 분포 — {label}", fontsize=13, fontweight="bold",
                 fontproperties=KR_FONT)
    ax.set_xlabel("토큰 수", fontproperties=KR_FONT)
    ax.set_ylabel("건수", fontproperties=KR_FONT)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    stats = get_length_stats(split_df["token_len"])
    textstr = (f"mean={stats['mean']}\nmedian={stats['median']}\n"
               f"p95={stats['p95']}\nmax={stats['max']}")
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
            fontproperties=KR_FONT)
    _apply_kr_font(ax)
plt.tight_layout()
plt.savefig(TOKEN_HIST_PNG, dpi=300)
plt.close()
print(f"  [저장] {TOKEN_HIST_PNG}")

#################################################
# 8. Vocab 정보
#################################################
print("\n8. Vocab 정보")
print(f"  Tokenizer   : {TOKENIZER_NAME}")
print(f"  Class       : {tokenizer.__class__.__name__}")
print(f"  Vocab Size  : {tokenizer.vocab_size:,}")

vocab_info = {
    "name"      : TOKENIZER_NAME,
    "class"     : tokenizer.__class__.__name__,
    "vocab_size": tokenizer.vocab_size,
}

#################################################
# 9. 데이터 품질 점검
#################################################
print("\n9. 데이터 품질 점검")

# 빈 텍스트
empty_train_mask = train_df["text"].astype(str).str.strip() == ""
empty_valid_mask = valid_df["text"].astype(str).str.strip() == ""
empty_train = int(empty_train_mask.sum())
empty_valid = int(empty_valid_mask.sum())

# 빈 텍스트 행 CSV 저장
empty_train_df = train_df[empty_train_mask].copy()
empty_valid_df = valid_df[empty_valid_mask].copy()
empty_train_df.to_csv(EMPTY_TRAIN_CSV, index=False, encoding="utf-8-sig")
empty_valid_df.to_csv(EMPTY_VALID_CSV, index=False, encoding="utf-8-sig")
print(f"  [빈 텍스트] Train: {empty_train:,}건  →  {EMPTY_TRAIN_CSV}")
print(f"  [빈 텍스트] Valid: {empty_valid:,}건  →  {EMPTY_VALID_CSV}")

# 매우 짧은 텍스트 (char 기준 5자 이하)
SHORT_THRESHOLD = 5
short_train = int((train_df["char_len"] <= SHORT_THRESHOLD).sum())
short_valid = int((valid_df["char_len"] <= SHORT_THRESHOLD).sum())

# 매우 긴 텍스트 (token 기준 512 초과 → BERT 최대 길이)
LONG_THRESHOLD = 512
long_train = int((train_df["token_len"] > LONG_THRESHOLD).sum())
long_valid = int((valid_df["token_len"] > LONG_THRESHOLD).sum())

# 결측값
null_train = int(train_df["text"].isnull().sum())
null_valid = int(valid_df["text"].isnull().sum())

print(f"  [짧은 텍스트 ≤{SHORT_THRESHOLD}] Train: {short_train:,}, Valid: {short_valid:,}")
print(f"  [긴 텍스트  >{LONG_THRESHOLD}]  Train: {long_train:,}, Valid: {long_valid:,}")
print(f"  [결측값]          Train: {null_train:,}, Valid: {null_valid:,}")

# 품질 점검 시각화
categories   = ["빈 텍스트", f"짧은 텍스트\n(≤{SHORT_THRESHOLD}자)", f"긴 텍스트\n(>{LONG_THRESHOLD} tokens)", "결측값"]
train_values = [empty_train, short_train, long_train, null_train]
valid_values = [empty_valid, short_valid, long_valid, null_valid]

x = range(len(categories))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar([i - w / 2 for i in x], train_values, width=w, label="Train", color="#4C72B0")
b2 = ax.bar([i + w / 2 for i in x], valid_values, width=w, label="Valid",  color="#DD8452")
ax.set_title("데이터 품질 점검", fontsize=13, fontweight="bold",
             fontproperties=KR_FONT)
ax.set_xticks(list(x))
ax.set_xticklabels(categories, fontproperties=KR_FONT)
ax.set_ylabel("건수", fontproperties=KR_FONT)
ax.legend(prop=KR_FONT)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
_apply_kr_font(ax)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{int(h):,}",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(QUALITY_REPORT_PNG, dpi=300)
plt.close()
print(f"  [저장] {QUALITY_REPORT_PNG}")

quality_info = {
    "empty_text" : {"train": empty_train, "valid": empty_valid},
    "short_text" : {"threshold": SHORT_THRESHOLD, "train": short_train, "valid": short_valid},
    "long_text"  : {"threshold": LONG_THRESHOLD,  "train": long_train,  "valid": long_valid},
    "null_text"  : {"train": null_train, "valid": null_valid},
}

#################################################
# SAVE analysis_result.json
#################################################
result = {
    "dataset_count": {
        "raw": {
            "total"     : len(raw_df),
            "train"     : len(raw_train_df),
            "valid"     : len(raw_valid_df),
        },
        "num_labels": num_labels,
        "after_dedup": {
            "total": len(train_df) + len(valid_df),
            "train": len(train_df),
            "valid": len(valid_df),
        },
    },
    "label_distribution": {
        "before_dedup": {
            "train": label_dist_train_raw,
            "valid": label_dist_valid_raw,
        },
        "after_dedup": {
            "train": label_dist_train,
            "valid": label_dist_valid,
        },
    },
    "duplicate": {
        "total" : dup_total,
        "train" : dup_train,
        "valid" : dup_valid,
        "cross" : dup_cross,
    },
    "char_length": {
        "train": char_stats_train,
        "valid": char_stats_valid,
    },
    "token_length": {
        "train": token_stats_train,
        "valid": token_stats_valid,
    },
    "tokenizer"   : vocab_info,
    "data_quality": quality_info,
    "saved_csv": {
        "duplicate_train" : str(DUP_TRAIN_CSV),
        "duplicate_valid" : str(DUP_VALID_CSV),
        "duplicate_cross" : str(DUP_CROSS_CSV),
        "empty_text_train": str(EMPTY_TRAIN_CSV),
        "empty_text_valid": str(EMPTY_VALID_CSV),
    },
}

with open(ANALYSIS_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("\n분석 완료")
print(f"결과 저장: {ANALYSIS_JSON}")