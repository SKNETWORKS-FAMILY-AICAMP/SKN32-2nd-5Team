from __future__ import annotations
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
# CONFIG (필요 시 외부에서 오버라이드 가능)
#################################################
DATA_PATH  = Path("../output/analysis_dataset.json")
REPORT_DIR = Path("../reports/data-preprocessing")
CSV_DIR    = Path("../output")

TOKENIZER_NAME  = "klue/bert-base"
_FONT_PATH = "../temp/NanumGothic.ttf"

#################################################
# 내부 유틸리티
#################################################
def _get_kr_font() -> fm.FontProperties:
    return fm.FontProperties(fname=_FONT_PATH)


def _apply_kr_font(ax: plt.Axes, font: fm.FontProperties) -> None:
    """ax의 모든 텍스트 요소에 한글 폰트 일괄 적용"""
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontproperties(font)


def _get_label_distribution(dataframe: pd.DataFrame) -> dict:
    result = {}
    counts = dataframe["label"].value_counts().sort_index()
    for label, count in counts.items():
        result[str(label)] = {
            "count": int(count),
            "ratio": round(count / len(dataframe) * 100, 2),
        }
    return result


def _get_length_stats(series: pd.Series) -> dict:
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


#################################################
# 단계별 함수
#################################################

def step1_load_and_clean(data_path: Path, report_dir: Path, csv_dir: Path = CSV_DIR) -> tuple[pd.DataFrame, dict]:
    """
    1. 데이터 로드
       1-1. 결측치 & 빈 텍스트 확인
       1-2. 텍스트 정제 (None→"" / strip / 빈 텍스트 제거)

    Returns
    -------
    df : 정제된 전체 DataFrame
    cleaning_info : 정제 통계 dict
    """
    print("=" * 50)
    print("1. 데이터 로드 및 텍스트 정제")
    print("=" * 50)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    print(f"  로드 완료: {len(df):,}건")

    # 1-1. 결측치 & 빈 텍스트 확인 (정제 전)
    print("\n  [1-1. 데이터 확인]")
    print(f"  shape    : {df.shape}")
    print(f"  columns  : {list(df.columns)}")
    null_counts = df.isnull().sum()
    print(f"  결측치 현황:\n{null_counts.to_string()}")
    empty_before = int((df["text"].astype(str).str.strip() == "").sum())
    print(f"  빈 텍스트 (strip 후 == \"\"): {empty_before:,}건")

    # 1-2. 텍스트 정제
    print("\n  [1-2. 텍스트 정제]")
    none_count  = int(df["text"].isnull().sum())
    df["text"]  = df["text"].fillna("")          # ① None → ""
    df["text"]  = df["text"].astype(str).str.strip()  # ② 앞뒤 공백 제거

    empty_mask  = df["text"] == ""               # ③ 빈 텍스트 탐지
    empty_count = int(empty_mask.sum())
    removed_csv = csv_dir / "empty_text_removed.csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    df[empty_mask].to_csv(removed_csv, index=False, encoding="utf-8-sig")
    df = df[~empty_mask].reset_index(drop=True)  # ③ 제거

    print(f"  ① None → \"\"       : {none_count:,}건 처리")
    print(f"  ② 앞뒤 공백 제거   : 전체 적용")
    print(f"  ③ 빈 텍스트 제거   : {empty_count:,}건 → {len(df):,}건 남음")
    print(f"     [저장] {removed_csv}")

    cleaning_info = {
        "before_total"  : len(df) + empty_count,
        "none_count"    : none_count,
        "empty_removed" : empty_count,
        "after_total"   : len(df),
        "removed_csv"   : str(removed_csv),
    }
    return df, cleaning_info


def step2_count(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    2. 전체 데이터 수 (정제 후 기준)

    Returns
    -------
    train_raw, valid_raw : split별 정제 후 DataFrame (중복 제거 전)
    count_info : 건수 통계 dict
    """
    print("\n2. 전체 데이터 수 (텍스트 정제 후)")

    train_raw  = df[df["dataset"] == "train"].copy()
    valid_raw  = df[df["dataset"] == "valid"].copy()
    num_labels = df["label"].nunique()

    print(f"  전체    : {len(df):,}")
    print(f"  Train   : {len(train_raw):,}")
    print(f"  Valid   : {len(valid_raw):,}")
    print(f"  라벨 수 : {num_labels}")

    count_info = {
        "total"     : len(df),
        "train"     : len(train_raw),
        "valid"     : len(valid_raw),
        "num_labels": num_labels,
    }
    return train_raw, valid_raw, count_info


def step3_duplicate(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame,
    csv_dir: Path = CSV_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    3. 중복 데이터 수 집계 + CSV 저장
    4. 중복 제거 후 데이터 수

    Returns
    -------
    train_df, valid_df : 중복 제거된 DataFrame
    dup_info : 중복 통계 dict
    """
    print("\n3. 중복 데이터 수 (label & text 기준)")

    dup_train_mask = train_raw.duplicated(subset=["label", "text"], keep=False)
    dup_valid_mask = valid_raw.duplicated(subset=["label", "text"], keep=False)

    dup_train = int(train_raw.duplicated(subset=["label", "text"]).sum())
    dup_valid = int(valid_raw.duplicated(subset=["label", "text"]).sum())
    dup_total = int(
        pd.concat([train_raw, valid_raw]).duplicated(subset=["label", "text"]).sum()
    )

    train_pairs     = set(zip(train_raw["label"], train_raw["text"]))
    valid_pairs     = set(zip(valid_raw["label"], valid_raw["text"]))
    cross_dup_pairs = train_pairs & valid_pairs
    dup_cross       = len(cross_dup_pairs)

    print(f"  Train 내부 중복         : {dup_train:,}건")
    print(f"  Valid 내부 중복         : {dup_valid:,}건")
    print(f"  Train ↔ Valid 교차 중복  : {dup_cross:,}건")
    print(f"  전체 합산 검증           : {dup_train + dup_valid + dup_cross:,}건  ←→  {dup_total:,}건")

    # CSV 저장
    dup_train_df = train_raw[dup_train_mask].sort_values(["label", "text"])
    dup_valid_df = valid_raw[dup_valid_mask].sort_values(["label", "text"])

    cross_mask_tr = train_raw.apply(
        lambda r: (r["label"], r["text"]) in cross_dup_pairs, axis=1
    )
    cross_mask_vl = valid_raw.apply(
        lambda r: (r["label"], r["text"]) in cross_dup_pairs, axis=1
    )
    dup_cross_df = pd.concat(
        [train_raw[cross_mask_tr], valid_raw[cross_mask_vl]], ignore_index=True
    ).sort_values(["label", "text", "dataset"])

    csv_dir.mkdir(parents=True, exist_ok=True)
    dup_train_csv = csv_dir / "duplicate_train.csv"
    dup_valid_csv = csv_dir / "duplicate_valid.csv"
    dup_cross_csv = csv_dir / "duplicate_cross.csv"
    dup_train_df.to_csv(dup_train_csv, index=False, encoding="utf-8-sig")
    dup_valid_df.to_csv(dup_valid_csv, index=False, encoding="utf-8-sig")
    dup_cross_df.to_csv(dup_cross_csv, index=False, encoding="utf-8-sig")
    print(f"\n  [저장] {dup_train_csv}  ({len(dup_train_df):,}행)")
    print(f"  [저장] {dup_valid_csv}  ({len(dup_valid_df):,}행)")
    print(f"  [저장] {dup_cross_csv}  ({len(dup_cross_df):,}행)")

    # 중복 제거
    print("\n4. 중복 제거 후 데이터 수")
    combined  = pd.concat([train_raw, valid_raw])
    combined  = combined.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    train_df  = combined[combined["dataset"] == "train"].copy().reset_index(drop=True)
    valid_df  = combined[combined["dataset"] == "valid"].copy().reset_index(drop=True)

    cross_mask = train_df.set_index(["label", "text"]).index.isin(
        valid_df.set_index(["label", "text"]).index
    )
    train_df = train_df[~cross_mask].reset_index(drop=True)

    print(f"  Train : {len(train_df):,}건  (교차 중복 {dup_cross:,}건 추가 제거)")
    print(f"  Valid : {len(valid_df):,}건")
    print(f"  전체  : {len(train_df) + len(valid_df):,}건")

    dup_info = {
        "total"        : dup_total,
        "train"        : dup_train,
        "valid"        : dup_valid,
        "cross"        : dup_cross,
        "after_dedup"  : {
            "total": len(train_df) + len(valid_df),
            "train": len(train_df),
            "valid": len(valid_df),
        },
        "saved_csv": {
            "duplicate_train": str(dup_train_csv),
            "duplicate_valid": str(dup_valid_csv),
            "duplicate_cross": str(dup_cross_csv),
        },
    }
    return train_df, valid_df, dup_info


def step5_label_distribution(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    report_dir: Path,
    font: fm.FontProperties,
) -> dict:
    """
    5. 라벨 별 데이터 수 (중복 제거 전 → 후 비교)
       그래프: label_distribution.png
    """
    print("\n5. 라벨 별 데이터 수")

    label_before_train = _get_label_distribution(train_raw)
    label_before_valid = _get_label_distribution(valid_raw)
    label_after_train  = _get_label_distribution(train_df)
    label_after_valid  = _get_label_distribution(valid_df)

    for split, before, after in [
        ("Train", label_before_train, label_after_train),
        ("Valid", label_before_valid, label_after_valid),
    ]:
        print(f"  [{split}]")
        for label in sorted(set(before) | set(after)):
            b = before.get(label, {"count": 0, "ratio": 0.0})
            a = after.get(label,  {"count": 0, "ratio": 0.0})
            print(f"    {label}: {b['count']:,} → {a['count']:,}건 ({a['ratio']}%)")

    # 시각화 (중복 제거 후 기준)
    label_dist_png = report_dir / "label_distribution.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (split_df, title) in zip(axes, [(train_df, "Train"), (valid_df, "Valid")]):
        counts = split_df["label"].value_counts().sort_index()
        bars   = ax.bar(counts.index.astype(str), counts.values,
                        color="#4C72B0", edgecolor="white")
        ax.set_title(f"라벨 분포 — {title}", fontsize=13, fontweight="bold",
                     fontproperties=font)
        ax.set_xlabel("라벨", fontproperties=font)
        ax.set_ylabel("건수", fontproperties=font)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9,
            )
        _apply_kr_font(ax, font)
    plt.tight_layout()
    plt.savefig(label_dist_png, dpi=300)
    plt.close()
    print(f"\n  [저장] {label_dist_png}")

    return {
        "before_dedup": {"train": label_before_train, "valid": label_before_valid},
        "after_dedup" : {"train": label_after_train,  "valid": label_after_valid},
    }


def step6_char_length(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    report_dir: Path,
    font: fm.FontProperties,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    6. 문장 길이 분석 (char 기준)
       그래프: char_length_distribution.png

    Returns DataFrame에 char_len 컬럼 추가
    """
    print("\n6. 문장 길이 (Character 기준)")

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["char_len"] = train_df["text"].astype(str).apply(len)
    valid_df["char_len"] = valid_df["text"].astype(str).apply(len)

    char_stats_train = _get_length_stats(train_df["char_len"])
    char_stats_valid = _get_length_stats(valid_df["char_len"])

    for split, stats in [("Train", char_stats_train), ("Valid", char_stats_valid)]:
        print(f"  [{split}] mean={stats['mean']} median={stats['median']} "
              f"std={stats['std']} min={stats['min']} max={stats['max']} "
              f"p90={stats['p90']} p95={stats['p95']} p99={stats['p99']}")

    char_hist_png = report_dir / "char_length_distribution.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (split_df, label) in zip(axes, [(train_df, "Train"), (valid_df, "Valid")]):
        ax.hist(split_df["char_len"], bins=50, color="#55A868", edgecolor="white")
        ax.set_title(f"문장 길이 분포 (char) — {label}", fontsize=13,
                     fontweight="bold", fontproperties=font)
        ax.set_xlabel("문자 수", fontproperties=font)
        ax.set_ylabel("건수", fontproperties=font)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        s = _get_length_stats(split_df["char_len"])
        textstr = f"mean={s['mean']}\nmedian={s['median']}\np95={s['p95']}\nmax={s['max']}"
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
                fontproperties=font)
        _apply_kr_font(ax, font)
    plt.tight_layout()
    plt.savefig(char_hist_png, dpi=300)
    plt.close()
    print(f"  [저장] {char_hist_png}")

    return train_df, valid_df, {"train": char_stats_train, "valid": char_stats_valid}


def step7_token_length(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    tokenizer_name: str,
    report_dir: Path,
    font: fm.FontProperties,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, object]:
    """
    7. 토크나이징 후 토큰 수 분석
       p95 / p99 초과 건수 포함
       그래프: token_length_distribution.png

    Returns DataFrame에 token_len 컬럼 추가
    """
    print("\n7. 토크나이징 후 토큰 수")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _token_len(text: str) -> int:
        return len(tokenizer.encode(str(text), add_special_tokens=False))

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["token_len"] = train_df["text"].astype(str).apply(_token_len)
    valid_df["token_len"] = valid_df["text"].astype(str).apply(_token_len)

    tok_stats_train = _get_length_stats(train_df["token_len"])
    tok_stats_valid = _get_length_stats(valid_df["token_len"])

    # p95 / p99 초과 건수
    def _over_counts(df: pd.DataFrame, stats: dict) -> dict:
        return {
            "over_p95": int((df["token_len"] > stats["p95"]).sum()),
            "over_p99": int((df["token_len"] > stats["p99"]).sum()),
        }

    over_train = _over_counts(train_df, tok_stats_train)
    over_valid = _over_counts(valid_df, tok_stats_valid)
    tok_stats_train.update(over_train)
    tok_stats_valid.update(over_valid)

    for split, stats in [("Train", tok_stats_train), ("Valid", tok_stats_valid)]:
        print(f"  [{split}] mean={stats['mean']} median={stats['median']} "
              f"std={stats['std']} min={stats['min']} max={stats['max']}")
        print(f"           p90={stats['p90']} p95={stats['p95']} (초과 {stats['over_p95']:,}건) "
              f"p99={stats['p99']} (초과 {stats['over_p99']:,}건)")

    token_hist_png = report_dir / "token_length_distribution.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (split_df, label, stats) in zip(
        axes,
        [(train_df, "Train", tok_stats_train), (valid_df, "Valid", tok_stats_valid)],
    ):
        ax.hist(split_df["token_len"], bins=50, color="#C44E52", edgecolor="white")
        # p95 / p99 수직선
        ax.axvline(stats["p95"], color="#E67E22", linestyle="--", linewidth=1.2,
                   label=f"p95={stats['p95']} ({stats['over_p95']:,}건 초과)")
        ax.axvline(stats["p99"], color="#8E44AD", linestyle="--", linewidth=1.2,
                   label=f"p99={stats['p99']} ({stats['over_p99']:,}건 초과)")
        ax.set_title(f"토큰 수 분포 — {label}", fontsize=13,
                     fontweight="bold", fontproperties=font)
        ax.set_xlabel("토큰 수", fontproperties=font)
        ax.set_ylabel("건수", fontproperties=font)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        textstr = (f"mean={stats['mean']}\nmedian={stats['median']}\n"
                   f"p95={stats['p95']}\nmax={stats['max']}")
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
                fontproperties=font)
        ax.legend(prop=font, fontsize=8)
        _apply_kr_font(ax, font)
    plt.tight_layout()
    plt.savefig(token_hist_png, dpi=300)
    plt.close()
    print(f"  [저장] {token_hist_png}")

    return train_df, valid_df, {"train": tok_stats_train, "valid": tok_stats_valid}, tokenizer


def step8_vocab(tokenizer, tokenizer_name: str) -> dict:
    """8. Vocab 정보"""
    print("\n8. Vocab 정보")
    vocab_info = {
        "name"      : tokenizer_name,
        "class"     : tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
    }
    print(f"  Tokenizer  : {vocab_info['name']}")
    print(f"  Class      : {vocab_info['class']}")
    print(f"  Vocab Size : {vocab_info['vocab_size']:,}")
    return vocab_info




def save_results(
    report_dir: Path,
    csv_dir: Path,
    cleaning_info: dict,
    count_info: dict,
    dup_info: dict,
    label_dist: dict,
    char_stats: dict,
    tok_stats: dict,
    vocab_info: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> dict:
    """분석 결과를 JSON으로 저장하고 최종 학습용 CSV를 저장"""
    csv_dir.mkdir(parents=True, exist_ok=True)
    clean_train_csv = csv_dir / "clean_train.csv"
    clean_valid_csv = csv_dir / "clean_valid.csv"
    train_df.to_csv(clean_train_csv, index=False, encoding="utf-8-sig")
    valid_df.to_csv(clean_valid_csv, index=False, encoding="utf-8-sig")
    print(f"\n[최종 데이터셋 저장]")
    print(f"  Train : {clean_train_csv}  ({len(train_df):,}건)")
    print(f"  Valid : {clean_valid_csv}  ({len(valid_df):,}건)")

    result = {
        "dataset_count": {
            "raw"          : count_info,
            "text_cleaning": cleaning_info,
            "after_dedup"  : dup_info["after_dedup"],
            "final"        : {
                "total": len(train_df) + len(valid_df),
                "train": len(train_df),
                "valid": len(valid_df),
            },
        },
        "duplicate"       : dup_info,
        "label_distribution": label_dist,
        "char_length"     : char_stats,
        "token_length"    : tok_stats,
        "tokenizer"       : vocab_info,
        "saved_csv": {
            **dup_info.get("saved_csv", {}),
            "empty_text_removed": str(cleaning_info.get("removed_csv", "")),
            "clean_train"       : str(clean_train_csv),
            "clean_valid"       : str(clean_valid_csv),
        },
    }

    analysis_json = report_dir / "analysis_result.json"
    with open(analysis_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"  결과 JSON : {analysis_json}")
    return result


#################################################
# 메인 실행 함수 (외부 import 진입점)
#################################################

def run_analysis(
    data_path: Path  = DATA_PATH,
    report_dir: Path = REPORT_DIR,
    csv_dir: Path    = CSV_DIR,
    tokenizer_name: str = TOKENIZER_NAME,
) -> dict:
    """
    전체 분석 파이프라인 실행

    Parameters
    ----------
    data_path       : 입력 JSON 파일 경로
    report_dir      : 보고서 및 PNG 저장 디렉토리
    csv_dir         : CSV 파일 저장 디렉토리
    tokenizer_name  : HuggingFace 토크나이저 이름

    Returns
    -------
    result : analysis_result.json 내용과 동일한 dict
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    font = _get_kr_font()

    # 순서: 1 → 2 → 3(중복) → 4(중복제거) → 5(라벨) → 6(char) → 7(token) → 8(vocab)
    df, cleaning_info                         = step1_load_and_clean(data_path, report_dir, csv_dir)
    train_raw, valid_raw, count_info          = step2_count(df)
    train_df, valid_df, dup_info              = step3_duplicate(train_raw, valid_raw, csv_dir)
    label_dist                                = step5_label_distribution(
                                                    train_raw, valid_raw,
                                                    train_df, valid_df,
                                                    report_dir, font)
    train_df, valid_df, char_stats            = step6_char_length(train_df, valid_df, report_dir, font)
    train_df, valid_df, tok_stats, tokenizer  = step7_token_length(
                                                    train_df, valid_df,
                                                    tokenizer_name, report_dir, font)
    vocab_info                                = step8_vocab(tokenizer, tokenizer_name)
    result = save_results(
        report_dir, csv_dir, cleaning_info, count_info, dup_info,
        label_dist, char_stats, tok_stats, vocab_info,
        train_df, valid_df,
    )
    print("\n" + "=" * 50)
    print("분석 완료")
    print("=" * 50)
    return result


#################################################
# 직접 실행
#################################################
if __name__ == "__main__":
    run_analysis()