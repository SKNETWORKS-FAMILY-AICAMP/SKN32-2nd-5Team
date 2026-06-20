from pathlib import Path
import json
from datetime import datetime

#################################################
# CONFIG
#################################################
REPORT_DIR    = Path("../reports")
ANALYSIS_JSON = REPORT_DIR / "analysis_result.json"
REPORT_PATH   = REPORT_DIR / "preprocessing_report.md"

#################################################
# LOAD
#################################################
with open(ANALYSIS_JSON, "r", encoding="utf-8") as f:
    r = json.load(f)

now = datetime.now().strftime("%Y-%m-%d %H:%M")

#################################################
# HELPER
#################################################
def stats_table(stats: dict) -> str:
    """통계 dict → 마크다운 테이블 행 문자열"""
    return (
        f"| {stats['mean']:,} | {stats['median']:,} | {stats['std']:,} "
        f"| {stats['min']:,} | {stats['max']:,} "
        f"| {stats['p90']:,} | {stats['p95']:,} | {stats['p99']:,} |"
    )

def label_compare_rows(before: dict, after: dict) -> str:
    """before/after 라벨 분포 비교 테이블 행"""
    rows = []
    all_labels = sorted(set(before.keys()) | set(after.keys()))
    for label in all_labels:
        b = before.get(label, {"count": 0, "ratio": 0.0})
        a = after.get(label,  {"count": 0, "ratio": 0.0})
        diff = a["count"] - b["count"]
        diff_str = f"{diff:+,}"
        rows.append(
            f"| {label} "
            f"| {b['count']:,} ({b['ratio']}%) "
            f"| {a['count']:,} ({a['ratio']}%) "
            f"| {diff_str} |"
        )
    return "\n".join(rows)

#################################################
# SHORTCUT VARIABLES
#################################################
dc    = r["dataset_count"]
dup   = r["duplicate"]
ld    = r["label_distribution"]
char  = r["char_length"]
tok   = r["token_length"]
vocab = r["tokenizer"]
qual  = r["data_quality"]
csv   = r["saved_csv"]

#################################################
# REPORT
#################################################
report = f"""# 데이터 전처리 결과 보고서

> 작성일: {now}

---

## 1. 데이터 로드

| 항목 | 값 |
|---|---|
| 데이터 경로 | `../output/analysis_dataset.json` |
| 토크나이저 | `{vocab['name']}` |

---

## 2. 전체 데이터 수

### 원본 (중복 제거 전)

| 구분 | 건수 |
|---|---|
| 전체 | {dc['raw']['total']:,} |
| Train | {dc['raw']['train']:,} |
| Valid | {dc['raw']['valid']:,} |
| 라벨 수 | {dc['num_labels']} |

---

## 3. 라벨 별 데이터 수

### Train — Before / After 중복 제거 비교

| 라벨 | Before (건 / 비율) | After (건 / 비율) | 변화량 |
|---|---|---|---|
{label_compare_rows(ld['before_dedup']['train'], ld['after_dedup']['train'])}

### Valid — Before / After 중복 제거 비교

| 라벨 | Before (건 / 비율) | After (건 / 비율) | 변화량 |
|---|---|---|---|
{label_compare_rows(ld['before_dedup']['valid'], ld['after_dedup']['valid'])}

![Label Distribution](label_distribution.png)

---

## 4. 중복 데이터 수 (label & text 기준)

| 중복 유형 | 건수 |
|---|---|
| Train 내부 중복 | {dup['train']:,} |
| Valid 내부 중복 | {dup['valid']:,} |
| Train ↔ Valid 교차 중복 | {dup['cross']:,} |
| **전체 합산** | **{dup['train'] + dup['valid'] + dup['cross']:,}** |

> **교차 중복(Cross-split Duplicate)**: `raw_df` 전체 기준 중복 건수({dup['total']:,})와 각 split 내부 중복 합산이 다른 이유는,
> Train과 Valid **양쪽에 동일한 `(label, text)` 쌍이 존재**하는 경우가 {dup['cross']:,}건 있기 때문입니다.
> 이는 **데이터 누수(Data Leakage)** 가능성이 있으므로 Valid를 우선 보존하고 Train에서 해당 데이터를 제거합니다.

### 저장된 중복 데이터 CSV

| 파일 | 설명 |
|---|---|
| `duplicate_train.csv` | Train 내부 중복 행 전체 |
| `duplicate_valid.csv` | Valid 내부 중복 행 전체 |
| `duplicate_cross.csv` | Train ↔ Valid 교차 중복 행 전체 |

---

## 5. 중복 제거 후 데이터 수

> 제거 순서: ① 각 split 내부 중복 제거 → ② Train ↔ Valid 교차 중복 제거 (Valid 우선 보존)

| 구분 | 건수 | 비고 |
|---|---|---|
| Train | {dc['after_dedup']['train']:,} | 교차 중복 {dup['cross']:,}건 추가 제거 |
| Valid | {dc['after_dedup']['valid']:,} | — |
| **전체** | **{dc['after_dedup']['total']:,}** | — |

---

## 6. 문장 길이 (Character 기준)

| 구분 | mean | median | std | min | max | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| Train {stats_table(char['train'])}
| Valid {stats_table(char['valid'])}

![Character Length Distribution](char_length_distribution.png)

---

## 7. 토크나이징 후 토큰 수

| 구분 | mean | median | std | min | max | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| Train {stats_table(tok['train'])}
| Valid {stats_table(tok['valid'])}

![Token Length Distribution](token_length_distribution.png)

---

## 8. Vocab 정보

| 항목 | 값 |
|---|---|
| Tokenizer | `{vocab['name']}` |
| Class | `{vocab['class']}` |
| Vocabulary Size | {vocab['vocab_size']:,} |

---

## 9. 데이터 품질 점검

| 항목 | 기준 | Train | Valid |
|---|---|---|---|
| 빈 텍스트 | `text == ""` | {qual['empty_text']['train']:,} | {qual['empty_text']['valid']:,} |
| 짧은 텍스트 | `char ≤ {qual['short_text']['threshold']}` | {qual['short_text']['train']:,} | {qual['short_text']['valid']:,} |
| 긴 텍스트 | `token > {qual['long_text']['threshold']}` | {qual['long_text']['train']:,} | {qual['long_text']['valid']:,} |
| 결측값 | `text is null` | {qual['null_text']['train']:,} | {qual['null_text']['valid']:,} |

### 저장된 빈 텍스트 CSV

| 파일 | 설명 |
|---|---|
| `empty_text_train.csv` | Train 빈 텍스트 행 전체 ({qual['empty_text']['train']:,}건) |
| `empty_text_valid.csv` | Valid 빈 텍스트 행 전체 ({qual['empty_text']['valid']:,}건) |

![Data Quality Report](data_quality_report.png)

---

## 10. 모델 입력 사양

### 공통 Tokenizer
- KLUE SentencePiece (`{vocab['name']}`)

### 사용 모델

| # | 모델 |
|---|---|
| 1 | LSTM |
| 2 | Text CNN |
| 3 | Transformer Encoder (Scratch) |
| 4 | KLUE-BERT Fine-tuning |

---

## 11. 생성 산출물

### 보고서
- `preprocessing_report.md`

### 시각화
- `label_distribution.png` — Train / Valid 라벨 분포 (중복 제거 전)
- `char_length_distribution.png` — Train / Valid 문장 길이 분포
- `token_length_distribution.png` — Train / Valid 토큰 수 분포
- `data_quality_report.png` — 데이터 품질 점검 결과

### 중복 데이터 CSV
- `duplicate_train.csv` — Train 내부 중복 행
- `duplicate_valid.csv` — Valid 내부 중복 행
- `duplicate_cross.csv` — Train ↔ Valid 교차 중복 행

### 빈 텍스트 CSV
- `empty_text_train.csv` — Train 빈 텍스트 행
- `empty_text_valid.csv` — Valid 빈 텍스트 행

### 분석 결과
- `analysis_result.json`
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report.lstrip())

print("보고서 생성 완료")
print(REPORT_PATH)