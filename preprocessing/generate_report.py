from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime

#################################################
# CONFIG
#################################################
REPORT_DIR    = Path("../reports")
CSV_DIR       = Path("../output")
ANALYSIS_JSON = REPORT_DIR / "analysis_result.json"
REPORT_PATH   = REPORT_DIR / "preprocessing_report.md"

#################################################
# 헬퍼 함수
#################################################

def _stats_row(split: str, stats: dict) -> str:
    """통계 dict → 마크다운 테이블 행"""
    return (
        f"| {split} "
        f"| {stats['mean']:,} | {stats['median']:,} | {stats['std']:,} "
        f"| {stats['min']:,} | {stats['max']:,} "
        f"| {stats['p90']:,} | {stats['p95']:,} | {stats['p99']:,} |"
    )


def _token_stats_row(split: str, stats: dict) -> str:
    """토큰 통계 dict → 마크다운 테이블 행 (p95/p99 초과 건수 포함)"""
    return (
        f"| {split} "
        f"| {stats['mean']:,} | {stats['median']:,} | {stats['std']:,} "
        f"| {stats['min']:,} | {stats['max']:,} "
        f"| {stats['p90']:,} "
        f"| {stats['p95']:,} ({stats.get('over_p95', 0):,}건 초과) "
        f"| {stats['p99']:,} ({stats.get('over_p99', 0):,}건 초과) |"
    )


def _label_compare_rows(before: dict, after: dict) -> str:
    """before/after 라벨 분포 비교 테이블 행"""
    rows = []
    for label in sorted(set(before) | set(after)):
        b = before.get(label, {"count": 0, "ratio": 0.0})
        a = after.get(label,  {"count": 0, "ratio": 0.0})
        rows.append(
            f"| {label} "
            f"| {b['count']:,} ({b['ratio']}%) "
            f"| {a['count']:,} ({a['ratio']}%) "
            f"| {a['count'] - b['count']:+,} |"
        )
    return "\n".join(rows)


#################################################
# 보고서 생성 함수
#################################################

def generate_report(
    analysis_json: Path = ANALYSIS_JSON,
    report_path: Path   = REPORT_PATH,
    csv_dir: Path       = CSV_DIR,
) -> Path:
    """
    analysis_result.json을 읽어 preprocessing_report.md를 생성합니다.

    Parameters
    ----------
    analysis_json : 분석 결과 JSON 경로
    report_path   : 출력 마크다운 경로
    csv_dir       : CSV 파일이 저장된 디렉토리

    Returns
    -------
    report_path : 생성된 파일 경로
    """
    with open(analysis_json, "r", encoding="utf-8") as f:
        r = json.load(f)

    now      = datetime.now().strftime("%Y-%m-%d %H:%M")
    dc       = r["dataset_count"]
    dup      = r["duplicate"]
    ld       = r["label_distribution"]
    char     = r["char_length"]
    tok      = r["token_length"]
    vocab    = r["tokenizer"]
    cleaning = dc.get("text_cleaning", {})

    report = f"""# 데이터 전처리 결과 보고서

> 작성일: {now}

---

## 1. 데이터 로드 및 텍스트 정제

| 항목 | 값 |
|---|---|
| 데이터 경로 | `../output/analysis_dataset.json` |
| 토크나이저 | `{vocab['name']}` |

### 1-1. 데이터 확인 (정제 전)

| 항목 | 값 |
|---|---|
| 로드 건수 | {cleaning.get('before_total', 0):,} |
| 결측값 (text) | {cleaning.get('none_count', 0):,}건 |
| 빈 텍스트 | {cleaning.get('empty_removed', 0):,}건 |

### 1-2. 텍스트 정제

| 단계 | 처리 내용 | 처리 건수 |
|---|---|---|
| ① None → `""` | text가 None인 행을 빈 문자열로 변환 | {cleaning.get('none_count', 0):,}건 |
| ② 앞뒤 공백 제거 | `str.strip()` 적용 | 전체 적용 |
| ③ 빈 텍스트 제거 | `text == ""` 행 drop → `empty_text_removed.csv` 저장 | {cleaning.get('empty_removed', 0):,}건 |
| **정제 후 전체** | | **{cleaning.get('after_total', 0):,}건** |

---

## 2. 전체 데이터 수 (텍스트 정제 후)

| 구분 | 건수 |
|---|---|
| 전체 | {dc['raw']['total']:,} |
| Train | {dc['raw']['train']:,} |
| Valid | {dc['raw']['valid']:,} |
| 라벨 수 | {dc['raw']['num_labels']} |

---

## 3. 중복 데이터 수 (label & text 기준)

| 중복 유형 | 건수 |
|---|---|
| Train 내부 중복 | {dup['train']:,} |
| Valid 내부 중복 | {dup['valid']:,} |
| Train ↔ Valid 교차 중복 | {dup['cross']:,} |
| **전체 합산** | **{dup['train'] + dup['valid'] + dup['cross']:,}** |

> **교차 중복(Cross-split Duplicate)**: Train과 Valid 양쪽에 동일한 `(label, text)` 쌍이 존재하는 경우로,
> **데이터 누수(Data Leakage)** 방지를 위해 Valid를 우선 보존하고 Train에서 제거합니다.

### 저장된 중복 데이터 CSV

| 파일 | 설명 |
|---|---|
| `{csv_dir}/duplicate_train.csv` | Train 내부 중복 행 전체 |
| `{csv_dir}/duplicate_valid.csv` | Valid 내부 중복 행 전체 |
| `{csv_dir}/duplicate_cross.csv` | Train ↔ Valid 교차 중복 행 전체 |

---

## 4. 중복 제거 후 데이터 수

> 제거 순서: ① 각 split 내부 중복 → ② Train ↔ Valid 교차 중복 (Valid 우선 보존)

| 구분 | 건수 | 비고 |
|---|---|---|
| Train | {dup['after_dedup']['train']:,} | 교차 중복 {dup['cross']:,}건 추가 제거 |
| Valid | {dup['after_dedup']['valid']:,} | — |
| **전체** | **{dup['after_dedup']['total']:,}** | — |

---

## 5. 라벨 별 데이터 수 (중복 제거 전 → 후 비교)

### Train

| 라벨 | Before (건 / 비율) | After (건 / 비율) | 변화량 |
|---|---|---|---|
{_label_compare_rows(ld['before_dedup']['train'], ld['after_dedup']['train'])}

### Valid

| 라벨 | Before (건 / 비율) | After (건 / 비율) | 변화량 |
|---|---|---|---|
{_label_compare_rows(ld['before_dedup']['valid'], ld['after_dedup']['valid'])}

![라벨 분포](label_distribution.png)

---

## 6. 문장 길이 (Character 기준)

| 구분 | mean | median | std | min | max | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
{_stats_row('Train', char['train'])}
{_stats_row('Valid', char['valid'])}

![문장 길이 분포](char_length_distribution.png)

---

## 7. 토크나이징 후 토큰 수

| 구분 | mean | median | std | min | max | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
{_token_stats_row('Train', tok['train'])}
{_token_stats_row('Valid', tok['valid'])}

> p95 / p99 기준선은 히스토그램에 수직선으로 표시됩니다.

![토큰 수 분포](token_length_distribution.png)

---

## 8. Vocab 정보

| 항목 | 값 |
|---|---|
| Tokenizer | `{vocab['name']}` |
| Class | `{vocab['class']}` |
| Vocabulary Size | {vocab['vocab_size']:,} |


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
- `label_distribution.png` — 라벨 분포 (중복 제거 후)
- `char_length_distribution.png` — 문장 길이 분포
- `token_length_distribution.png` — 토큰 수 분포 (p95/p99 기준선 포함)

### 중복 데이터 CSV
- `duplicate_train.csv` — Train 내부 중복 행
- `duplicate_valid.csv` — Valid 내부 중복 행
- `duplicate_cross.csv` — Train ↔ Valid 교차 중복 행

### 빈 텍스트 CSV
- `{csv_dir}/empty_text_removed.csv` — 정제 단계에서 제거된 빈 텍스트 행

### 최종 학습용 데이터셋
- `{csv_dir}/clean_train.csv`
- `{csv_dir}/clean_valid.csv`

### 분석 결과
- `analysis_result.json`
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.lstrip())

    print(f"보고서 생성 완료: {report_path}")
    return report_path


#################################################
# 직접 실행
#################################################
if __name__ == "__main__":
    generate_report()