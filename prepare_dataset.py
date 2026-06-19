from pathlib import Path
import json
import csv
import hashlib

from labeling import (
    process_dataset,
    TRAIN_FILE_NAME,
    VALID_FILE_NAME,
    Label,
)

# =========================
# 설정
# =========================
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

REVIEW_SAMPLE_RATE = 0.01  # 약 1%

TEXT_KEYS = [
    "고객질문(요청)",
    "상담사질문(요청)",
    "고객답변",
    "상담사답변",
]


# =========================
# 텍스트 생성
# =========================
def build_text(record: dict) -> str:
    return " ".join(
        str(record.get(key, "")).strip()
        for key in TEXT_KEYS
        if record.get(key)
    )


# =========================
# 검수 샘플 선정
# =========================
def is_review_target(dialog_id: str) -> bool:
    value = int(
        hashlib.md5(
            dialog_id.encode("utf-8")
        ).hexdigest(),
        16,
    )

    return value % 10000 < REVIEW_SAMPLE_RATE * 10000


# =========================
# Label Dict -> Flat List
# =========================
def flatten_labels(label_dict: dict) -> list[dict]:
    result = []

    for label, rows in label_dict.items():
        for row in rows:
            result.append(
                {
                    **row,
                    "label": label.value,
                }
            )

    return result


# =========================
# 분석용 JSON 생성
# =========================
def save_analysis_json(
    train_data: list[dict],
    valid_data: list[dict],
):
    analysis_data = []

    for dataset_type, rows in (
        ("train", train_data),
        ("valid", valid_data),
    ):
        for row in rows:

            analysis_data.append(
                {
                    "dataset": dataset_type,
                    "dialog_id": row.get("대화셋일련번호"),
                    "seq": row.get("문장번호"),
                    "label": row["label"],
                    "text": build_text(row),
                }
            )

    output_file = OUTPUT_DIR / "analysis_dataset.json"

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            analysis_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[SAVE] analysis_dataset.json "
        f"({len(analysis_data):,}건)"
    )


# =========================
# 검수용 CSV 생성
# =========================
def save_review_csv(
    all_data: list[dict],
):
    review_rows = []

    for row in all_data:

        dialog_id = str(
            row.get("대화셋일련번호", "")
        )
        seq = int(
            row.get("문장번호", 0)
        )

        if not dialog_id:
            continue

        if not is_review_target(dialog_id):
            continue

        review_rows.append(
            {
                "dialog_id": dialog_id,
                "seq": seq,
                "text": build_text(row),
                "auto_label": row["label"],
                "worker_1": "",
                "worker_2": "",
                "worker_3": "",
            }
        )

    output_file = OUTPUT_DIR / "review_dataset.csv"

    with output_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dialog_id",
                "seq",
                "text",
                "auto_label",
                "worker_1",
                "worker_2",
                "worker_3",
            ],
        )

        writer.writeheader()
        writer.writerows(review_rows)

    print(
        f"[SAVE] review_dataset.csv "
        f"({len(review_rows):,}건)"
    )


# =========================
# Main
# =========================
def main():

    print("[1] 라벨 데이터 로드")

    train_labels = process_dataset(
        TRAIN_FILE_NAME
    )

    valid_labels = process_dataset(
        VALID_FILE_NAME
    )

    train_data = flatten_labels(
        train_labels
    )

    valid_data = flatten_labels(
        valid_labels
    )

    all_data = (
        train_data +
        valid_data
    )

    print("\n[2] 분석용 데이터 생성")

    save_analysis_json(
        train_data,
        valid_data,
    )

    print("\n[3] 검수용 데이터 생성")

    save_review_csv(
        all_data
    )

    print("\n완료")


if __name__ == "__main__":
    main()