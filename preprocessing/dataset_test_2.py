from pathlib import Path
from collections import Counter
import csv


# =========================
# 설정
# =========================

REVIEW_FILE = Path("../output/review_dataset.csv")
ERROR_FILE = Path("../output/error_cases.csv")


# =========================
# 라벨 정규화
# =========================
def normalize_label(value: str) -> str:
    """
    입력 예시

    r
    retain
    RETAIN

    c
    churn_signal
    CHURN_SIGNAL

    반환
    r 또는 c
    """

    value = str(value).strip().lower()

    if not value:
        return ""

    if value.startswith("r"):
        return "r"

    if value.startswith("c"):
        return "c"

    return ""


# =========================
# 다수결
# =========================
def majority_vote(*labels: str) -> str:
    labels = [label for label in labels if label]

    if not labels:
        return ""

    # 2개일 때는 둘이 같아야만 인정
    if len(labels) == 2:
        if labels[0] == labels[1]:
            return labels[0]

        return ""

    # 3개 이상은 기존 다수결
    return Counter(labels).most_common(1)[0][0]


# =========================
# 메인
# =========================
def main():

    total_count = 0
    correct_count = 0
    skip_count = 0

    disagreement_count = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    error_rows = []

    with REVIEW_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            # -----------------
            # Auto Label
            # -----------------
            y_pred = normalize_label(
                row.get("auto_label", "")
            )

            # -----------------
            # Worker Label
            # -----------------
            worker_1 = normalize_label(
                row.get("worker_1", "")
            )

            worker_2 = normalize_label(
                row.get("worker_2", "")
            )

            worker_3 = normalize_label(
                row.get("worker_3", "")
            )

            worker_labels = [
                worker_1,
                worker_2,
                worker_3,
            ]

            worker_labels = [
                x
                for x in worker_labels
                if x
            ]

            # 작업자 모두 비어있는 경우
            if not worker_labels:
                skip_count += 1
                continue

            # -----------------
            # 불일치 여부
            # -----------------
            if len(set(worker_labels)) > 1:
                disagreement_count += 1

            # -----------------
            # Human Label
            # -----------------
            y_true = majority_vote(
                worker_1,
                worker_2,
                # worker_3,
            )

            if not y_true or not y_pred:
                skip_count += 1
                continue

            total_count += 1

            # -----------------
            # Accuracy
            # -----------------
            if y_true == y_pred:
                correct_count += 1

            else:
                error_rows.append(
                    {
                        "dialog_id": row.get(
                            "dialog_id",
                            ""
                        ),
                        "auto_label": row.get(
                            "auto_label",
                            ""
                        ),
                        "human_label": y_true,
                        "worker_1": worker_1,
                        "worker_2": worker_2,
                        "worker_3": worker_3,
                        "text": row.get(
                            "text",
                            ""
                        ),
                    }
                )

            # -----------------
            # Confusion Matrix
            # -----------------

            if y_true == "c" and y_pred == "c":
                tp += 1

            elif y_true == "r" and y_pred == "r":
                tn += 1

            elif y_true == "r" and y_pred == "c":
                fp += 1

            elif y_true == "c" and y_pred == "r":
                fn += 1

    if total_count == 0:
        print("비교할 데이터가 없습니다.")
        return

    # =====================
    # Metrics
    # =====================

    accuracy = correct_count / total_count

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1_score = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # =====================
    # Error Case 저장
    # =====================

    with ERROR_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dialog_id",
                "auto_label",
                "human_label",
                "worker_1",
                "worker_2",
                "worker_3",
                "text",
            ]
        )

        writer.writeheader()
        writer.writerows(error_rows)

    # =====================
    # 출력
    # =====================

    print("=" * 60)
    print("라벨 품질 검증 결과")
    print("=" * 60)

    print()
    print("[기본 정보]")
    print(f"비교 대상 : {total_count:,}건")
    print(f"정답 개수 : {correct_count:,}건")
    print(f"오답 개수 : {total_count - correct_count:,}건")
    print(f"스킵 개수 : {skip_count:,}건")

    print()
    print("[Accuracy]")
    print(f"{accuracy * 100:.2f}%")

    print()
    print("[Confusion Matrix]")
    print(
        f"TP (실제 churn → 예측 churn)  : "
        f"{tp:,}"
    )
    print(
        f"FN (실제 churn → 예측 retain) : "
        f"{fn:,}"
    )
    print(
        f"FP (실제 retain → 예측 churn) : "
        f"{fp:,}"
    )
    print(
        f"TN (실제 retain → 예측 retain): "
        f"{tn:,}"
    )

    print()
    print("[Churn 기준 성능]")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1_score * 100:.2f}%")

    print()
    print("[검수자 일치도]")
    print(
        f"불일치 건수 : "
        f"{disagreement_count:,}건"
    )
    print(
        f"불일치 비율 : "
        f"{disagreement_count / total_count * 100:.2f}%"
    )

    print()
    print(
        f"[SAVE] 오답 케이스 저장 : "
        f"{ERROR_FILE}"
    )


# =========================
# 시작
# =========================
if __name__ == "__main__":
    main()