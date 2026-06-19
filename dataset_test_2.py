from pathlib import Path
import csv


# =========================
# 설정
# =========================

# 비교할 csv 파일 경로
REVIEW_FILE = Path("./output/review_dataset.csv")


# =========================
# 라벨 전처리 함수
# =========================
def get_first_char(value: str) -> str:
    """
    문자열의 첫 글자만 소문자로 반환한다.

    예)
    churn -> c
    retain -> r
    Churn -> c
    """

    # 공백 제거 + 소문자 변환
    value = str(value).strip().lower()

    # 값이 비어있으면 빈 문자열 반환
    if not value:
        return ""

    # 첫 글자만 반환
    return value[0]


# =========================
# 메인 함수
# =========================
def main() -> None:

    # 전체 비교 개수
    total_count = 0

    # 맞춘 개수
    correct_count = 0

    # 비어 있어서 비교하지 못한 개수
    skip_count = 0

    # ---------------------
    # Confusion Matrix용 변수
    # ---------------------

    # TP(True Positive)
    # 실제 churn(c), 예측 churn(c)
    tp = 0

    # TN(True Negative)
    # 실제 retain(r), 예측 retain(r)
    tn = 0

    # FP(False Positive)
    # 실제 retain(r), 예측 churn(c)
    fp = 0

    # FN(False Negative)
    # 실제 churn(c), 예측 retain(r)
    fn = 0

    # =========================
    # csv 파일 읽기
    # =========================
    with REVIEW_FILE.open("r", encoding="utf-8-sig", newline="") as f:

        # DictReader 사용
        # 컬럼명을 이용해서 데이터를 읽을 수 있다.
        reader = csv.DictReader(f)

        # 한 행씩 반복
        for row in reader:

            # ---------------------
            # 정답값(worker_1)
            # ---------------------
            y_true = get_first_char(
                row.get("worker_1", "")
            )

            # ---------------------
            # 모델 예측값(auto_label)
            # ---------------------
            y_pred = get_first_char(
                row.get("auto_label", "")
            )

            # 둘 중 하나라도 비어 있으면 건너뜀
            if not y_true or not y_pred:
                skip_count += 1
                continue

            # 비교 대상 개수 증가
            total_count += 1

            # 정답 여부 확인
            if y_true == y_pred:
                correct_count += 1

            # =====================
            # Confusion Matrix 계산
            # =====================

            # 실제 churn, 예측 churn
            if y_true == "c" and y_pred == "c":
                tp += 1

            # 실제 retain, 예측 retain
            elif y_true == "r" and y_pred == "r":
                tn += 1

            # 실제 retain, 예측 churn
            elif y_true == "r" and y_pred == "c":
                fp += 1

            # 실제 churn, 예측 retain
            elif y_true == "c" and y_pred == "r":
                fn += 1

    # 비교할 데이터가 없는 경우
    if total_count == 0:
        print("비교할 데이터가 없습니다.")
        return

    # =========================
    # Accuracy 계산
    # =========================
    accuracy = correct_count / total_count

    # =========================
    # churn Accuracy 계산
    # =========================
    if tp + fn > 0:
        churn_accuracy = tp / (tp + fn)
    else:
        churn_accuracy = 0

    # =========================
    # Precision 계산
    # churn이라고 예측한 것 중
    # 실제 churn인 비율
    # =========================
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    # =========================
    # Recall 계산
    # 실제 churn 중에서
    # 모델이 찾아낸 비율
    # =========================
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    # =========================
    # F1-score 계산
    # Precision과 Recall의 조화평균
    # =========================
    if precision + recall > 0:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1_score = 0

    # =========================
    # 결과 출력
    # =========================

    print("[worker_1 정답 기준 auto_label 성능]")
    print(f"비교 대상: {total_count:,}건")
    print(f"정답 개수: {correct_count:,}건")
    print(f"오답 개수: {total_count - correct_count:,}건")
    print(f"스킵: {skip_count:,}건")
    print(f"Accuracy : {accuracy * 100:.2f}%")

    print()

    print("[혼돈행렬]")
    print(f"TP (실제 churn → 예측 churn)  : {tp:,} ({tp / total_count * 100:.2f}%)")
    print(f"FN (실제 churn → 예측 retain) : {fn:,} ({fn / total_count * 100:.2f}%)")
    print(f"FP (실제 retain → 예측 churn) : {fp:,} ({fp / total_count * 100:.2f}%)")
    print(f"TN (실제 retain → 예측 retain): {tn:,} ({tn / total_count * 100:.2f}%)")

    print()

    print("[churn 기준 성능]")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1_score * 100:.2f}%")


# =========================
# 프로그램 시작
# =========================
if __name__ == "__main__":
    main()