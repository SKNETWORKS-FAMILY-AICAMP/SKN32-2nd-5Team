from pathlib import Path
from collections import defaultdict
from enum import Enum
import json
import re

# =========================
# 설정
# =========================
DATA_ROOT = Path("../data")

TRAIN_FILE_NAME = "Training.json"
VALID_FILE_NAME = "Validation.json"

TARGET_KEYS = [
    # "고객의도",
    # "상담사의도",
    # "개체명 ",
    # "용어사전",
    # "지식베이스",
    "고객질문(요청)",
    "상담사질문(요청)",
    "고객답변",
    "상담사답변",
]

CHURN_KEYWORDS = (
    "해지",
    "해약",
    "환매",
    "탈퇴",
    "계약종료",
    "중도해지",
    "철회",
    "해제",
    "환불",
    "민원",
    "불만",
    "타사이전",
    "계약이전",
    "중도인출",
    "매도",
    "정리",
    "금액.*많",
    "보험료.*많",
    "돈.*많",
    "많이.*나가",
    "너무.*많",
    "비싸",
    "부담",
    "줄이",
    "없애",
    "취소",
    "변경",
    "안되",
    "안됩니다",
    "고민",
    "사망일",
    "사망진단서",
    "금융감독원",
    "금감원",
    "만기",

)

CHURN_PATTERN = re.compile("|".join(map(re.escape, CHURN_KEYWORDS)))


# =========================
# Label
# =========================
class Label(str, Enum):
    CHURN = "churn_signal"
    RETAIN = "retain"


# =========================
# 데이터 로드
# =========================
def load_json(file_path: Path) -> list[dict]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[SUCC] 파싱 성공: {file_path.name}")
        return data

    except FileNotFoundError:
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path.resolve()}")

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패: {e}")

    return []


# =========================
# 텍스트 생성
# =========================
def build_search_text(record: dict) -> str:
    return " ".join(
        str(record.get(key, ""))
        for key in TARGET_KEYS
    )


# =========================
# 라벨 분류
# =========================
def classify_label(text: str) -> Label:
    if CHURN_PATTERN.search(text):
        return Label.CHURN

    return Label.RETAIN


# =========================
# 데이터 분리
# =========================
def split_label(raw_data: list[dict]) -> dict[Label, list]:
    labels = defaultdict(list)

    for record in raw_data:
        text = build_search_text(record)
        label = classify_label(text)

        labels[label].append(record)

    return labels


# =========================
# 통계 출력
# =========================
def print_statistics(labels: dict[Label, list]) -> None:
    total_count = sum(len(items) for items in labels.values())

    print(f"전체 데이터 수 : {total_count:,}개")

    for label in Label:
        count = len(labels[label])

        print(
            f"- {label.value:<13}: "
            f"{count:,}개 "
            f"({count / total_count * 100:.2f}%)"
        )

        if count:
            print(f"  샘플: {labels[label][0]}")


# =========================
# 데이터셋 처리
# =========================
def process_dataset(file_name: str) -> dict[Label, list]:
    file_path = DATA_ROOT / file_name

    raw_data = load_json(file_path)

    print(f"\n[{file_name}]")

    labels = split_label(raw_data)

    print_statistics(labels)

    return labels


# =========================
# 실행
# =========================
if __name__ == "__main__":
    train_labels = process_dataset(TRAIN_FILE_NAME)

    print("\n" + "-" * 60 + "\n")

    valid_labels = process_dataset(VALID_FILE_NAME)