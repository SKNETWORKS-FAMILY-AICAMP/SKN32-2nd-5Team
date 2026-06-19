from pathlib import Path
from collections import defaultdict
from enum import Enum
import json
import re

# =========================
# 설정
# =========================
DATA_ROOT = Path("./data")

TRAIN_FILE_NAME = "Training.json"
VALID_FILE_NAME = "Validation.json"

TARGET_KEYS = [
    "고객질문(요청)",
    "상담사질문(요청)",
    "고객답변",
    "상담사답변",
]

CHURN_KEYWORDS = (
    # 해지 / 철회
    "해지",
    "해약",
    "중도.*해지",
    "담보해지",
    "계약종료",
    "철회",
    "해제",
    "환불",
    "탈퇴",
    "환매",
    "중도인출",
    "매도",

    # 타사 이동
    "타사이전",
    "계약이전",

    # 감액 / 축소
    "감액",
    "감액대상",
    "줄이",
    "없애",
    "정리",

    # 보험료 부담
    "금액.*많",
    "보험료.*많",
    "돈.*많",
    "많이.*나가",
    "너무.*많",
    "보험.*인상",
    "얼마나.*오르",
    "비싸",
    "부담",

    # 이용 불가 / 불가능
    "안되",
    "안됩니다",
    "가능하지.*않",
    "불가능",
    "불가.*상품",
    "이용.*없으십니다",

    # 갱신 거절
    "갱신.*거절",
    "갱신거절",

    # 보상 관련 불만
    "보상.*요구",
    "보상.*불가능",

    # 사망보험금
    "사망일",
    "사망진단서",
    "사망.*진단서",

    # 민원 / 불만
    "민원",
    "불만",
    "금융감독원",
    "금감원",

    # 미납
    "미납",

    # 상실
    "상실",

    # 재가입
    "재가입.*보장.*변경",

    # 연락 보류
    "다시.*생각",
    "다시.*연락",

    # 문제 발생
    "보안.*문제",
    "잘못",

    # 의문 / 거부
    "꼭.*의문",
    "어렵게",
    "죄송합니다",

    # 기타
    "취소",
)

CHURN_PATTERN = re.compile("|".join(CHURN_KEYWORDS))


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