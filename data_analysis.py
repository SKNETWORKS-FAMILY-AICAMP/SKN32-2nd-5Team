from pathlib import Path
import json

# =========================
# 설정
# =========================
DATA_ROOT = Path("./data")

TRAIN_FILE_NAME = "Training.json"
VALID_FILE_NAME = "Validation.json"

TARGET_KEYS = ["고객의도", "상담사의도", "개체명 ", "용어사전"]

HAEJI_KEYWORDS = ("해지", "해약", "취소", "민원")
GAIP_KEYWORDS = ("가입", "신규", "등록", "개설")


# =========================
# 데이터 로드
# =========================
def load_json(file_path: Path) -> list:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[SUCC] 파싱 성공: {file_path.name}")
        return data

    except FileNotFoundError:
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path.resolve()}")
        return []

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패: {e}")
        return []


# =========================
# 레이블 분리
# =========================
def split_label(raw_data: list):
    haeji_list = []
    gaip_list = []
    etc_list = []

    for record in raw_data:
        text = " ".join(
            str(record.get(key, ""))
            for key in TARGET_KEYS
        )

        has_haeji = any(keyword in text for keyword in HAEJI_KEYWORDS)
        has_gaip = any(keyword in text for keyword in GAIP_KEYWORDS)

        # 해지 우선
        if has_haeji:
            haeji_list.append(record)

        # 가입
        elif has_gaip:
            gaip_list.append(record)

        # 기타
        else:
            etc_list.append(record)

    print(f"- 해지 데이터 : {len(haeji_list):,}개")
    print(f"- 가입 데이터 : {len(gaip_list):,}개")
    print(f"- 기타 데이터 : {len(etc_list):,}개")

    if haeji_list:
        print(f"   해지 샘플: {haeji_list[0]}")

    if gaip_list:
        print(f"   가입 샘플: {gaip_list[0]}")

    if etc_list:
        print(f"   기타 샘플: {etc_list[0]}")

    return haeji_list, gaip_list, etc_list


# =========================
# 데이터셋 처리
# =========================
def process_dataset(file_name: str):
    file_path = DATA_ROOT / file_name

    raw_data = load_json(file_path)

    print(f"\n {file_name}")
    print(f"전체 데이터 수: {len(raw_data):,}개")

    return split_label(raw_data)


# =========================
# 실행
# =========================
if __name__ == "__main__":
    train_haeji, train_gaip, train_etc = process_dataset(TRAIN_FILE_NAME)

    print("\n" + "-" * 60 + "\n")

    valid_haeji, valid_gaip, valid_etc = process_dataset(VALID_FILE_NAME)