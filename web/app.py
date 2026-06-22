import streamlit as st
import pandas as pd
import json
import os

# --- 파일 경로 설정 ---
VOC_FILE = "voc_test.json"
PRED_FILE = "voc_test_result.json"
CUSTOMER_NAMES = {
    "D001": "김민준", "D002": "박서연", "D003": "이도현", "D004": "최유진", "D005": "정지훈"
}

st.set_page_config(layout="wide", page_title="Churn-Zero AI")

# CSS 스타일 정의 (밝은 테마 + 체크박스 검정 글씨)
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; color: #333; }
    h3 { color: #333 !important; }
    .model-label { font-size: 1.0rem; font-weight: bold; color: #555; margin-bottom: 5px; }

    /* [수정] 체크박스 내부 텍스트 태그(p)를 명확히 지정하여 검정색 강제 적용 */
    div[data-testid="stCheckbox"] label p {
        color: black !important;
        font-weight: bold !important;
    }

    /* 버튼 스타일 */
    div.stButton > button {
        background-color: white !important;
        color: #333 !important;
        border: 1px solid #ddd !important;
        text-align: left !important;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }

    /* 채팅 말풍선 */
    .bubble { padding: 10px 15px; border-radius: 15px; display: inline-block; box-shadow: 1px 1px 2px #ccc; max-width: 80%; margin-bottom: 2px; }
    .bubble-user { background-color: white; color: black; border-top-left-radius: 0px; }
    .bubble-ai { background-color: #fee500; color: black; border-top-right-radius: 0px; }
    .prob-text { font-size: 0.75rem; font-weight: bold; margin-top: 2px; margin-left: 5px; }
    </style>
    """, unsafe_allow_html=True)


# 데이터 로드 및 병합 함수
@st.cache_data
def load_and_merge_data(voc_path, pred_path):
    if not os.path.exists(voc_path) or not os.path.exists(pred_path):
        return pd.DataFrame()

    with open(voc_path, 'r', encoding='utf-8') as f:
        voc_data = json.load(f)
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    records = []
    pred_map = {}
    for item in pred_data:
        d_id = item.get('dialog_id')
        for conv in item.get('conversation', []):
            pred_map[(d_id, conv['seq'])] = conv

    for item in voc_data:
        d_id = item.get('dialog_id')
        for conv in item.get('conversation', []):
            seq = conv.get('seq')
            pred_info = pred_map.get((d_id, seq), {})

            records.append({
                'dialog': d_id,
                'seq': seq,
                '화자': conv.get('role'),
                'text': conv.get('text'),
                'lstm_churn': pred_info.get('LSTM_predict', {}).get('churn_prob', 0),
                'trans_churn': pred_info.get('Transformer (Scratch)_predict', {}).get('churn_prob', 0)
            })
    return pd.DataFrame(records)


df = load_and_merge_data(VOC_FILE, PRED_FILE)

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 상담 내역")
    if not df.empty:
        for d_id in df['dialog'].unique():
            if st.button(CUSTOMER_NAMES.get(d_id, "고객"), key=f"btn_{d_id}", use_container_width=True):
                st.session_state['selected_dialog'] = d_id

with col2:
    target_id = st.session_state.get('selected_dialog')
    if target_id:
        st.markdown(f"### {CUSTOMER_NAMES.get(target_id, '고객')}님과의 상담 내역")

        st.markdown('<p class="model-label">모델 선택</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        is_lstm = c1.checkbox("LSTM", key=f"check_lstm_{target_id}")
        is_trans = c2.checkbox("Transformer Encoder", key=f"check_trans_{target_id}")

        conv = df[df['dialog'] == target_id].sort_values('seq')

        for _, row in conv.iterrows():
            if row['화자'] == '고객':
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-start;"><div class="bubble bubble-user">{row["text"]}</div></div>',
                    unsafe_allow_html=True)

                # 모델별 확률 출력
                if is_lstm:
                    st.markdown(
                        f'<div class="prob-text" style="color: #d9534f;">LSTM 이탈확률: {row["lstm_churn"]:.2%}</div>',
                        unsafe_allow_html=True)
                if is_trans:
                    st.markdown(
                        f'<div class="prob-text" style="color: #d9534f;">Transformer 이탈확률: {row["trans_churn"]:.2%}</div>',
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-end;"><div class="bubble bubble-ai">{row["text"]}</div></div>',
                    unsafe_allow_html=True)
                st.write("")
    else:
        st.info("좌측 목록에서 고객을 선택해주세요.")
