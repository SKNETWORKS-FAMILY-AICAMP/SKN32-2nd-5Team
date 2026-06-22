import streamlit as st
import pandas as pd
import json
import os

# --- 설정 ---
FILE_PATH = "voc_test.json"
CUSTOMER_NAMES = {
    "D001": "김민준", "D002": "박서연", "D003": "이도현", "D004": "최유진", "D005": "정지훈"
}

st.set_page_config(layout="wide", page_title="Churn-Zero AI")

# CSS 스타일 정의
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; color: white; }
    h3 { color: white !important; }
    .model-label { font-size: 1.0rem; font-weight: bold; color: #aaa; margin-bottom: 5px; }
    div.stButton > button {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #555 !important;
        text-align: left !important;
        font-weight: bold;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }
    .bubble { padding: 10px 15px; border-radius: 15px; display: inline-block; box-shadow: 1px 1px 2px #000; max-width: 80%; margin-bottom: 2px; }
    .bubble-user { background-color: #444; color: white; border-top-left-radius: 0px; }
    .bubble-ai { background-color: #f1c40f; color: black; border-top-right-radius: 0px; }
    .prob-text { font-size: 0.75rem; font-weight: bold; margin-top: 2px; margin-left: 5px; }
    </style>
    """, unsafe_allow_html=True)


# 데이터 로드 함수
@st.cache_data
def load_json_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for item in data:
        for conv in item.get('conversation', []):
            records.append({
                'dialog': item.get('dialog_id'),
                'seq': conv.get('seq'),
                '화자': conv.get('role'),
                'text': conv.get('text')
            })
    return pd.DataFrame(records)


df = load_json_data(FILE_PATH)

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 상담 내역")
    if not df.empty:
        for d_id in df['dialog'].unique():
            name = CUSTOMER_NAMES.get(d_id, "고객")
            if st.button(f"{name}", key=f"btn_{d_id}", use_container_width=True):
                st.session_state['selected_dialog'] = d_id

with col2:
    target_id = st.session_state.get('selected_dialog')
    if target_id:
        st.markdown(f"### {CUSTOMER_NAMES.get(target_id, '고객')}님과의 대화")

        st.markdown('<p class="model-label">모델 선택</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        is_lstm = c1.checkbox("LSTM", key=f"check_lstm_{target_id}")
        is_trans = c2.checkbox("Transformer Encoder", key=f"check_trans_{target_id}")

        # 모델별 확률 설정 (예시 데이터)
        model_results = []
        if is_lstm: model_results.append(("LSTM", 65))
        if is_trans: model_results.append(("Transformer Encoder", 72))

        conv = df[df['dialog'] == target_id].sort_values('seq')

        for _, row in conv.iterrows():
            if row['화자'] == '고객':
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-start;"><div class="bubble bubble-user">{row["text"]}</div></div>',
                    unsafe_allow_html=True)

                # 수직 나열 출력
                for model_name, prob in model_results:
                    prob_color = "#28a745" if prob <= 50 else "#ff4b4b"
                    st.markdown(f'<div class="prob-text" style="color: {prob_color};">{model_name} 이탈확률: {prob}%</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-end;"><div class="bubble bubble-ai">{row["text"]}</div></div>',
                    unsafe_allow_html=True)
                st.write("")
    else:
        st.info("좌측 목록에서 고객을 선택해주세요.")