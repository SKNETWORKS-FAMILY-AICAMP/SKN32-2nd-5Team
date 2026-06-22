import streamlit as st
import pandas as pd
import os

# --- 설정 ---
FILE_PATH = "voc_test.csv"
CUSTOMER_NAMES = {
    "D001": "김민준", "D002": "박서연", "D003": "이도현", "D004": "최유진", "D005": "정지훈"
}

st.set_page_config(layout="wide", page_title="Churn-Zero AI")

# CSS: 마우스 호버 시 파란색으로 변경되는 스타일 추가
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }

    /* 버튼 스타일 재정의 (호버 시 파란색) */
    div.stButton > button {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ddd !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }

    /* 채팅 스타일 */
    .bubble { padding: 10px 15px; border-radius: 15px; display: inline-block; box-shadow: 1px 1px 2px #ccc; max-width: 80%; margin-bottom: 2px; }
    .bubble-user { background-color: white; color: black; border-top-left-radius: 0px; }
    .bubble-ai { background-color: #fee500; color: black; border-top-right-radius: 0px; }
    .prob-text { font-size: 0.65rem; color: #ff4b4b; font-weight: bold; margin-bottom: 10px; margin-left: 5px; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data(path):
    return pd.read_csv(path, encoding='utf-8', on_bad_lines='warn', engine='python')


df = load_data(FILE_PATH) if os.path.exists(FILE_PATH) else pd.DataFrame()

col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("<h3 style='color:black;'>상담 내역</h3>", unsafe_allow_html=True)
    # 버튼을 리스트 형태로 출력
    for d_id in df['dialog'].unique():
        name = CUSTOMER_NAMES.get(d_id, "고객")
        if st.button(name, key=f"btn_{d_id}", use_container_width=True):
            st.session_state['selected_dialog'] = d_id

with col2:
    target_id = st.session_state.get('selected_dialog')
    if target_id:
        st.markdown(f"<h3 style='color:black;'>{CUSTOMER_NAMES.get(target_id, '고객')}님과의 대화</h3>",
                    unsafe_allow_html=True)
        conv = df[df['dialog'] == target_id].sort_values('seq')
        for _, row in conv.iterrows():
            if row['화자'] == '고객':
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-start;"><div class="bubble bubble-user">{row["text"]}</div></div>',
                    unsafe_allow_html=True)
                st.markdown('<div class="prob-text">이탈 확률: 65%</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-end;"><div class="bubble bubble-ai">{row["text"]}</div></div>',
                    unsafe_allow_html=True)
                st.write("")
    else:
        st.info("좌측 목록에서 고객을 선택해주세요.")