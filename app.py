import streamlit as st
from PIL import Image

from train import load_model, predict


st.set_page_config(page_title="위성 토지피복 분류기", layout="centered")
st.title("AI 위성사진 토지피복 분류기")
st.write("위성 이미지를 업로드하면 CNN 모델이 토지 유형을 분류합니다.")

# 모델 로드 (앱 시작 시 한 번만)
@st.cache_resource
def get_model():
    return load_model()

try:
    model, class_names, device = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# 이미지 업로드
uploaded = st.file_uploader("위성 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="업로드된 이미지", use_container_width=True)

    with col2:
        if st.button("분류하기"):
            with st.spinner("분석 중..."):
                pred_class, probs = predict(image, model, class_names, device)

            st.success(f"예측 결과: **{pred_class}**")
            st.subheader("클래스별 확률")
            st.bar_chart(probs)
