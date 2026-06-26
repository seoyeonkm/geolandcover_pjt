"""Streamlit app entry for land cover inference demo."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Ensure imports work even when running from the app directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_proba


st.set_page_config(page_title="Land Cover Classifier", page_icon="🛰️", layout="centered")
st.title("Satellite Land Cover Classifier")
st.write("위성사진 이미지를 업로드하면 예측 클래스와 클래스별 확률을 보여줍니다.")

default_checkpoint_path = str(
    PROJECT_ROOT / "outputs" / "checkpoints" / "best_cnn_eurosat.pt"
)

checkpoint_path = st.text_input(
    "모델 체크포인트 경로",
    value=default_checkpoint_path,
)

if not Path(checkpoint_path).exists():
    st.warning("체크포인트 파일이 없습니다. 먼저 학습을 완료해 주세요.")

uploaded = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("예측 실행"):
        with st.spinner("모델 추론 중..."):
            try:
                pred_class, prob_by_class = predict_proba(
                    image=image,
                    checkpoint_path=checkpoint_path,
                )
            except FileNotFoundError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"추론 중 오류가 발생했습니다: {error}")
            else:
                st.success(f"예측 클래스: {pred_class}")

                prob_df = pd.DataFrame(
                    {
                        "class": list(prob_by_class.keys()),
                        "probability": list(prob_by_class.values()),
                    }
                ).set_index("class")

                st.subheader("클래스별 확률")
                st.bar_chart(prob_df)

                st.dataframe(
                    (prob_df * 100).round(2).rename(columns={"probability": "percent(%)"})
                )
