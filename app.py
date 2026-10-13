
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import torch

st.set_page_config(
    page_title="Fresh vs Rotten Fruits AI",
    page_icon="??",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0fdf4, #ffffff);
}
.hero {
    background: linear-gradient(120deg, #14532d, #16a34a);
    padding: 28px;
    border-radius: 18px;
    color: white;
    text-align: center;
}
</style>
<div class="hero">
<h1>?? Fresh vs Rotten Fruits AI</h1>
<p>YOLO11-powered fruit freshness detection</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best.pt"
    return YOLO(str(model_path))

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

st.write("")
uploaded = st.file_uploader(
    "Fruit ki image upload karo",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.image(image, caption="Uploaded Fruit",
                 use_container_width=True)

    with right:
        if st.button("?? Analyze Fruit", type="primary"):
            with st.spinner("AI image analyze kar raha hai..."):
                result = model.predict(
                    image,
                    imgsz=224,
                    device="cpu",
                    verbose=False
                )[0]

            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            label = str(result.names[class_id])

            st.subheader("Prediction Result")
            st.metric("Class", label.replace("_", " ").title())
            st.metric("Confidence", f"{confidence * 100:.2f}%")

            if label.lower().startswith("fresh"):
                st.success("?? Fresh Fruit")
            elif label.lower().startswith("rotten"):
                st.error("?? Rotten Fruit")
            else:
                st.warning("Result verify karein.")

            st.write("### All class probabilities")
            for name, score in zip(
                result.names.values(),
                result.probs.data.cpu().tolist()
            ):
                st.write(f"**{name}** � {score * 100:.2f}%")
                st.progress(float(score))

st.caption("YOLO11 classification | Predictions depend on model quality.")
