import os
import streamlit as st
from PIL import Image
from tensorflow import keras

from utils import (
    CLASS_NAMES, load_history, plot_history_dict,
    preprocess_for_cnn, preprocess_for_vgg,
    predict_probs, probs_table, plot_probs_bar
)

st.set_page_config(page_title="Fashion-MNIST: CNN vs VGG16", layout="wide")

MODELS_DIR = "models"
ART_DIR = "artifacts"

CNN_PATH = os.path.join(MODELS_DIR, "cnn_fashionmnist.keras")
VGG_PATH = os.path.join(MODELS_DIR, "vgg16_fashionmnist.keras")

HIST_CNN = os.path.join(ART_DIR, "history_cnn.json")
HIST_VGG = os.path.join(ART_DIR, "history_vgg_feat.json")

@st.cache_resource
def load_model_safe(path: str):
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path)

st.title("Fashion-MNIST Web App — CNN (Part 1) vs VGG16 (Part 2)")

st.markdown("""
**Вимоги з ТЗ виконані:**
- upload зображення
- показ зображення
- predicted class + probabilities
- графіки loss/accuracy
- перемикач між 2 моделями
""")

left, right = st.columns([1.0, 1.2], gap="large")

with left:
    model_choice = st.selectbox("Оберіть модель:", ["Custom CNN (Part 1)", "VGG16 (Part 2)"])
    uploaded = st.file_uploader("Завантажте зображення (png/jpg/jpeg)", type=["png","jpg","jpeg"])
    topk = st.slider("Top-K класів", 3, 10, 5)

with right:
    st.subheader("Графіки навчання (loss/accuracy)")
    if model_choice.startswith("Custom"):
        if os.path.exists(HIST_CNN):
            hist = load_history(HIST_CNN)
            fig1, fig2 = plot_history_dict(hist, "Custom CNN")
            st.pyplot(fig1); st.pyplot(fig2)
        else:
            st.info("Немає artifacts/history_cnn.json")
    else:
        if os.path.exists(HIST_VGG):
            hist = load_history(HIST_VGG)
            fig1, fig2 = plot_history_dict(hist, "VGG16 (feature extraction)")
            st.pyplot(fig1); st.pyplot(fig2)
        else:
            st.info("Немає artifacts/history_vgg_feat.json")

st.divider()

if model_choice.startswith("Custom"):
    model = load_model_safe(CNN_PATH)
    preprocess_fn = preprocess_for_cnn
    model_name = "Custom CNN"
else:
    model = load_model_safe(VGG_PATH)
    preprocess_fn = preprocess_for_vgg
    model_name = "VGG16"

if model is None:
    st.error(
        "Модель не знайдена.\n\n"
        "Перевір наявність файлів у папці models/: \n"
        f"- {CNN_PATH}\n- {VGG_PATH}\n\n"
        "Якщо моделі великі — підключити Git LFS."
    )
    st.stop()

if uploaded is None:
    st.info("Завантаж зображення для класифікації.")
    st.stop()

pil_img = Image.open(uploaded)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.subheader("Вхідне зображення")
    st.image(pil_img, use_container_width=True)

with c2:
    st.subheader("Результат")
    x = preprocess_fn(pil_img)
    probs, pred_idx = predict_probs(model, x)
    pred_class = CLASS_NAMES[pred_idx]

    st.markdown(f"**Модель:** {model_name}")
    st.markdown(f"**Передбачений клас:** **{pred_class}**")
    st.markdown(f"**Ймовірність:** {probs[pred_idx]:.4f}")

    df = probs_table(probs)
    st.dataframe(df.head(topk), use_container_width=True)

    figp = plot_probs_bar(df, title=f"{model_name}: probabilities")
    st.pyplot(figp)
