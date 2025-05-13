import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os
import zipfile
from io import BytesIO
import cv2
from heatmap import get_gradcam_heatmap_sequential, overlay_heatmap

st.set_page_config(page_title="UrbanEye", layout="centered")

# Custom UI style
st.markdown("""
    <style>
    body {
        background-color: #FFF2F2;
    }
    .stApp {
        background-color: #FFF2F2;
    }
    h1, h2, h3, h4 {
        color: #2D336B;
    }
    .stButton>button {
        background-color: #A9B5DF;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

MODEL_PATH = "urbanEye_model.keras"  #MODEL_PATH = "my_resnet_model.keras"
CLASSES_PATH = "classes.txt"
IMG_SIZE = (224, 224)#IMG_SIZE = (185, 628)
LAST_CONV_LAYER = "conv5_block3_out"  # adjust to your model

# Persistent temp folder
PERSISTENT_TMP_DIR = os.path.join(tempfile.gettempdir(), "streamlit_uploaded_images")
os.makedirs(PERSISTENT_TMP_DIR, exist_ok=True)

@st.cache_resource
def load_class_labels():
    with open(CLASSES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

class_labels = load_class_labels()

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Prediction function
def predict_image(image_path, model, class_names, img_size=(224, 224)):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    return img, predicted_idx, confidence, predictions[0]

# Grad-CAM heatmap rendering
def generate_heatmap(image_path):
    base_model = model.layers[0]
    classifier_head = tf.keras.Sequential(model.layers[1:])

    img_raw = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img_raw)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array.copy())

    preds = model.predict(np.expand_dims(img_preprocessed, axis=0), verbose=0)
    pred_label = np.argmax(preds[0])

    heatmap = get_gradcam_heatmap_sequential(base_model, classifier_head, img_preprocessed, pred_label, LAST_CONV_LAYER)
    superimposed = overlay_heatmap(img_array, heatmap)

    return superimposed, class_labels[pred_label], preds[0][pred_label]

# GUI
st.title("UrbanEye")
st.subheader("Upload a single image or a .zip folder with multiple images to get predictions.")

uploaded_file = st.file_uploader("Choose an image or ZIP folder", type=["jpg", "jpeg", "png", "zip"])

# Session state initialization
if "show_heatmap_for" not in st.session_state:
    st.session_state["show_heatmap_for"] = None
if "heatmap_image_path" not in st.session_state:
    st.session_state["heatmap_image_path"] = None

if uploaded_file is not None:
    if uploaded_file.name.endswith(".zip"):
        # Clear persistent folder
        for f in os.listdir(PERSISTENT_TMP_DIR):
            os.remove(os.path.join(PERSISTENT_TMP_DIR, f))

        # Extract and persist files
        with zipfile.ZipFile(BytesIO(uploaded_file.read()), "r") as zip_ref:
            zip_ref.extractall(PERSISTENT_TMP_DIR)

        image_files = [
            os.path.join(PERSISTENT_TMP_DIR, f)
            for f in os.listdir(PERSISTENT_TMP_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            st.warning("No valid image files found in the ZIP.")
        else:
            st.subheader(f"Found {len(image_files)} image(s). Processing...")

            for idx, image_path in enumerate(image_files):
                image_key = f"image_{idx}"
                cols = st.columns([1, 2])

                with cols[0]:
                    original_image = Image.open(image_path).convert("RGB")
                    st.image(original_image, caption=os.path.basename(image_path), use_container_width=True)

                with cols[1]:
                    _, predicted_idx, confidence, _ = predict_image(image_path, model, class_labels, IMG_SIZE)
                    st.markdown(f"**Prediction:** {class_labels[predicted_idx]}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")

                    button_label = f"Show Heatmap for {os.path.basename(image_path)}"
                    if st.button(button_label, key=f"btn_{idx}"):
                        st.session_state["show_heatmap_for"] = image_key
                        st.session_state["heatmap_image_path"] = image_path
                        st.rerun()

                # Render heatmap if this image was selected
                if st.session_state.get("show_heatmap_for") == image_key:
                    with st.spinner("Generating heatmap..."):
                        heatmap_img, label, conf = generate_heatmap(st.session_state["heatmap_image_path"])
                        st.image(heatmap_img, caption=f"Grad-CAM: {label} ({conf:.2%})", use_container_width=True)

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        original_image = Image.open(tmp_file_path).convert("RGB")
        st.image(original_image, caption="Uploaded Image", use_container_width=True)

        _, predicted_idx, confidence, full_pred = predict_image(tmp_file_path, model, class_labels, IMG_SIZE)

        st.markdown(f"**Prediction:** {class_labels[predicted_idx]}")
        st.markdown(f"**Confidence:** {confidence:.2%}")

        st.subheader("All Class Scores:")
        for i, score in enumerate(full_pred):
            st.write(f"{class_labels[i]}: {score:.2%}")

        if st.button("Show Heatmap for Uploaded Image"):
            with st.spinner("Generating heatmap..."):
                heatmap_img, label, conf = generate_heatmap(tmp_file_path)
                st.image(heatmap_img, caption=f"Grad-CAM: {label} ({conf:.2%})", use_container_width=True)
