import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
import os
import cv2
import time
import zipfile
from io import BytesIO
from tensorflow.keras.models import load_model
from send_email import send_email_with_image
from heatmap import get_gradcam_heatmap_sequential, overlay_heatmap, generate_heatmap

st.set_page_config(page_title="UrbanEye", layout="centered")
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FFF2F2 !important;
        color: #2D336B !important;
    }

    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #2D336B !important;
    }

    .stButton>button {
        background-color: #A9B5DF !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    .stFileUploader, .stTextInput, .stSelectbox {
        background-color: #FFF2F2 !important;
        color: #2D336B !important;
    }
            
    </style>
    """, unsafe_allow_html=True)

### parameters
MODEL_PATH = "urbanEye_model.keras"
CLASSES_PATH = "classes.txt"
SOLUTIONS_PATH = "solutions.txt"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "conv5_block3_out"

### persistent temp folder -> for heatmap image
PERSISTENT_TMP_DIR = os.path.join(tempfile.gettempdir(), "streamlit_uploaded_images")
os.makedirs(PERSISTENT_TMP_DIR, exist_ok=True)

### here, we used streamlit's caching system -> to store the resources and improve perfomarce
@st.cache_resource
def load_class_labels():
    with open(CLASSES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)
@st.cache_resource
def load_solutions():
    solution_dict = {}
    with open(SOLUTIONS_PATH, "r") as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(":", 1)
                solution_dict[key.strip()] = value.strip()
    return solution_dict

### read class labels and model
class_labels = load_class_labels()
model = load_trained_model()
solutions = load_solutions()

### prediction function
def predict_image(image_path, model, class_names, img_size=(224, 224)):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    start_time = time.time()  ### start timing
    predictions = model.predict(img_array, verbose=0)
    end_time = time.time()    ### end timing

    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    duration = end_time - start_time
    return img, predicted_idx, confidence, predictions[0], duration

### gui
st.title("UrbanEye")
st.subheader("Upload a single image or a .zip folder with multiple images to get predictions.")

uploaded_file = st.file_uploader("Choose an image or ZIP folder", type=["jpg", "jpeg", "png", "zip"])

### session state initialization
if "show_heatmap_for" not in st.session_state:
    st.session_state["show_heatmap_for"] = None
if "heatmap_image_path" not in st.session_state:
    st.session_state["heatmap_image_path"] = None

if uploaded_file is not None:
    ### case1: when the user uploads only one image
    if not uploaded_file.name.endswith(".zip"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        original_image = Image.open(tmp_file_path).convert("RGB")
        st.image(original_image, caption="Uploaded Image", use_container_width=True)

        _, predicted_idx, confidence, full_pred, duration = predict_image(tmp_file_path, model, class_labels, IMG_SIZE)

        st.markdown(f"**Prediction:** {class_labels[predicted_idx]}")
        predicted_class = class_labels[predicted_idx]
        if predicted_class in solutions:
            st.markdown(f"**Suggested Action:** {solutions[predicted_class]}")
        st.markdown(f"**Confidence:** {confidence:.2%}")
        st.markdown(f"**Prediction Time:** {duration:.2f} seconds")

        if st.button("Show All Class Scores"):
            st.subheader("All Class Scores:")
            for i, score in enumerate(full_pred):
                st.write(f"{class_labels[i]}: {score:.2%}")

        ### render heatmap
        if st.button("Show Heatmap for Uploaded Image"):
            with st.spinner("Generating heatmap..."):
                heatmap_img, label, conf = generate_heatmap(tmp_file_path, class_labels, model, IMG_SIZE, LAST_CONV_LAYER)
                st.image(heatmap_img, caption=f"Grad-CAM: {label} ({conf:.2%})", use_container_width=True)

        ### send an email to the authorities if the user click on button
        st.subheader("Do you want to report the problem to the authorities?")
        with st.form("email_form"):
            location = st.text_input("Location of the problem")
            risk_level = st.selectbox("Risk Level", ["Select Risk Level", "Low", "Medium", "High"])
            user_message = st.text_area("Write other details  if you want.")
            send_btn = st.form_submit_button("Send email")

            if send_btn:
                if not location.strip():
                    st.warning("Please enter the location of the problem.")
                elif risk_level == "Select Risk Level":
                    st.warning("Please select a risk level.")
                else:
                    subject = "UrbanEye Report: Model Prediction and Image with problem from the city."
                    body = f"Predicted Issue: {class_labels[predicted_idx]}\nConfidence: {confidence:.2%}\n\nUser Message:\n{user_message}"
                    success = send_email_with_image(subject, body, tmp_file_path)
                    if success:
                        st.success("Email sent successfully.")
                    else:
                        st.error("For now this function is not activated. It will be in the feature.")
                        #st.error("Failed to send email. Check logs or configuration.")

    ### case2: when the user uploads a zip with multiple images
    else:
        ### clear persistent folder
        for f in os.listdir(PERSISTENT_TMP_DIR):
            os.remove(os.path.join(PERSISTENT_TMP_DIR, f))
        ### extract and persist files
        with zipfile.ZipFile(BytesIO(uploaded_file.read()), "r") as zip_ref:
            zip_ref.extractall(PERSISTENT_TMP_DIR)

        image_files = [
            os.path.join(PERSISTENT_TMP_DIR, f)
            for f in os.listdir(PERSISTENT_TMP_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            st.warning("No valid image files found in the ZIP. Please make sure than the ZIP contains only the images, not also the folder.")
        else:
            st.subheader(f"Found {len(image_files)} image(s). Processing...")

            for idx, image_path in enumerate(image_files):
                image_key = f"image_{idx}"
                cols = st.columns([1, 2])

                with cols[0]:
                    original_image = Image.open(image_path).convert("RGB")
                    st.image(original_image, caption=os.path.basename(image_path), use_container_width=True)

                with cols[1]:
                    _, predicted_idx, confidence, _, duration = predict_image(image_path, model, class_labels, IMG_SIZE)
                    st.markdown(f"**Prediction:** {class_labels[predicted_idx]}")
                    predicted_class = class_labels[predicted_idx]
                    if predicted_class in solutions:
                        st.markdown(f"**Suggested Action:** {solutions[predicted_class]}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    st.markdown(f"**Prediction Time:** {duration:.2f} seconds")

                    button_label = f"Show Heatmap for {os.path.basename(image_path)}"
                    if st.button(button_label, key=f"btn_{idx}"):
                        st.session_state["show_heatmap_for"] = image_key
                        st.session_state["heatmap_image_path"] = image_path
                        st.rerun()

                ### render heatmap
                if st.session_state.get("show_heatmap_for") == image_key:
                    with st.spinner("Generating heatmap..."):
                        heatmap_img, label, conf = generate_heatmap(st.session_state["heatmap_image_path"], class_labels, model, IMG_SIZE, LAST_CONV_LAYER)
                        st.image(heatmap_img, caption=f"Grad-CAM: {label} ({conf:.2%})", use_container_width=True)
