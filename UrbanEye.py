import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
import tempfile
import zipfile
import os
import time
import base64
import random
from PIL import Image
from io import BytesIO
from geopy.geocoders import Nominatim
#from streamlit_geolocation import streamlit_geolocation
from tensorflow.keras.models import load_model
from send_email import send_email_with_image
from heatmap2 import get_gradcam_heatmap_sequential, overlay_heatmap

st.set_page_config(page_title="UrbanEye", layout="wide")
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
def predict_image(image_path, model, class_names, img_size=IMG_SIZE):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    preprocessed_img = tf.expand_dims(img_array, axis=0)

    start_time = time.time()  ### start timing
    predictions = model.predict(preprocessed_img, verbose=0)
    end_time = time.time()    ### end timing

    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    duration = end_time - start_time

    return img, predicted_idx, confidence, predictions[0], preprocessed_img, img_array, duration

### heatmap function
def generate_heatmap(preprocessed_img, original_img_array, pred_label_idx):
    base_model = model.layers[0]
    classifier_head = tf.keras.Sequential(model.layers[1:])

    heatmap = get_gradcam_heatmap_sequential(base_model, classifier_head, preprocessed_img[0], pred_label_idx, LAST_CONV_LAYER)
    superimposed = overlay_heatmap(original_img_array, heatmap)

    return superimposed

### get random same color for a class
@st.cache_resource
def generate_class_colors(class_labels):
    random.seed(42)
    colors = {}
    for label in class_labels:
        colors[label] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return colors
CLASS_COLORS = generate_class_colors(class_labels)
def get_color_for_class(class_name):
    return CLASS_COLORS.get(class_name, "#7886C7")  ### fallback color

### convert a numpy array into a PIL image (Python Imaging Library)
def numpy_to_pil(img_array):
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

### gui
st.markdown(
    "<h1 style='text-align: center; color: #000000;'>UrbanEye🕵🏻‍♂️</h1><br>",
    unsafe_allow_html=True,
)
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
        col1, col2, col3 = st.columns([3, 3, 3])

        with col1:
            DISPLAY_HEIGHT = 500
            w, h = original_image.size
            aspect_ratio = w / h
            new_width = int(DISPLAY_HEIGHT * aspect_ratio)
            resized_image = original_image.resize((new_width, DISPLAY_HEIGHT))

            st.image(resized_image)
            _, predicted_idx, confidence, full_pred, preprocessed_img, raw_img_array, duration = predict_image(tmp_file_path, model, class_labels, IMG_SIZE)
            predicted_class = class_labels[predicted_idx]
            color = get_color_for_class(predicted_class)

            st.markdown(
                f"<div style='background-color:{color}; color: white; padding: 5px; border-radius:5px; text-align:center; width: 100%;'>"
                f"{os.path.basename(uploaded_file.name)}</div>",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(f"**Prediction:** {class_labels[predicted_idx]}")
            predicted_class = class_labels[predicted_idx]
            ### user feedback
            feedback = st.radio(f"Is this image correctly labeled as {predicted_class}?", ["Yes", "No"])
            if feedback == "Yes":
                ### show the suggested action
                if predicted_class in solutions:
                    st.markdown(f"**Suggested Action:** {solutions[predicted_class]}")
            else:
                ### show correction input
                corrected_label = st.selectbox("Please correct the label:", options=class_labels)
                if st.button("Submit Correction"):
                    st.success(f"Thanks! You selected: `{corrected_label}`")
                    st.markdown(f"**Suggested Action for '{corrected_label}':** {solutions[corrected_label]}")

            st.markdown(f"**Confidence:** {confidence:.2%}")
            st.markdown(f"**Prediction Time:** {duration:.2f} seconds")

            ### show all classes scores
            if st.button("Show All Class Scores"):
                st.subheader("All Class Scores:")
                for i, score in enumerate(full_pred):
                    st.write(f"{class_labels[i]}: {score:.2%}")

            ### render heatmap
            if st.button("Show Heatmap for Uploaded Image"):
                with st.spinner("Generating heatmap..."):
                    heatmap_img = generate_heatmap(preprocessed_img, raw_img_array, predicted_idx)
                    st.image(heatmap_img, caption=f"HeatMap: {predicted_class} ({confidence:.2%})", use_container_width=True)

        with col3:
            ### send an email to the authorities if the user click on button
            st.subheader("Do you want to report the problem to the authorities?")
            # st.markdown("Click to allow your location usage:")
            # location = streamlit_geolocation()
            # ok = 0
            # if st.button("Find Location"):
            #     geolocator = Nominatim(user_agent="streamlit-app")
            #     location = geolocator.reverse((location['latitude'], location['longitude']), exactly_one=True)
            #     ok = 1
            # value = location.address if ok else ""
            with st.form("email_form"):
                location_btn = st.text_input("Enter location:", value="")
                severity_level = st.selectbox("Severity Level", ["Select Severity Level", "Low", "Medium", "High"])
                user_message = st.text_area("Write other details  if you want.")
                send_btn = st.form_submit_button("Send email")

                if send_btn:
                    if not location_btn.strip():
                        st.warning("Please enter the location of the problem.")
                    if severity_level == "Select severity Level":
                        st.warning("Please select a severity level.")
                    else:
                        subject = "UrbanEye Report: Model Prediction and Image with problem from the city."
                        body = f"Predicted Issue: {class_labels[predicted_idx]}\nConfidence: {confidence:.2%}\nLocation: {location_btn}\nSeverity Level: {severity_level}\n\nUser Message:\n{user_message}"
                        success = send_email_with_image(subject, body, tmp_file_path)
                        if success:
                            st.success("Email sent successfully.")
                        else:
                            st.success("For now this function is not activated. It will be in the future.☺️")
                            #st.error("Failed to send email. Check logs or configuration.")

    ### case2: when the user uploads a zip with multiple images
    else:
        with tempfile.TemporaryDirectory() as temp_zip_dir:
            with zipfile.ZipFile(BytesIO(uploaded_file.read()), "r") as zip_ref:
                ### get file list in ZIP in order
                zip_file_names = [f for f in zip_ref.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                if not zip_file_names:
                    st.warning("No valid image files found in the ZIP.")
                else:
                    st.subheader(f"Found {len(zip_file_names)} image(s). Processing...")

                    ### extract files in order
                    image_paths = []
                    for f in zip_file_names:
                        extracted_path = zip_ref.extract(f, path=temp_zip_dir)
                        image_paths.append(extracted_path)

                    ### display images in 3 columns and with the same size
                    cols = st.columns(3)
                    DISPLAY_HEIGHT = 300
                    for idx, image_path in enumerate(image_paths):
                        col = cols[idx % 3]
                        with col:
                            ### load original image
                            original_image = Image.open(image_path).convert("RGB")
                            w, h = original_image.size
                            aspect_ratio = w / h
                            new_width = int(DISPLAY_HEIGHT * aspect_ratio)
                            resized_image = original_image.resize((new_width, DISPLAY_HEIGHT))

                            ### predict
                            _, predicted_idx, confidence, full_pred, preprocessed_img, raw_img_array, duration = predict_image(image_path, model, class_labels, IMG_SIZE)
                            predicted_class = class_labels[predicted_idx]
                            color = get_color_for_class(predicted_class)

                            unique_key = f"show_heatmap_{idx}"

                            ### initialize session state toggle if not present
                            if unique_key not in st.session_state:
                                st.session_state[unique_key] = False  ### False = show original image, True = show heatmap

                            ### button to toggle image/heatmap
                            if st.button("Toggle Heatmap", key=f"btn_{unique_key}"):
                                st.session_state[unique_key] = not st.session_state[unique_key]

                            ### show either image or heatmap based on toggle
                            if st.session_state[unique_key]:
                                with st.spinner("Generating heatmap..."):
                                    heatmap_img = generate_heatmap(preprocessed_img, raw_img_array, predicted_idx)
                                    heatmap_pil = numpy_to_pil(heatmap_img)

                                    ### resize heatmap to fixed height & maintain aspect ratio
                                    w_hm, h_hm = heatmap_pil.size
                                    aspect_ratio_hm = w_hm / h_hm
                                    resized_heatmap = heatmap_pil.resize((int(DISPLAY_HEIGHT * aspect_ratio_hm), DISPLAY_HEIGHT))

                                    ### show heatmap centered
                                    st.image(resized_heatmap, caption="Heatmap", use_container_width=False)
                            else:
                                ### show original image centered
                                st.image(resized_image, use_container_width=False)

                            ### below image/heatmap info & prediction
                            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
                            st.markdown(
                                f"<div style='background-color:{color}; color: white; padding: 5px; border-radius:5px; text-align:center;'>"
                                f"{os.path.basename(image_path)}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Prediction:** {class_labels[predicted_idx]}")
                            predicted_class = class_labels[predicted_idx]
                            ### user feedback
                            feedback = st.radio(f"Is this image correctly labeled as {predicted_class}?", ["Yes", "No"], key=f"radio_{unique_key}")
                            if feedback == "Yes":
                                ### show the suggested action
                                if predicted_class in solutions:
                                    st.markdown(f"**Suggested Action:** {solutions[predicted_class]}")
                            else:
                                ### show correction input
                                corrected_label = st.selectbox("Please correct the label:", options=class_labels)
                                if st.button("Submit Correction", key=f"btn2_{unique_key}"):
                                    st.success(f"Thanks! You selected: `{corrected_label}`")
                                    st.markdown(f"**Suggested Action for '{corrected_label}':** {solutions[corrected_label]}")
                            st.markdown(f"**Confidence:** {confidence:.2%}")
                            st.markdown(f"**Prediction Time:** {duration:.2f} seconds")
