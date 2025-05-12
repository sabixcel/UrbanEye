import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

st.set_page_config(page_title="Image Classifier", layout="centered")
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

#config
MODEL_PATH = "urbanEye_model.keras"
CLASSES_PATH = "classes.txt"
IMG_SIZE = (224, 224)

#load class names
@st.cache_resource
def load_class_labels():
    with open(CLASSES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

class_labels = load_class_labels()

#load model
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

#prediction function
def predict_image(image_path, model, class_names, img_size=(224, 224)):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    return img, predicted_idx, confidence, predictions[0]

#interface
st.title("📷 Image Classifier")
st.subheader("Upload an image to get a prediction from the model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #save uploaded image to a temporary file (to display it on the gui)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    original_image = Image.open(tmp_file_path).convert("RGB")
    st.image(original_image, caption="Uploaded Image", use_column_width=True)
    _, predicted_idx, confidence, full_pred = predict_image(tmp_file_path, model, class_labels, IMG_SIZE)

    st.markdown(f"<h4 style='color:#7886C7;'>Prediction: {class_labels[predicted_idx]}</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#A9B5DF;'>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)

    st.subheader("All Class Scores:")
    for i, score in enumerate(full_pred):
        st.write(f"{class_labels[i]}: {score:.2%}")
