import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from model.model_download import download_model

# ✅ Load the model once
@st.cache_resource
def load_model():
    model_path = download_model()
    return tf.keras.models.load_model(model_path)

model = load_model()

# ✅ Define class labels
class_name = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
    'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
    'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
    'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
    'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
    'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ✅ Prediction function
def model_predict(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        return np.argmax(prediction), np.max(prediction)
    except Exception as e:
        return None, str(e)

# ✅ Streamlit App UI
st.sidebar.title("🌿 Plant Disease Detector")
mode = st.sidebar.selectbox("Mode", ["Home", "Detect"])

if mode == "Home":
    st.markdown("## 🌱 Plant Disease Detection for Sustainable Agriculture")
    st.write("Upload a plant image to check for disease using a trained deep learning model.")

elif mode == "Detect":
    st.header("Upload Leaf Image for Detection")
    test_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if test_image:
        path = test_image.name
        with open(path, "wb") as f:
            f.write(test_image.getbuffer())

        st.image(test_image, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            idx, conf = model_predict(path)
            if idx is not None:
                st.success(f"Prediction: {class_name[idx]}")
                st.info(f"Confidence Score: {conf:.2f}")
            else:
                st.error(f"Prediction failed: {conf}")
            os.remove(path)