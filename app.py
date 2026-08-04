import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌱",
    layout="wide"
)

# Load Local Trained Model
MODEL_PATH = "model/plant_disease_model.h5"

@st.cache_resource
def load_plant_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_plant_model()

# Class Names matching PlantVillage dataset
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Local Care Guidelines (No External API Required)
TREATMENTS = {
    'Early_blight': "Apply copper-based fungicides. Ensure crop rotation and remove infected lower leaves.",
    'Late_blight': "Use systemic fungicides immediately (e.g., Mancozeb). Avoid overhead irrigation and burn infected stems.",
    'Bacterial_spot': "Spray copper hydroxide mixed with mancozeb. Avoid working in fields when foliage is wet.",
    'Leaf_Mold': "Improve greenhouse ventilation, reduce humidity below 85%, and apply suitable fungicides.",
    'Septoria_leaf_spot': "Remove affected leaves. Apply organic copper or sulfur sprays every 7–10 days.",
    'Spider_mites': "Apply insecticidal soap or neem oil. Ensure adequate leaf moisture to discourage mite buildup.",
    'YellowLeaf__Curl_Virus': "Control whitefly vectors using yellow sticky traps and insecticidal sprays. Remove infected plants.",
    'mosaic_virus': "No cure available once infected. Disinfect tools with bleach solution and rogue out infected plants.",
    'healthy': "No disease detected! Maintain regular watering, proper soil nutrition, and periodic inspection."
}

def get_treatment(disease_name):
    for key, advice in TREATMENTS.items():
        if key in disease_name:
            return advice
    return "Ensure balanced irrigation, proper soil nutrition, and consult a local agricultural extension specialist."

# UI Layout
st.title("🌱 Crop Disease Detection & Care Assistant")
st.write("Upload a leaf image to receive instantaneous disease diagnosis using your locally trained MobileNetV2 model.")

st.sidebar.header("Image Upload")
uploaded_file = st.sidebar.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    
    with col2:
        if model is None:
            st.error(f"Model file not found at `{MODEL_PATH}`. Please ensure `plant_disease_model.h5` is placed inside the `model/` directory.")
        else:
            with st.spinner("Analyzing image using local model..."):
                # Preprocess image
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict using offline TensorFlow model
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                
                disease_label = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
                clean_name = disease_label.replace("___", " - ").replace("_", " ")

            st.success(f"**Diagnosis:** {clean_name}")
            st.info(f"**Model Confidence:** {confidence:.2f}%")
            
            st.subheader("📋 Recommended Treatment Plan")
            treatment_advice = get_treatment(disease_label)
            st.write(treatment_advice)
else:
    st.info("👈 Please upload a leaf image from the sidebar to start diagnosis.")