import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil

# --- CONFIGURATION ---
MODEL_PATH = "model.ftlite"  # Path to your trained model
IMAGE_SIZE = (224, 224)  # Replace with the size your model expects
LABELS = ['Cowboyboots', 'FlipFlops', 'Loafers', 'Sandals', 'Sneakers']  # Replace with your actual class names
SAVE_FOLDER = "sorted_shoes"

# --- LOAD MODEL ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- FUNCTION TO PREDICT ---
def predict_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize if needed
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    return LABELS[class_idx]

# --- STREAMLIT UI ---
st.title("Shoe Classifier and Organizer")

uploaded_file = st.file_uploader("Upload a shoe image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_label = predict_image(image)
    st.write(f"### Model Prediction: `{predicted_label}`")

    final_label = st.selectbox("Is the prediction correct?", LABELS, index=LABELS.index(predicted_label))

    if st.button("Save Image"):
        save_path = os.path.join(SAVE_FOLDER, final_label)
        os.makedirs(save_path, exist_ok=True)

        # Save the uploaded image
        image_filename = uploaded_file.name
        image.save(os.path.join(save_path, image_filename))

        st.success(f"Image saved to `{save_path}`!")

