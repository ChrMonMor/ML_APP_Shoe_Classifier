import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import shutil
import os
import keras
print(tf.__version__)
print(keras.__version__)


# Load your trained model
model = tf.keras.models.load_model("model.keras")


# Categories
CATEGORIES = ['Cowboyboots', 'FlipFlops', 'Loafers', 'Sandals', 'Sneakers']

st.title("Image Classifier & Sorter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image for the model
    img_resized = img.resize((180, 180))  # Change to your model's input size
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CATEGORIES[np.argmax(predictions)]
    st.write(f"**Model Prediction:** {predicted_class}")
    score = tf.nn.softmax(predictions[-1])
    # Output
    for i, class_name in enumerate(CATEGORIES):
        st.write(f"{class_name}: {100 * score[i]:.2f}%")

    # Optionally still print the top result
    st.write(
        "\nTop prediction: {} ({:.2f}%)"
        .format(CATEGORIES[np.argmax(score)], 100 * np.max(score))
    )

    # Let user choose folder
    category_choice = st.selectbox("Select category to save image:", CATEGORIES)

    if st.button("Move image to folder"):
        save_path = os.path.join("Shoes", category_choice, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Image saved to {category_choice} folder!")
