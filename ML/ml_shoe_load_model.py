import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show error messages (suppress warnings)

import uuid
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
loaded_model = load_model("model.keras")

# Define class names
class_names = ['Cowboyboots', 'FlipFlops', 'Loafers', 'Sandals', 'Sneakers'] 

# Image URL (change this to test different images)
img_url = "https://pitaya.dk/cdn/shop/files/IMG_2868.jpg"

# Download image with unique filename to avoid cache
img_filename = str(uuid.uuid4()) + ".jpg"
img_path = tf.keras.utils.get_file(img_filename, img_url)

# Load and preprocess image
img = tf.keras.utils.load_img(img_path, target_size=(180, 180))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = tf.expand_dims(img_array, 0)  # Batch dimension

# Predict
predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Output
print(
    "This image most likely belongs to {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
