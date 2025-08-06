import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
loaded_model = load_model("model.keras")  # <- FIXED LINE

# Define class names
class_names = ['Cowboyboots', 'FlipFlops', 'Loafers', 'Sandals', 'Sneakers'] 

# Predict on new data
sandal_url = "https://peti-sko.dk/cdn/shop/files/gul-line-sandal-woden-peti-sko-813876_600x.webp"
sandal_path = tf.keras.utils.get_file('Gul_sandal', origin=sandal_url)

img = tf.keras.utils.load_img(sandal_path, target_size=(180, 180))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
