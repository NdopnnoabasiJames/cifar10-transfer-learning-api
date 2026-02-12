import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image