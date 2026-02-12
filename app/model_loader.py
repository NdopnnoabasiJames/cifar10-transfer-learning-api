import tensorflow as tf

MODEL_PATH = "models/mobilenet_cifar.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]