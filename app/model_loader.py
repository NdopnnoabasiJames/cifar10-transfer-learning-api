import tensorflow as tf

MODEL_PATH = "models/cifar10_mobilenet_feature_extractor.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]