import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 64

# Load test data
(_, _), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_test = y_test.squeeze()

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load saved fine-tuned model
model = tf.keras.models.load_model("models/cifar10_mobilenet_finetuned.keras")

# Evaluate
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")