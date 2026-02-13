import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os


# Load CIFAR-10


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = y_train.squeeze()
y_test = y_test.squeeze()


# Train / Validation Split


VAL_SPLIT = 0.1
val_size = int(len(X_train) * VAL_SPLIT)

X_val = X_train[:val_size]
y_val = y_train[:val_size]

X_train = X_train[val_size:]
y_train = y_train[val_size:]


# Dataset Pipeline


BATCH_SIZE = 64
IMG_SIZE = 224

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(45000)
    .map(preprocess)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(preprocess)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# Load Previously Trained Model


model = tf.keras.models.load_model(
    "models/cifar10_mobilenet_feature_extractor.keras"
)


# Unfreeze Last 30 Layers


base_model = model.layers[0]

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

print("Number of trainable layers:",
      sum([layer.trainable for layer in base_model.layers]))


# Recompile with Lower LR


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Fine-Tune


history = model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds
)


# Save Fine-Tuned Model


os.makedirs("models", exist_ok=True)
model.save("models/cifar10_mobilenet_finetuned.keras")

print("Fine-tuned model saved successfully.")