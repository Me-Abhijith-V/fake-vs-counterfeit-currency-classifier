
from pathlib import Path
import json

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 25
SEED = 42

# Class 0 = fake, class 1 = real
CLASS_NAMES = ["fake", "real"]

MODEL_PATH = BASE_DIR / "currency_cnn.keras"

tf.keras.utils.set_random_seed(SEED)


# ---------------- LOAD DATASET ----------------

def load_dataset(folder_name, shuffle):
    folder = DATASET_DIR / folder_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    return tf.keras.utils.image_dataset_from_directory(
        folder,
        class_names=CLASS_NAMES,
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


train_ds = load_dataset("training", True)
val_ds = load_dataset("validation", False)
test_ds = load_dataset("testing", False)

print("Class mapping:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ---------------- BUILD CNN ----------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(
        16, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        32, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    # Output score for the "real" class
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ---------------- TRAIN MODEL ----------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save the final model with the restored best weights.
model.save(MODEL_PATH)

print(f"\nSaved model: {MODEL_PATH}")


# ---------------- TRAINING GRAPHS ----------------

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(BASE_DIR / "training_history.png")
plt.show()


# ---------------- TEST EVALUATION ----------------

best_model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = best_model.evaluate(
    test_ds, verbose=1
)

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

# Output is probability of class "real"
real_scores = best_model.predict(
    test_ds, verbose=1
).reshape(-1)

y_pred = (real_scores >= 0.5).astype(int)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nClassification report:")
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["fake", "real"],
    zero_division=0,
)

print(report)

with open(BASE_DIR / "evaluation_report.txt", "w") as file:
    file.write(report)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["fake", "real"],
)

display.plot(values_format="d")
plt.title("Testing Confusion Matrix")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png")
plt.show()
