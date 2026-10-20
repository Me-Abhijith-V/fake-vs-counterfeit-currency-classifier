
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "currency_cnn.keras"
IMAGE_SIZE = (128, 128)


# ---------------- SELECT IMAGE POPUP ----------------

def select_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    image_path = filedialog.askopenfilename(
        parent=root,
        title="Select Currency Note Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not image_path:
        print("No image selected. Exiting.")
        return None

    return image_path


# ---------------- PREPROCESS IMAGE ----------------

def preprocess_image(image_path):
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path.resolve()}"
        )

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        # Keep original image for displaying
        original_image = image.copy()

        # Resize for model input
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            image, dtype=np.float32
        )

    # Shape: (1, 128, 128, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array, original_image


# ---------------- PREDICT IMAGE ----------------

def predict_image(model, image_path):
    image_array, original_image = preprocess_image(
        image_path
    )

    # Predict using the trained model
    # Assumption: sigmoid output represents REAL probability
    real_probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    # Keep probability within a valid range
    real_probability = np.clip(
        real_probability, 0.0, 1.0
    )

    fake_probability = 1.0 - real_probability

    # Determine predicted class
    if real_probability >= 0.5:
        label = "REAL"
        confidence = real_probability * 100
    else:
        label = "FAKE"
        confidence = fake_probability * 100

    # Print results in terminal
    print("\n========== PREDICTION RESULTS ==========")
    print(f"Image: {image_path}")
    print(f"Predicted class: {label}")
    print(f"Confidence score: {confidence:.2f}%")
    print(f"Real-class score: {real_probability:.2%}")
    print(f"Fake-class score: {fake_probability:.2%}")
    print("========================================")

    # Display prediction window
    plt.figure(figsize=(8, 7))
    plt.imshow(original_image)
    plt.axis("off")

    plt.title(
        f"Prediction: {label}\n"
        f"Score: {confidence:.2f}%",
        fontsize=17,
        pad=20
    )

    plt.tight_layout()
    plt.show()

    print(
        "\nNote: This is an experimental model prediction, "
        "not proof of currency authenticity."
    )


# ---------------- MAIN PROGRAM ----------------

def main():
    # Verify that the trained model exists
    if not MODEL_PATH.is_file():
        print(f"ERROR: Model not found at:\n{MODEL_PATH}")
        print("Place currency_cnn.keras beside predict.py.")
        return

    # Select image using popup
    image_path = select_image()

    if image_path is None:
        return

    # Load existing trained model, no training
    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")
    print("Predicting selected image...")

    predict_image(model, image_path)


if __name__ == "__main__":
    main()
