import tensorflow as tf
import numpy as np
from PIL import Image
import sys

# ---------------- CONFIG ----------------
MODEL_PATH = "fish_disease_model.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Bacterial Red disease",
    "Bacterial gill disease",
    "Bacterial disease",
    "Fungal disease",
    "Healthy fish",
    "Parasitic disease",
    "Viral disease",
    "White tail disease",
    "Columnaris disease",
    "Aeromonas disease",
    "EUS disease"
]
# ----------------------------------------

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Processing image...")
    image = load_and_preprocess_image(image_path)

    print("Running prediction...")
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100

    print("\n--- Prediction Result ---")
    print(f"Predicted class : {CLASS_NAMES[predicted_class]}")
    print(f"Confidence      : {confidence:.2f}%")
