import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

IMG_SIZE = 224

def preprocess(example):
    image = example["image"]
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    return {
        "image": image,
        "label": example["label"]
    }
