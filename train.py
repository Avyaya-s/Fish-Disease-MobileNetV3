import tensorflow as tf
from datasets import load_dataset
from preprocess import preprocess
from model import build_model

BATCH_SIZE = 32
EPOCHS = 3

print("Loading dataset...")
dataset = load_dataset("Saon110/bd-fish-disease-dataset")

labels = set(dataset["train"]["label"])
num_classes = len(labels)
print("Number of classes:", num_classes)

dataset = dataset.map(preprocess)

train_ds = dataset["train"].to_tf_dataset(
    columns=["image"],
    label_cols=["label"],
    shuffle=True,
    batch_size=BATCH_SIZE
)

val_ds = dataset["valid"].to_tf_dataset(
    columns=["image"],
    label_cols=["label"],
    shuffle=False,
    batch_size=BATCH_SIZE
)

test_ds = dataset["test"].to_tf_dataset(
    columns=["image"],
    label_cols=["label"],
    shuffle=False,
    batch_size=BATCH_SIZE
)

print("Building model...")
model = build_model(num_classes=num_classes)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training started...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

print("Evaluating...")
model.evaluate(test_ds)

model.save("fish_disease_model.h5")
model.save("fish_disease_model.keras")
print("Model saved successfully")

