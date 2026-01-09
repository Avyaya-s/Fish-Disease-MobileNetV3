from datasets import load_dataset

dataset = load_dataset("Saon110/bd-fish-disease-dataset")

# Collect unique labels from training set
labels = set(dataset["train"]["label"])

# Sort for consistent order
labels = sorted(labels)

print("Class names (unique labels):")
for i, label in enumerate(labels):
    print(f"{i}: {label}")
