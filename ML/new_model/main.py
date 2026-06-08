from pyexpat import model

from model import Model_relu_softmax
from layer import Layer
from mnist_loader import normalise_data
from functions import *
from pathlib import Path

data_path = str(Path(__file__).parent.parent / "data")

images, labels = normalise_data(
    10, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
)
print(len(images))


# --- 3. Proces Forward Pass ---
# Przejście przez warstwy
learning_rate = 0.01  # Jak duże kroki robimy?
epochs = 25
batch_size = 128

model = Model_relu_softmax([784, 128, 64, 10])
print("Model created. Starting training...")
model.train(images, labels, batch_size, learning_rate, epochs)

test_images, test_labels = normalise_data(
    10, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
)
print("Testing model on test data...")
model.test(test_images, test_labels)
print("Exporting model...")
model.export("model.npz")
print("Model exported to model.npz")

new_model = Model_relu_softmax.from_file("model.npz")
print("Model loaded from model.npz")
print("Testing loaded model... (accuracy should be the same as before)")
new_model.test(test_images, test_labels)
