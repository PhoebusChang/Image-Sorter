import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Directory to save images
output_dir = "cifar10_images"
os.makedirs(output_dir, exist_ok=True)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
images = np.concatenate([x_train, x_test])
labels = np.concatenate([y_train, y_test]).flatten()

# CIFAR-10 label names
label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Save images and labels
for idx, (img, label) in enumerate(zip(images, labels)):
    label_name = label_names[label]
    label_dir = os.path.join(output_dir, label_name)
    os.makedirs(label_dir, exist_ok=True)
    img_path = os.path.join(label_dir, f"{idx}.png")
    Image.fromarray(img).save(img_path)

print(f"Downloaded and saved {len(images)} images in '{output_dir}' folder.")