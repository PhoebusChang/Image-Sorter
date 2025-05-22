# Image Sorter

A simple project for sorting and classifying images using basic neural networks in PyTorch and PyTorch Lightning.

## Features

- **Download CIFAR-10 Dataset:**  
  Use `download.py` to automatically download the CIFAR-10 dataset and organize images into folders by class.

- **Train a Neural Network:**  
  `main.py` loads the images, splits them into training and test sets, and trains a simple feedforward neural network to classify the images.

- **Predict New Images:**  
  After training, you can use the model to predict the class of new images (e.g., `test.png`).

## Usage

### 1. Download and Prepare Images

Run the following command to download and organize the CIFAR-10 dataset:

```sh
python download.py
```

This will create a `cifar10_images` folder with subfolders for each class.

### 2. Train the Model

Run the main script:

```sh
python main.py
```

- When prompted, type `y` to train the model.
- The model will be trained and saved as `model.pth`.

### 3. Predict Image Classes

- Place an image named `test.png` in the project directory.
- Run `main.py` again and type `n` when asked to train.
- The script will predict the class of `test.png` and display the result.

## Requirements

- Python 3.7+
- torch
- torchvision
- pytorch-lightning
- pillow
- matplotlib

Install dependencies with:

```sh
pip install torch torchvision pytorch-lightning pillow matplotlib
```

## Folder Structure

```
cifar10_images/
    airplane/
    automobile/
    bird/
    cat/
    deer/
    dog/
    frog/
    horse/
    ship/
    truck/
```

## Notes

- The model expects images of size 32x32 pixels and 3 color channels (RGB).
- You can modify the scripts to use other datasets or architectures as needed.
