 Fashion MNIST Classification

## Objective

Build a machine learning model to classify Fashion-MNIST images into one of 10 clothing categories.

## Dataset

Fashion-MNIST Dataset

Classes:
- T-shirt/Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

## Model Architecture

A Feed Forward Artificial Neural Network (ANN) was implemented using PyTorch.

Architecture:
- Flatten Layer
- Fully Connected Layer
- ReLU Activation
- Fully Connected Layer
- Output Layer (10 classes)

## Training Configuration

- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Epochs: 10
- Batch Size: 64
- Framework: PyTorch

## Results

The model was trained on the Fashion-MNIST dataset and achieved good classification performance on the test set.

Training and validation loss/accuracy plots are available in the notebook.

## Repository Contents

- Fashion_MNIST.ipynb
- fashion_mnist_model.pkl
- submission.csv
- Training plots
