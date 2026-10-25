# Pneumonia Detection from Chest X-Ray Images

An academic machine learning project developed as part of my Master's studies in Applied Mathematics.

The project explores the use of a convolutional neural network (CNN) to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**. The main objective was to apply deep learning techniques to an image classification problem and investigate how model architecture, training parameters, and class imbalance affect predictive performance.

## Project Overview

The project covers the main stages of a machine learning workflow:

- Preparing and loading chest X-ray images for training.
- Designing and training a convolutional neural network using PyTorch.
- Exploring different hyperparameter configurations to improve model performance.
- Accounting for class imbalance during training.
- Evaluating the trained model using classification metrics and visualisations.

The work combines practical implementation with an interest in understanding how model performance should be measured and interpreted, rather than relying on accuracy alone.

## Dataset

The project uses the publicly available [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), which contains chest X-ray images labelled as either normal or indicative of pneumonia.

The dataset is organised into training, validation, and test sets. These serve different purposes: training the model, comparing model configurations, and assessing performance on held-out data.

## Methodology

### Convolutional Neural Network

A CNN was implemented in PyTorch to learn visual patterns from the X-ray images and assign each image to one of the two classes. The model includes configurable architectural and regularisation parameters, allowing different configurations to be explored.

### Hyperparameter Tuning

A separate tuning script searches across combinations of training and model parameters, including learning rate, batch size, dropout, activation function, and classifier size.

| Hyperparameter | Range |
| --- | --- |
| Fully-connected units | $\{64,\ 128,\ 256,\ 512\}$ |
| Learning rate | $\{0.001,\ 0.0005\}$ |
| Batch size | $\{16,\ 32\}$ |
| Dropout | $\{0.0,\ 0.3,\ 0.5\}$ |
| Activation Function | $\{\mathrm{relu},\ \mathrm{leakyrelu}\}$ |

Configurations are compared using **balanced accuracy** on the validation set, since the class distributions are uneven.

### Model Evaluation

Model performance is examined using classification metrics, including accuracy, precision, recall, and F1 score, alongside visualisations such as the confusion matrix.

These metrics provide complementary perspectives on the model's behaviour. In particular, precision and recall help illustrate the trade-off between correctly identifying pneumonia cases and incorrectly classifying normal X-rays as pneumonia.

## Repository Structure

| File | Description |
| --- | --- |
| `fetch_dataset.py` | Downloads the dataset and identifies its location. |
| `model.py` | Defines the convolutional neural network architecture. |
| `train.py` | Handles model training and checkpoint saving. |
| `tune.py` | Explores hyperparameter configurations and compares their performance. |
| Evaluation notebook | Examines the trained model's performance and presents evaluation results. |

## Getting Started

### Requirements

- Python
- uv
- PyTorch and Torchvision
- NumPy and the other libraries used by the training and evaluation scripts
- Jupyter Notebook to explore the evaluation results

### Setup

1. Install uv.
2. Clone the repository and enter the project directory:

```bash
git clone https://github.com/jakkemm/pneumonia-detection.git
cd pneumonia-detection
```

3. Install the project dependencies:

```bash
uv sync
```

4. Activate the virtual environment (optional):

```bash
source .venv/bin/activate
```

Alternatively, run Python commands directly through uv without activating the environment:

```bash
uv run python train.py
```

## Results

In-progress...

## Limitations

This project is an educational experiment in image classification, not a clinically validated diagnostic system.

Performance on a single dataset does not necessarily translate to images from different hospitals, equipment, patient populations, or clinical settings. Further validation, more representative data, and careful assessment of potential dataset biases would be necessary before considering any real-world medical application.

## Context

This project was completed as part of a Master's degree in Applied Mathematics. It provided practical experience with deep learning, experimental evaluation, and the application of statistical and computational methods to a real-world classification problem.
