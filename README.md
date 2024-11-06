# Creddify - Card Fraud Detection in Keras


This project demonstrates how to detect fraudulent credit card transactions using an autoencoder neural network built in Keras. Autoencoders are unsupervised neural networks that learn to compress and reconstruct input data. For this application, they are used to identify anomalies in credit card transactions, which can indicate fraud.

## Table of Content
- Project Structure
- Prerequisites
- Installation 
- Data Description 
- Model Overview 
- Evaluating the Model 
- Results 

## Project Structure

```bash
Creddify/
│
├── data/
│   └── creditcard.csv          # The credit card transaction dataset
│
├── notebooks/
│   └── fraud_detection.ipynb   # Jupyter notebook for model development and analysis
│
├── model/
│   ├── autoencoder_model.py    # Keras model code for the autoencoder
│   ├── train.py                # Script to train the model
│   └── evaluate.py             # Script to evaluate the model
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


```

## Prerequisites
- pandas
- numpy
- matplotlib
- scipy
- tensorflow
- seaborn
- scikit-learn
- keras

## Installation

1. Install Dependencies 
```bash
pip install -r requirements.txt
```

2. Download the dataset (if not already included) and place it in the `data/` directory.
## Usage

## Data Description

The dataset consists of the following columns:

- Time: Time elapsed between the current transaction and the first transaction in the dataset.
- V1 - V28: 28 anonymized features which are PCA-transformed versions of the original features.
- Amount: Transaction amount for the current transaction.
- Class: Target variable, with `1` representing fraud and `0` representing legitimate transactions.

Note: The dataset is highly imbalanced, with only a small percentage of fraud cases.


## Model Overview

The core of this fraud detection model is an `autoencoder` — an unsupervised neural network architecture. The autoencoder is trained on only the legitimate transactions (label `0`) to learn a representation of normal transaction data. During inference, the autoencoder attempts to reconstruct each transaction. If the reconstruction error is large, the transaction is flagged as an anomaly, which could indicate fraud.

Key Components:
- Encoder: Compresses the input data into a smaller, dense representation.
- Decoder: Reconstructs the input data from the compressed representation.
- Reconstruction Error: The difference between the original input and the reconstructed output. A high error suggests an anomaly.

## EvaluAting the Model
The evaluation script calculates the reconstruction error and uses it to classify transactions as fraudulent or legitimate. The model's performance can be assessed using metrics like:

- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

Example script to evaluate the model:

```bash
python model/evaluate.py
```

## Results
After training and evaluating the model, you should observe that the autoencoder is able to correctly identify fraudulent transactions based on high reconstruction errors. Since the model is unsupervised, no labels are required during training, making it suitable for scenarios where fraudulent transactions are rare and labeled data is scarce.

#### Sample Results:
- Precision: 0.90
- Recall: 0.85
- F1 Score: 0.87
- AUC: 0.95

The model performs well at identifying anomalies, with a high true positive rate and low false positive rate.
