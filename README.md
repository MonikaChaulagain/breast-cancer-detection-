# Breast Cancer Detection Using Neural Networks

## Project Overview

This project implements a binary classification model using a neural network to predict whether a breast tumor is malignant or benign. The model is trained and tested on the well-known Breast Cancer Wisconsin (Diagnostic) Dataset provided by scikit-learn (`load_breast_cancer`). The goal is to assist in early diagnosis by accurately classifying tumors based on cell nuclei features extracted from biopsy images.

## Dataset

* Source: `sklearn.datasets.load_breast_cancer`
* Samples: 569
* Features: 30 numeric features related to cell nuclei (e.g., radius, texture, perimeter, area)
* Labels: Binary (0 = benign, 1 = malignant)

## Methodology

* Data preprocessing includes feature scaling using StandardScaler.
* The neural network is trained using PyTorch with a binary cross-entropy loss function.
* The model outputs are probabilities, thresholded to generate binary predictions.
* Evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrix.
* Threshold tuning was performed to optimize sensitivity (recall) and specificity (precision).

## Results

* Achieved accuracy: \~97% on the test set.
* High recall on malignant cases, minimizing false negatives which is  critical for clinical applications.
* Detailed classification report and confusion matrix included.

## Usage

* Clone the repository.
* Install dependencies: `pip install -r requirements.txt` (include packages like torch, scikit-learn, numpy, matplotlib).
* Run the training script: `python train.py`
* Evaluate the model: `python evaluate.py`

## Future Work

* Implement probability calibration to improve model confidence.
* Explore ensemble methods to increase robustness.
* Validate on larger and more diverse datasets.
* Integrate explainability methods (e.g., SHAP, LIME) for clinical interpretability.

## References

* [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
* PyTorch official documentation

