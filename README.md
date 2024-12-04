
# Parkinson's Disease Diagnosis and Prediction

This project provides a complete pipeline for diagnosing and predicting Parkinson's Disease using machine learning models (XGBoost and SVM). It includes preprocessing, model training, evaluation, and deployment using Docker and Kubernetes.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Deployment](#deployment)
- [How to Use](#how-to-use)
- [Results](#results)
- [References](#references)

---

## Introduction

Parkinson's Disease is a neurodegenerative disorder characterized by motor and non-motor symptoms. This project uses vocal parameter data to classify individuals as Parkinson's patients or healthy controls.

---

## Dataset

- The dataset contains 195 samples with 24 columns.
- **Target Variable**: `status` (1 = Parkinson's, 0 = Healthy).
- Features include vocal parameters such as jitter, shimmer, and harmonic-to-noise ratio.

### Sample Dataset

| MDVP:Fo(Hz) | MDVP:Fhi(Hz) | MDVP:Flo(Hz) | MDVP:Jitter(%) | MDVP:Shimmer | status |
|-------------|--------------|--------------|----------------|--------------|--------|
| 119.992     | 157.302      | 74.997       | 0.00784        | 0.04374      | 1      |

---

## Preprocessing

- Dropped irrelevant columns (`name`).
- Applied stratified train-test split.
- Handled class imbalance using SMOTE.

---

## Models

1. **XGBoost**:
   - Optimized using `eval_metric='logloss'`.
   - Achieved ~20% improvement in accuracy over baseline models.

2. **SVM**:
   - Tuned for optimal hyperparameters.
   - Ensured robust classification with minimal overfitting.

---

## Deployment

### Docker
The project includes a `Dockerfile` for containerizing the application.

### Kubernetes
Deployment YAML files for managing the application using Kubernetes.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/parkinsons-diagnosis.git
   cd parkinsons-diagnosis
   ```

2. Build the Docker image:
   ```bash
   docker build -t parkinsons-diagnosis .
   ```

3. Run the container:
   ```bash
   docker run -p 5000:5000 parkinsons-diagnosis
   ```

4. Deploy to Kubernetes:
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   ```

---

## Results

- **XGBoost Accuracy**: ~85%
- **SVM Accuracy**: ~82%

---

## References

1. [XGBoost Documentation](https://xgboost.readthedocs.io/)
2. [SVM Basics](https://scikit-learn.org/stable/modules/svm.html)
3. [Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)

---
