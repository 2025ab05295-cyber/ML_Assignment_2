# Breast Cancer Classification ML Application

## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a tumor is malignant or benign using medical measurements. The models are evaluated using multiple performance metrics and deployed as an interactive Streamlit web application.

---

## Dataset Description

The dataset used is the **Breast Cancer Wisconsin dataset**.

- Total samples: 569
- Total features: 30 numerical medical measurements
- Target variable:
  - 0 → Malignant
  - 1 → Benign

The dataset is widely used as a benchmark classification problem and satisfies the assignment requirement of minimum feature and instance size.

A sample test dataset (`breast_cancer.csv`) is included in this repository for evaluation and testing of the Streamlit application.

---

## Models Implemented

The following classification algorithms were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

---

## Evaluation Metrics

Each model is evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|----------|-----|----------|--------|----|-----|
| Logistic Regression | 0.9561 | 0.9977 | 0.9859 | 0.9655 | 0.9756 | 0.9068 |
| Decision Tree | 0.9298 | 0.9299 | 0.9296 | 0.9429 | 0.9362 | 0.8526 |
| kNN | 0.9561 | 0.9959 | 1.0000 | 0.9659 | 0.9827 | 0.9086 |
| Naive Bayes | 0.9737 | 0.9984 | 1.0000 | 0.9731 | 0.9863 | 0.9447 |
| Random Forest | 0.9649 | 0.9949 | 0.9859 | 0.9722 | 0.9790 | 0.9253 |
| XGBoost | 0.9561 | 0.9908 | 0.9718 | 0.9860 | 0.9789 | 0.9064 |

---

## Observations

- Logistic Regression provides a strong and stable baseline model.
- Decision Tree achieves good accuracy but may overfit due to its high variance.
- kNN performs well but depends on distance metrics and dataset scaling.
- Naive Bayes achieves the highest overall performance and efficiency.
- Random Forest improves stability using ensemble voting.
- XGBoost performs competitively with boosted ensemble learning.

---

## Streamlit Application Features

The deployed Streamlit application includes:

- CSV dataset upload
- Model selection dropdown
- Real-time prediction
- Classification report display
- Confusion matrix visualization

---

## Deployment

The application is deployed using **Streamlit Community Cloud** and runs as a live interactive ML dashboard.

---

## Repository Structure
heart-disease-ml/

- ML_Assignment_2/
  - app.py
  - train_models.py
  - explore_dataset.py
  - breast_cancer.csv
  - model_results.csv
  - requirements.txt
  - README.md
  - model/
    - model_Logistic_Regression.pkl
    - model_Decision_Tree.pkl
    - model_kNN.pkl
    - model_Naive_Bayes.pkl
    - model_Random_Forest.pkl
    - model_XGBoost.pkl

## Conclusion

This project demonstrates a complete machine learning workflow including model training, evaluation, comparison, deployment, and interactive visualization using Streamlit. It simulates a real-world ML pipeline from data processing to cloud deployment.
