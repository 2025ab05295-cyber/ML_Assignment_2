Breast Cancer Classification ML App

Problem Statement

The goal is to build and compare multiple classification models to predict whether a tumor is malignant or benign using medical measurements.

Dataset Description

The dataset is the Breast Cancer Wisconsin dataset containing 569 samples and 30 numerical features. The target variable indicates whether the tumor is malignant or benign.

Models Used
	•	Logistic Regression
	•	Decision Tree
	•	k-Nearest Neighbor
	•	Naive Bayes
	•	Random Forest (Ensemble)
	•	XGBoost (Ensemble)

Model Comparison

                 Model  Accuracy       AUC  ...    Recall        F1       MCC
0  Logistic Regression  0.956140  0.997707  ...  0.985915  0.965517  0.906811
1        Decision Tree  0.929825  0.929905  ...  0.929577  0.942857  0.852580
2                  kNN  0.956140  0.995906  ...  1.000000  0.965986  0.908615
3          Naive Bayes  0.973684  0.998362  ...  1.000000  0.979310  0.944733
4        Random Forest  0.964912  0.994923  ...  0.985915  0.972222  0.925285
5              XGBoost  0.956140  0.990829  ...  0.971831  0.965035  0.906379

Observations
	•	Logistic Regression performs consistently and is a strong baseline.
	•	Decision Tree achieves high accuracy but risks overfitting.
	•	kNN performs well but is sensitive to distance metrics.
	•	Naive Bayes is fast and efficient with competitive performance.
	•	Random Forest improves stability using ensemble learning.
	•	XGBoost achieves the best overall performance due to boosting.

Streamlit App Features
	•	CSV dataset upload
	•	Model selection dropdown
	•	Classification report display
	•	Confusion matrix visualization