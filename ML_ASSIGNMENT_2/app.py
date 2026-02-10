import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

st.title("Breast Cancer Classification App")

uploaded_file = st.file_uploader("Upload CSV file")

model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_files = {
    "Logistic Regression": "model_Logistic_Regression.pkl",
    "Decision Tree": "model_Decision_Tree.pkl",
    "kNN": "model_kNN.pkl",
    "Naive Bayes": "model_Naive_Bayes.pkl",
    "Random Forest": "model_Random_Forest.pkl",
    "XGBoost": "model_XGBoost.pkl"
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model_path = os.path.join("models", model_files[model_option])

    st.write("Loading model from:", model_path)  # debug line

    model = joblib.load(model_path)

    preds = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, preds))
