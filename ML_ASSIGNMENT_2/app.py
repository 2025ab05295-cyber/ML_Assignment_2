import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.title("Breast Cancer Classification App")

st.write("Upload test dataset (CSV format)")

uploaded_file = st.file_uploader("Choose CSV file")

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

    model = joblib.load(model_files[model_option])

    preds = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, preds))