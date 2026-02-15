import numpy as np
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

st.set_page_config(page_title="ML Classification App", layout="centered")

st.title("Breast Cancer Classification App")
st.subheader("Sample Dataset")

sample_url = "https://raw.githubusercontent.com/2025ab05295-cyber/ML_Assignment_2/main/breast_cancer.csv"

st.markdown(
    f"[⬇ Download sample dataset]({sample_url})",
    unsafe_allow_html=True
)
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

st.divider()

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        model_path = os.path.join("model", model_files[model_option])
        model = joblib.load(model_path)

        preds = model.predict(X)

        st.divider()
        st.subheader("Model Evaluation Results")

        acc = accuracy_score(y, preds)
        st.metric("Accuracy", round(acc, 4))

        report = classification_report(y, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.subheader("Classification Report (Table)")
        st.dataframe(report_df)

        # Confusion Matrix numeric
        cm = confusion_matrix(y, preds)

        st.subheader("Confusion Matrix (Values)")
        st.write(cm)

        # Confusion Matrix heatmap
        st.subheader("Confusion Matrix (Diagram)")

        fig, ax = plt.subplots()
        cax = ax.matshow(cm)
        plt.title("Confusion Matrix")
        fig.colorbar(cax)

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center')

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

    except Exception as e:
        st.error("Error processing dataset. Please upload correct CSV format.")
        st.exception(e)
