import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Sidebar
st.sidebar.title("🚀 Navigation")
option = st.sidebar.radio("Go to", ["Dashboard", "Predict", "Upload Data"])

# Title
st.title("💳 Credit Card Fraud Detection System")

# ---------------- DASHBOARD ----------------
if option == "Dashboard":
    st.subheader("📊 Dataset Overview")

    df = pd.read_csv("creditcard.csv")

    st.write(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Class Distribution")
        sns.countplot(x="Class", data=df)
        st.pyplot()

    with col2:
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(6,4))
        sns.heatmap(df.corr(), cmap="coolwarm")
        st.pyplot()

# ---------------- PREDICTION ----------------
elif option == "Predict":
    st.subheader("🔍 Fraud Prediction")

    input_data = []

    for i in range(30):
        val = st.number_input(f"Feature V{i+1}", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction")
        else:
            st.success("✅ Legit Transaction")

# ---------------- UPLOAD ----------------
elif option == "Upload Data":
    st.subheader("📂 Upload CSV")

    file = st.file_uploader("Upload file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        if st.button("Run Predictions"):
            X = df.drop("Class", axis=1, errors='ignore')
            X_scaled = scaler.transform(X)

            preds = model.predict(X_scaled)
            df["Prediction"] = preds

            st.write(df.head())