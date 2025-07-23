# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_model.pkl")
# model = joblib.load(r"C:\Users\mansh\OneDrive\Desktop\Cohort 1.0\week1\day2\project\titanic_model.pkl")


st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival.")

# Input widgets
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked From", ["C", "Q", "S"])


# Feature engineering
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create input DataFrame
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "Fare": [fare],
    "Sex_male": [sex_male],
    "Embarked_Q": [embarked_Q],
    "Embarked_S": [embarked_S]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"
    st.subheader(f"Prediction: {result}")

