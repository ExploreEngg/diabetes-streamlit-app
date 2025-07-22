# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")

# Load trained model
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
model_file = f"trained_models/{model_choice.replace(' ', '_').lower()}_model.joblib"

@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

try:
    model = load_model(model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Upload test data
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Test Data", test_data.head())

    # Predict
    try:
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)[:, 1]
        test_data['Diabetes Prediction'] = predictions
        test_data['Probability'] = probabilities

        st.success("✅ Prediction complete!")
        st.write(test_data)

        # Visual output
        st.subheader("📈 Prediction Distribution")
        st.bar_chart(test_data['Diabetes Prediction'].value_counts())

    except Exception as e:
        st.error(f"Prediction failed: {e}")
