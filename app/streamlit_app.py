"""
Streamlit Application for Customer Churn Prediction
Provides an interactive interface for predicting customer churn.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .churn-yes {
        background-color: #ff6b6b;
        color: white;
    }
    .churn-no {
        background-color: #51cf66;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        model = tf.keras.models.load_model('keras_model.keras')
        return preprocessor, model
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.error("Please run train_model.py first to train and save the models.")
        st.stop()


def make_prediction(input_data_df, preprocessor, model):
    input_transformed = preprocessor.transform(input_data_df)
    raw_prediction = model.predict(input_transformed, verbose=0)
    probability = raw_prediction[0][0]
    prediction = 1 if probability > 0.5 else 0
    return prediction, probability


def main():
    st.title("🔮 Customer Churn Prediction System")
    st.markdown("---")
    st.write("Predict whether a customer will churn (leave) based on their profile information.")
    preprocessor, model = load_models()
    tab1, tab2, tab3 = st.tabs(["📊 Single Prediction", "📈 Batch Prediction", "ℹ️ Information"])

    with tab1:
        st.header("Single Customer Prediction")
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
            balance = st.number_input("Account Balance", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
        with col2:
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
            is_active = st.selectbox("Is Active Member", options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
            salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=75000.0, step=1000.0)
            geography = st.selectbox("Geography", options=["France","Germany","Spain"])
        gender = st.selectbox("Gender", options=["Female","Male"])
        has_credit_card = st.selectbox("Has Credit Card", options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
        input_data = pd.DataFrame({'CreditScore':[credit_score],'Age':[age],'Tenure':[tenure],'Balance':[balance],'NumOfProducts':[num_products],'HasCrCard':[has_credit_card],'IsActiveMember':[is_active],'EstimatedSalary':[salary],'Gender':[gender],'Geography':[geography]})
        if st.button("🔮 Predict Churn", key="single_predict"):
            try:
                prediction, probability = make_prediction(input_data, preprocessor, model)
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 1:
                        st.markdown("""<div class='prediction-box churn-yes'>⚠️ HIGH RISK<br>Customer likely to churn</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class='prediction-box churn-no'>✅ LOW RISK<br>Customer likely to stay</div>""", unsafe_allow_html=True)
                with col2:
                    st.metric("Churn Probability", f"{probability:.2%}")
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

    with tab2:
        st.header("Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} records.")
                if st.button("🔮 Predict All", key="batch_predict"):
                    with st.spinner("Making predictions..."):
                        df_transformed = preprocessor.transform(df)
                        raw_predictions = model.predict(df_transformed, verbose=0)
                        probabilities = raw_predictions.flatten()
                        predictions = (probabilities > 0.5).astype(int)
                        results_df = df.copy()
                        results_df['Churn_Prediction'] = predictions
                        results_df['Churn_Probability'] = probabilities
                        results_df['Risk_Level'] = results_df['Churn_Prediction'].apply(lambda x: '🔴 High Risk' if x==1 else '🟢 Low Risk')
                        st.dataframe(results_df)
                        csv = results_df.to_csv(index=False)
                        st.download_button(label="📥 Download Results as CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    with tab3:
        st.header("About This Application")
        st.write("Model: MLP (Keras), Preprocessing: OneHot + StandardScaler")

if __name__ == "__main__":
    main()
