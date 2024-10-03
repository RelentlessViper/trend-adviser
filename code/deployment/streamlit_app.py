import streamlit as st
import requests
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

# Apply responsive CSS for smaller screens
st.markdown(
    """
    <style>
    .dataframe {
        font-size: 16px !important;
        width: 100% !important;
    }
    @media (max-width: 768px) {
        .dataframe {
            font-size: 12px !important;
        }
    }
    @media (max-width: 480px) {
        .dataframe {
            font-size: 10px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define FastAPI URL
FASTAPI_URL = "http://fastapi:80/predict/"

st.title("S&P 500 Stock Price Prediction")

st.subheader("Enter the Date Range for Prediction:")
start_date = st.text_input("Start Date (YYYY.MM.DD)", "2023.09.01")
end_date = st.text_input("End Date (YYYY.MM.DD)", "2024.09.01")

if st.button("Predict"):
    if start_date >= end_date:
        st.error("Start date must be less than end date.")
    else:
        try:
            payload = {"start_date": start_date, "end_date": end_date}
            response = requests.post(FASTAPI_URL, json=payload)

            if response.status_code == 200:
                data = response.json()

                df = pd.DataFrame(data)
                df.index = pd.to_datetime(df.loc[:, 'Date'].apply(func=lambda x: x.split('T')[0]))
                df = df.drop(labels=['Date'], axis=1)

                # Show table and graph of actual and predicted prices
                st.subheader("Predictions Table:")
                st.dataframe(df.style.set_properties(**{'font-size': '20px', 'width': '100%'}))


                st.subheader("Closing Prices of S&P 500: Actual vs Predicted")
                plt.figure(figsize=(15, 8))
                plt.plot(df.index, df['Close'], label="Actual", marker='o')
                plt.plot(df.index, df['Predicted'], label="Predicted", marker='x')
                plt.xlabel("Date")
                plt.ylabel("Closing Price")
                plt.legend()
                st.pyplot(plt)
                
                # Display evaluation metrics
                mse = mean_squared_error(df['Close'], df['Predicted'])
                mae = mean_absolute_error(df['Close'], df['Predicted'])
                r2 = r2_score(df['Close'], df['Predicted'])

                st.subheader("Model Evaluation Metrics")
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
                st.write(f"**R-squared (R2):** {r2:.2f}")
            else:
                st.error(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
