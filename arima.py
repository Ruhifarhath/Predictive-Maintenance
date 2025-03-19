import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("synthetic_maintenance_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.set_index("Timestamp", inplace=True)

# Selecting relevant feature for time series forecasting
failure_series = df["Failure"].astype(float)  # Ensuring numeric type

# Split data into training and testing sets
train_size = int(len(failure_series) * 0.8)
train, test = failure_series[:train_size], failure_series[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5,1,0))  # ARIMA(p,d,q) parameters
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))

df.loc[test.index, "Predicted_Failure_Probability"] = predictions

# Calculate Model Metrics
mae = mean_absolute_error(test, predictions)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)

# Streamlit Interface
st.title("Predictive Maintenance Dashboard - ARIMA Forecasting")

# Show dataset
st.subheader("Data Overview")
st.write(df.head())

# Plot actual vs predicted failure probabilities
st.subheader("Failure Probability Forecasting (ARIMA)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index, train, label="Training Data", color='blue')
ax.plot(test.index, test, label="Actual Failure Probability", color='green')
ax.plot(test.index, predictions, label="Predicted Failure Probability", color='red')
ax.legend()
st.pyplot(fig)

# Model Performance Metrics
st.subheader("Model Performance Metrics")
st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

st.write("Dashboard provides ARIMA-based failure probability predictions!")
