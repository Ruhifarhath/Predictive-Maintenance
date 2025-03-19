import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("synthetic_maintenance_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.set_index("Timestamp", inplace=True)

# Encode categorical columns
categorical_cols = ["Last_Maintenance", "Failure_Type"]
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure strings before encoding
        label_encoders[col] = le

# Selecting relevant feature for time series forecasting
failure_series = df["Failure"].astype(float)  # Ensuring numeric type

# Creating lag features for time series forecasting
lags = 10
for lag in range(1, lags + 1):
    df[f"lag_{lag}"] = df["Failure"].shift(lag)

df.dropna(inplace=True)

# Reset index to avoid duplicate timestamp issues
df = df.reset_index()

# Splitting data into training and testing sets
X = df.drop(columns=["Failure", "Timestamp"])  # ✅ Drop Timestamp column
y = df["Failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fit XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Ensure predictions length matches y_test
predictions = predictions[:len(y_test)]  # Trim if necessary
y_test = y_test.iloc[:len(predictions)]  # Trim target if necessary

# Assign predictions safely
X_test = X_test.iloc[:len(predictions)]  # Trim X_test to avoid mismatch
df.loc[X_test.index, "Predicted_Failure_Probability"] = predictions

# Calculate Model Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Streamlit Interface
st.title("Predictive Maintenance Dashboard - XGBoost Forecasting")

# Show dataset
st.subheader("Data Overview")
st.write(df.head())

# Plot actual vs predicted failure probabilities
st.subheader("Failure Probability Forecasting (XGBoost)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_train.index, y_train, label="Training Data", color='blue')
ax.plot(y_test.index, y_test, label="Actual Failure Probability", color='green')
ax.plot(y_test.index, predictions, label="Predicted Failure Probability", color='red')
ax.legend()
st.pyplot(fig)

# Model Performance Metrics
st.subheader("Model Performance Metrics")
st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

st.write("Dashboard provides XGBoost-based failure probability predictions!")
