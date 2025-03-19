import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load trained model
model = tf.keras.models.load_model("predictive_maintenance_model.h5")

# Load dataset
df = pd.read_csv("synthetic_maintenance_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Drop non-relevant columns
df_cleaned = df.drop(columns=["Machine_ID", "Last_Maintenance", "Failure_Type", "Timestamp"])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cleaned.drop(columns=["Failure"]))

# Reshape for LSTM
X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Predict failure probabilities
failure_probabilities = model.predict(X_lstm)
df["Failure_Probability"] = failure_probabilities

# Streamlit Interface
st.title("Predictive Maintenance Dashboard")

# Show dataset
st.subheader("Data Overview")
st.write(df.head())

# Plot failure probability over time
st.subheader("Failure Probability Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=df["Timestamp"], y=df["Failure_Probability"], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Machine-wise failure distribution
st.subheader("Machine-wise Failure Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["Failure_Probability"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# Show high-risk failure cases
st.subheader("High-Risk Maintenance Alerts")
high_risk_cases = df[df["Failure_Probability"] > 0.5][["Timestamp", "Failure_Probability"]]
st.write(high_risk_cases)

st.write("Dashboard provides real-time failure prediction insights!")