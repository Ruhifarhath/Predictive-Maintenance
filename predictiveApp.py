#!/usr/bin/env python3
"""
Predictive Maintenance for Industrial Equipment using Deep Learning & Time Series Forecasting
--------------------------------------------------------------------------------------------
This script demonstrates a predictive maintenance system that leverages deep learning.
It includes:
  - An LSTM model for predicting equipment failures based on sensor time series data.
  - An LSTM autoencoder for anomaly detection in the sensor data.
  - A Streamlit web interface for user input (via dropdowns) to predict failure.
  
If a sensor data CSV file ('sensor_data.csv') exists with columns such as:
  - timestamp, temperature, vibration, pressure, failure_flag
the script will use it; otherwise, synthetic data is generated for demonstration.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Data Loading & Preprocessing
# ----------------------------
def load_data(file_path='sensor_data.csv'):
    """
    Loads sensor data from a CSV file. If the file does not exist,
    synthetic data is generated for demonstration.
    """
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
    else:
        print("Sensor data file not found. Generating synthetic data...")
        date_rng = pd.date_range(start='2020-01-01', end='2021-01-01', freq='H')
        np.random.seed(42)
        data = {
            'timestamp': date_rng,
            'temperature': np.random.normal(loc=75, scale=5, size=len(date_rng)),
            'vibration': np.random.normal(loc=0.5, scale=0.1, size=len(date_rng)),
            'pressure': np.random.normal(loc=30, scale=3, size=len(date_rng)),
        }
        df = pd.DataFrame(data)
        # Introduce failure flags based on sensor thresholds (for demonstration)
        df['failure_flag'] = ((df['temperature'] > 80) &
                              (df['vibration'] > 0.6) &
                              (df['pressure'] > 32)).astype(int)
    return df

def preprocess_data(df, window_size=24):
    """
    Preprocesses the data:
      - Sorts by timestamp.
      - Scales sensor features using MinMaxScaler.
      - Creates sliding windows (time series sequences) for training.
    
    Returns:
      X: Array of shape (num_samples, window_size, num_features)
      y: Array of corresponding failure flags (binary labels)
      scaler: Fitted scaler (in case inverse transformation is needed)
    """
    df = df.sort_values('timestamp')
    sensor_cols = ['temperature', 'vibration', 'pressure']
    
    scaler = MinMaxScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    
    X, y = [], []
    for i in range(len(df) - window_size):
        # Create a window of sensor data; predict the failure flag at the next time step.
        X.append(df[sensor_cols].iloc[i:i+window_size].values)
        y.append(df['failure_flag'].iloc[i+window_size])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# ----------------------------
# Model Building Functions
# ----------------------------
def build_lstm_model(input_shape):
    """
    Builds an LSTM-based binary classifier for failure prediction.
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_autoencoder_model(timesteps, features):
    """
    Builds an LSTM autoencoder model for anomaly detection.
    The autoencoder is trained to reconstruct normal sensor data.
    """
    input_layer = Input(shape=(timesteps, features))
    # Encoder
    encoded = LSTM(64, activation='relu', return_sequences=True)(input_layer)
    encoded = LSTM(32, activation='relu')(encoded)
    # Bottleneck: Repeat the encoded vector for each time step
    bottleneck = RepeatVector(timesteps)(encoded)
    # Decoder
    decoded = LSTM(32, activation='relu', return_sequences=True)(bottleneck)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    output_layer = TimeDistributed(Dense(features))(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def plot_training_history(history, title_prefix=""):
    """
    Plots training loss and (if available) accuracy from the training history.
    """
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    if 'accuracy' in history.history:
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.title(f"{title_prefix} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
    plt.show()

# ----------------------------
# Training Function (with caching for Streamlit)
# ----------------------------
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_and_train():
    """
    Loads data, preprocesses it, builds the LSTM model, trains it,
    and returns the trained model, scaler, window_size, and test accuracy.
    """
    df = load_data()
    window_size = 24  # Use past 24 time steps (hours) for prediction
    X, y, scaler = preprocess_data(df, window_size)
    
    # Split the data into training and test sets
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    # Train the model (suppress verbose output in Streamlit)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model Accuracy: {accuracy:.2f}")  # Also prints accuracy in console
    return model, scaler, window_size, accuracy

# ----------------------------
# Streamlit Web Interface
# ----------------------------
def run_streamlit():
    st.title("Predictive Maintenance: Failure Prediction")
    st.write("Enter the latest sensor readings below to predict equipment failure.")
    
    # Load (or train) model and preprocessing parameters
    model, scaler, window_size, model_accuracy = load_and_train()
    
    # Display overall model accuracy
    st.write(f"**Model Accuracy:** {model_accuracy:.2f}")
    
    st.subheader("Sensor Inputs")
    # Define dropdown options for sensor values (values chosen for demonstration)
    temp_options = [65, 70, 75, 80, 85]
    vib_options = [0.4, 0.45, 0.5, 0.55, 0.6]
    pres_options = [27, 30, 33]
    
    selected_temp = st.selectbox("Temperature", temp_options)
    selected_vib = st.selectbox("Vibration", vib_options)
    selected_pres = st.selectbox("Pressure", pres_options)
    
    if st.button("Predict Failure"):
        # Create a DataFrame for the single input reading
        user_data = pd.DataFrame({
            'temperature': [selected_temp],
            'vibration': [selected_vib],
            'pressure': [selected_pres]
        })
        # Scale the input data using the same scaler as training
        user_data_scaled = scaler.transform(user_data)
        # Since the LSTM model expects a sequence, replicate the input to form a sequence of length window_size
        sequence = np.repeat(user_data_scaled, window_size, axis=0)  # shape (window_size, features)
        sequence = sequence.reshape(1, window_size, -1)
        
        prediction = model.predict(sequence)
        pred_class = int(prediction[0] > 0.5)
        result = "⚠️ Failure predicted" if pred_class == 1 else "✅ No failure predicted"
        st.write(f"**Prediction:** {result}")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == '__main__':
    # When running via Streamlit, this block is executed.
    # The model training will be cached to avoid re-training on every interaction.
    run_streamlit()
