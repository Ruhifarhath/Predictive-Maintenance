import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("synthetic_maintenance_data.csv")

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Drop non-relevant columns
df_cleaned = df.drop(columns=["Machine_ID", "Last_Maintenance", "Failure_Type", "Timestamp"])

# Define features (X) and target (y)
X = df_cleaned.drop(columns=["Failure"])
y = df_cleaned["Failure"]

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data to fit LSTM (samples, time steps, features)
X_train_lstm = X_train_scaled.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[1])),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (Failure: 0 or 1)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train_lstm, y_train, epochs=20, batch_size=64, validation_data=(X_test_lstm, y_test), verbose=1)

# Save trained model
model.save("predictive_maintenance_model.h5")

print("Model training complete and saved as 'predictive_maintenance_model.h5'")
