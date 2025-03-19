import os
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score

# ---------------------------
# Data Loading
# ---------------------------
def load_data():
    file_path = "ai 2020.csv"  # Ensure this file is in the working directory
    df = pd.read_csv(file_path)
    return df

df = load_data()

# ---------------------------
# Training Functions
# ---------------------------
def train_classification_model(df):
    # Create a new column "Failure Mode" from the failure mode columns.
    def get_failure_mode(row):
        if row['Machine failure'] == 0:
            return 0  # No Failure
        else:
            if row['TWF'] == 1:
                return 1
            elif row['HDF'] == 1:
                return 2
            elif row['PWF'] == 1:
                return 3
            elif row['OSF'] == 1:
                return 4
            elif row['RNF'] == 1:
                return 5
            else:
                return 0

    df['Failure Mode'] = df.apply(get_failure_mode, axis=1)
    feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X = df[feature_columns].values.astype(np.float32)
    y = df['Failure Mode'].values

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_classes = 6

    # Build a simple feedforward neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    model.save("failure_detection_model.h5")
    return model

def train_forecasting_model(df, window_size=10):
    # Prepare data for time series forecasting.
    # Use sensor features to predict the next step's "Machine failure" (0 or 1).
    feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    data = df[feature_columns].values.astype(np.float32)
    target = df['Machine failure'].values.astype(np.float32)

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(target[i+window_size])  # predict the next step
    X = np.array(X)
    y = np.array(y)

    # Build a simple LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(window_size, X.shape[2])),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    model.save("failure_forecast_model.h5")
    return model

# ---------------------------
# Model Loading (Train if Not Exist)
# ---------------------------
@st.cache_resource
def load_classification_model():
    model_path = "failure_detection_model.h5"
    if not os.path.exists(model_path):
        st.write("Training classification model...")
        model = train_classification_model(df)
    else:
        model = tf.keras.models.load_model(model_path)
    return model

classification_model = load_classification_model()

@st.cache_resource
def load_forecast_model():
    model_path = "failure_forecast_model.h5"
    if not os.path.exists(model_path):
        st.write("Training forecasting model...")
        model = train_forecasting_model(df)
    else:
        model = tf.keras.models.load_model(model_path)
    return model

forecast_model = load_forecast_model()

# ---------------------------
# Prediction Functions
# ---------------------------
def predict_failure(features):
    features = np.array(features).reshape(1, -1)
    prediction = classification_model.predict(features)
    failure_classes = ['No Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    return failure_classes[np.argmax(prediction)]

def calculate_accuracy():
    feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X = df[feature_columns].values.astype(np.float32)
    # Ensure "Failure Mode" column exists for accuracy calculation.
    if 'Failure Mode' not in df.columns:
        def get_failure_mode(row):
            if row['Machine failure'] == 0:
                return 0
            else:
                if row['TWF'] == 1:
                    return 1
                elif row['HDF'] == 1:
                    return 2
                elif row['PWF'] == 1:
                    return 3
                elif row['OSF'] == 1:
                    return 4
                elif row['RNF'] == 1:
                    return 5
                else:
                    return 0
        df['Failure Mode'] = df.apply(get_failure_mode, axis=1)
    y_true = df['Failure Mode'].values
    y_pred = [np.argmax(classification_model.predict(x.reshape(1, -1))) for x in X]
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def forecast_failure(time_series_data):
    # Reshape input to (1, window_size, num_features)
    input_data = np.array(time_series_data, dtype=np.float32).reshape(1, time_series_data.shape[0], time_series_data.shape[1])
    prediction = forecast_model.predict(input_data)
    # Forecasted probability of failure at the next time step
    prob_failure = prediction[0, 0]
    return prob_failure

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title("Predictive Maintenance - Deep Learning & Time Series Analysis")
    st.write("This system forecasts failures and maintenance needs to reduce downtime and improve operational efficiency.")
    
    mode = st.radio("Select Prediction Mode", ("Single Instance Prediction", "Time Series Forecasting"))
    
    if mode == "Single Instance Prediction":
        st.subheader("Single Instance Prediction")
        st.write("Select a machine record for failure prediction.")
        st.dataframe(df.head(20))
        
        row_index = st.number_input("Select Row Index", min_value=0, max_value=len(df)-1, value=0)
        selected_row = df.iloc[row_index]
        feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        features = selected_row[feature_columns].values
        features = np.array(features, dtype=np.float32)
        
        if st.button("Predict Failure"):
            result = predict_failure(features)
            st.success(f"Predicted Failure Mode: {result}")
            
        if st.button("Calculate Model Accuracy"):
            accuracy = calculate_accuracy()
            st.write(f"Model Accuracy: {accuracy:.2%}")
    
    else:
        st.subheader("Time Series Forecasting")
        window_size = 10
        max_start = len(df) - window_size - 1
        start_index = st.number_input("Select Start Index for Time Series", min_value=0, max_value=max_start, value=0)
        feature_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        time_series_data = df.iloc[start_index:start_index+window_size][feature_columns].values
        st.write("Time Series Data Preview:")
        st.dataframe(pd.DataFrame(time_series_data, columns=feature_columns))
        
        if st.button("Forecast Failure"):
            prob_failure = forecast_failure(time_series_data)
            st.success(f"Forecasted Failure Probability: {prob_failure*100:.2f}%")
    
if __name__ == "__main__":
    main()
