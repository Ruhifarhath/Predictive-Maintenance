import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Parameters
days = 90  # Number of days to simulate
machines = 10  # Number of machines
sampling_rate = 6  # Readings per hour (every 10 minutes)

def generate_synthetic_data(machines, days, sampling_rate):
    data = []
    start_time = datetime.now() - timedelta(days=days)
    num_samples = days * 24 * sampling_rate
    
    for machine_id in range(1, machines + 1):
        time = start_time
        temp = 30 + np.random.normal(0, 2)  # Initial temperature
        vibration = 0.5 + np.random.normal(0, 0.1)  # Initial vibration
        pressure = 100 + np.random.normal(0, 5)  # Initial pressure
        cycles = 0
        last_maintenance = time
        failure_probability = 0.001  # Initial base failure probability

        for _ in range(num_samples):
            # Simulate gradual degradation over time
            temp += np.random.normal(0.02, 0.1)
            vibration += np.random.normal(0.005, 0.02)
            pressure += np.random.normal(0.1, 0.5)
            cycles += 1

            # Increase failure probability over time
            failure_probability += 0.0001  # Slight increase over time

            # Introduce occasional anomalies
            if random.random() < 0.005:
                temp += np.random.uniform(5, 10)  # Sudden temperature spike
                vibration += np.random.uniform(0.5, 1.5)  # Vibration anomaly
                pressure += np.random.uniform(10, 20)  # Pressure surge

            # Check if failure occurs based on conditions or probability
            failure = 0
            failure_type = "None"
            if temp > 80 or vibration > 3 or pressure > 200 or random.random() < failure_probability:
                failure = 1
                failure_type = random.choice(["Overheating", "Mechanical Wear", "Pressure Leak"])
                last_maintenance = time  # Reset maintenance cycle
                temp = 30 + np.random.normal(0, 2)
                vibration = 0.5 + np.random.normal(0, 0.1)
                pressure = 100 + np.random.normal(0, 5)
                cycles = 0
                failure_probability = 0.001  # Reset failure probability after maintenance
            
            # Append data point
            data.append([machine_id, time, temp, vibration, pressure, cycles, last_maintenance, failure, failure_type])
            time += timedelta(minutes=(60 / sampling_rate))

    return pd.DataFrame(data, columns=["Machine_ID", "Timestamp", "Temperature", "Vibration", "Pressure", "Cycles", "Last_Maintenance", "Failure", "Failure_Type"])

# Generate synthetic dataset
df = generate_synthetic_data(machines, days, sampling_rate)

# Drop non-numeric columns before applying SMOTE
df_preprocessed = df.drop(columns=["Machine_ID", "Timestamp", "Last_Maintenance", "Failure_Type"])

# Split into features and labels
X = df_preprocessed.drop(columns=["Failure"])
y = df_preprocessed["Failure"]

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 50% failure cases
X_resampled, y_resampled = smote.fit_resample(X, y)

# Reconstruct the balanced DataFrame
df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced["Failure"] = y_resampled

# Save to CSV
df_balanced.to_csv("balanced_synthetic_maintenance_data.csv", index=False)

# Show dataset distribution
print(df_balanced["Failure"].value_counts())