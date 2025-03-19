import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load and process the dataset
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("ai 2020.csv")
    X = df.drop(columns=["UDI", "Product ID", "Machine failure"])
    y = df["Machine failure"]

    # One-hot encode 'Type'
    X = pd.get_dummies(X, columns=["Type"], drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "scaler.pkl")

    return X_train, X_test, y_train, y_test, X.columns.tolist()

# Train and save model
@st.cache_resource
def train_and_save_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    model.save("failure_model.h5")
    return model

# NLP Explanation method
def generate_nlp_explanation(input_data, prediction_label):
    explanation = []
    if prediction_label == "Failure":
        explanation.append("The machine is at high risk of failure based on the provided parameters.")
    else:
        explanation.append("The machine is operating normally based on the provided parameters.")

    air_temp, process_temp, rot_speed, torque, tool_wear, twf, hdf, pwf, osf, rnf, type_l, type_m = input_data

    if tool_wear > 150:
        explanation.append("High tool wear is detected, which can increase the likelihood of mechanical failures.")
    if rot_speed > 3000:
        explanation.append("The rotational speed is unusually high, which could stress machine components.")
    if torque > 50:
        explanation.append("Elevated torque might indicate the machine is under excessive load.")

    failure_types = [
        (twf, "Tool Wear Failure"),
        (hdf, "Heat Dissipation Failure"),
        (pwf, "Power Failure"),
        (osf, "Overstrain Failure"),
        (rnf, "Random Failure")
    ]
    for failure, desc in failure_types:
        if failure:
            explanation.append(f"{desc} condition is present, increasing failure probability.")

    return explanation

# Precaution recommendations
def generate_precautions(input_data, prediction_label):
    air_temp, process_temp, rot_speed, torque, tool_wear, twf, hdf, pwf, osf, rnf, type_l, type_m = input_data

    precautions = []

    if prediction_label == "Failure":
        precautions.append("⚠️ Immediate maintenance check is recommended to avoid unexpected breakdown.")

    if torque > 50:
        precautions.append("🔧 High torque detected — Check for possible overstrain and inspect lubrication and mechanical components.")
    if tool_wear > 150:
        precautions.append("🔧 Tool wear is high — Consider replacing or sharpening tools and closely monitor future wear rate.")
    if rot_speed > 3000:
        precautions.append("⚙️ High rotational speed detected — Ensure bearings are properly lubricated and balanced.")

    if twf:
        precautions.append("🛠️ Tool Wear Failure detected — Replace worn-out tools and review operational procedures.")
    if hdf:
        precautions.append("❄️ Heat Dissipation Failure detected — Improve cooling system and monitor temperature sensors.")
    if pwf:
        precautions.append("⚡ Power Failure detected — Review electrical supply and consider backup power options.")
    if osf:
        precautions.append("⚠️ Overstrain Failure detected — Evaluate operational limits and review stress factors.")
    if rnf:
        precautions.append("❓ Random Failure detected — Perform comprehensive system diagnostics.")

    if not precautions:
        precautions.append("✅ No immediate precautions needed. Continue regular monitoring and maintenance.")

    return precautions

# Load and train model
X_train, X_test, y_train, y_test, feature_columns = load_and_preprocess_data()
model = train_and_save_model(X_train, y_train)

# Predict and calculate accuracy
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

# Save predictions for confusion matrix display
np.save("y_test.npy", y_test)
np.save("y_pred.npy", y_pred)

# Load scaler
scaler = joblib.load("scaler.pkl")

# Streamlit App
st.title("Machine Failure Prediction & Explanation App")
st.write(f"### Model Accuracy on Test Data: **{accuracy:.2%}**")

with st.form("user_input_form"):
    st.subheader("Enter Machine Parameters")

    air_temp = st.number_input("Air Temperature [K]", value=300.0)
    process_temp = st.number_input("Process Temperature [K]", value=310.0)
    rot_speed = st.number_input("Rotational Speed [rpm]", value=1500)
    torque = st.number_input("Torque [Nm]", value=40.0)
    tool_wear = st.number_input("Tool Wear [min]", value=100)

    twf = 1 if st.selectbox("Tool Wear Failure (TWF)", ["No", "Yes"]) == "Yes" else 0
    hdf = 1 if st.selectbox("Heat Dissipation Failure (HDF)", ["No", "Yes"]) == "Yes" else 0
    pwf = 1 if st.selectbox("Power Failure (PWF)", ["No", "Yes"]) == "Yes" else 0
    osf = 1 if st.selectbox("Overstrain Failure (OSF)", ["No", "Yes"]) == "Yes" else 0
    rnf = 1 if st.selectbox("Random Failure (RNF)", ["No", "Yes"]) == "Yes" else 0

    machine_type = st.selectbox("Machine Type", ["L", "M"])

    submitted = st.form_submit_button("Predict")

if submitted:
    type_l, type_m = 0, 0
    if machine_type == "L":
        type_l = 1
    elif machine_type == "M":
        type_m = 1

    input_data = np.array([
        air_temp, process_temp, rot_speed, torque, tool_wear,
        twf, hdf, pwf, osf, rnf, type_l, type_m
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0][0]
    prediction_label = "Failure" if prediction > 0.5 else "No Failure"
    st.write(f"### Prediction: {prediction_label} (Confidence: {prediction:.2f})")

    explanation = generate_nlp_explanation(input_data[0], prediction_label)
    st.write("### Explanation - Key Factors")
    for line in explanation:
        st.write(f"- {line}")

    precautions = generate_precautions(input_data[0], prediction_label)
    st.write("### Recommended Precautions")
    for p in precautions:
        st.write(f"- {p}")

if st.checkbox("Show Confusion Matrix"):
    y_test = np.load("y_test.npy")
    y_pred = np.load("y_pred.npy")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))