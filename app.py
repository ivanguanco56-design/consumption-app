# ================= IMPORT LIBRARIES =================
import streamlit as st
import numpy as np
import pickle
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Electric Power Predictor",
    page_icon="⚡",
    layout="centered"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "power_model.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model

model = load_model()

# ================= TITLE =================
st.title("Electric Power Consumption Predictor")
st.write("Enter the values below to estimate household power usage.")

# ================= INPUT SECTION =================
st.subheader("Input Features")

reactive_power = st.number_input("Global Reactive Power", value=0.1)
voltage = st.number_input("Voltage", value=240.0)
intensity = st.number_input("Global Intensity", value=2.0)
sub1 = st.number_input("Sub Metering 1", value=0.0)
sub2 = st.number_input("Sub Metering 2", value=0.0)
sub3 = st.number_input("Sub Metering 3", value=1.0)

# ================= PREDICTION =================
if st.button("Predict Power Usage"):

    input_data = np.array([[reactive_power, voltage, intensity, sub1, sub2, sub3]])
    prediction = model.predict(input_data)

    st.subheader("Result")
    st.success(f"Predicted Power Usage: {prediction[0]:.3f} kW")

    st.write("This value represents the estimated household electricity consumption.")

# ================= FOOTER =================
st.markdown("---")
st.caption("AI-Based Electricity Consumption Prediction System | Capstone Project")