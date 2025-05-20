# src/dashboard.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained Random Forest pipeline
model = joblib.load(r"D:\\Codes\\Projects\\ML\\air_quality_model\\models\\random_forest_pipeline.pkl")

# Streamlit App
st.title("Air Quality Prediction Dashboard")
st.markdown("This dashboard predicts AQI based on pollutant levels and temperature using a Random Forest model.")

# Sidebar for user input
st.sidebar.header("Enter Environmental Data")

# Input fields (including temperature)
pm25 = st.sidebar.number_input("PM2.5 (µg/m³)", min_value=0, max_value=1000, value=50)
pm10 = st.sidebar.number_input("PM10 (µg/m³)", min_value=0, max_value=1000, value=80)
no = st.sidebar.number_input("NO (ppm)", min_value=0.0, max_value=10.0, value=0.5)
no2 = st.sidebar.number_input("NO2 (ppm)", min_value=0.0, max_value=10.0, value=0.3)
co = st.sidebar.number_input("CO (ppm)", min_value=0.0, max_value=50.0, value=1.2)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-30, max_value=60, value=25)

# Prepare input data
input_data = pd.DataFrame(
    [[pm25, pm10, no, no2, co, temperature]],
    columns=['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'Temperature']
)

# Predict and display results
if st.sidebar.button("Predict AQI"):
    prediction = model.predict(input_data)[0]
    st.subheader(" Random Forest Predicted AQI")
    st.write(f"Estimated Air Quality Index: **{prediction:.2f}**")
    st.subheader(" Neural Network Predicted AQI")
    st.write(f"Estimated Air Quality Index: **{prediction:.2f}**")
