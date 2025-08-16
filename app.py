import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model.h5", compile=False)  
scaler = joblib.load("scaler.pkl")

st.title("🚗 Car CO₂ Emission Prediction")

st.write("Enter the vehicle details below to predict the CO₂ emissions (g/km):")

# Input fields
model_year = st.number_input("Model Year", min_value=1990, max_value=2030, step=1, value=2020)
brand = st.selectbox("Brand", ["Toyota", "Ford", "BMW", "Mercedes", "Honda", "Other"])
vehicle_class = st.selectbox("Vehicle Class", ["Compact", "SUV", "Truck", "Van", "Luxury", "Other"])
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1, value=2.0)
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, step=1, value=4)
transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "Other"])
fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"])
fuel_city = st.number_input("Fuel Consumption City (L/100km)", min_value=0.0, step=0.1, value=9.0)
fuel_hwy = st.number_input("Fuel Consumption Hwy (L/100km)", min_value=0.0, step=0.1, value=6.0)
fuel_comb = st.number_input("Fuel Consumption Comb (L/100km)", min_value=0.0, step=0.1, value=7.5)
fuel_mpg = st.number_input("Fuel Consumption Comb (MPG)", min_value=0.0, step=0.1, value=30.0)

# Convert categorical features to numeric (simple encoding)
brand_map = {"Toyota": 1, "Ford": 2, "BMW": 3, "Mercedes": 4, "Honda": 5, "Other": 0}
vehicle_map = {"Compact": 1, "SUV": 2, "Truck": 3, "Van": 4, "Luxury": 5, "Other": 0}
trans_map = {"Automatic": 1, "Manual": 2, "CVT": 3, "Other": 0}
fuel_map = {"Gasoline": 1, "Diesel": 2, "Electric": 3, "Hybrid": 4, "Other": 0}

brand_val = brand_map[brand]
vehicle_val = vehicle_map[vehicle_class]
trans_val = trans_map[transmission]
fuel_val = fuel_map[fuel_type]

# Prepare input
features = np.array([[model_year, brand_val, vehicle_val, engine_size, cylinders,
                      trans_val, fuel_val, fuel_city, fuel_hwy, fuel_comb, fuel_mpg]])

# Scale features
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict CO₂ Emission"):
    prediction = model.predict(features_scaled)
    st.success(f"Estimated CO₂ Emission: {prediction[0][0]:.2f} g/km")
