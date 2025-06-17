import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

# UI Title
st.title("🏡 House Price Predictor")

st.write("Enter the house details below to estimate the price:")

# --- User Inputs ---
area = st.number_input("🧱 Area (in square feet)", min_value=200, max_value=10000, value=1000, step=50)
bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("🚿 Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
parking = st.number_input("🚗 Number of Parking Lots", min_value=0, max_value=5, value=1, step=1)

# --- Predict Button ---
if st.button("Predict Price"):
    # Combine input into array
    features = np.array([[area, bedrooms, bathrooms, parking]])
    
    # Predict using model
    predicted_price = model.predict(features)[0]
    
    # Display result
    st.success(f"💰 Estimated House Price: ₹{predicted_price:,.2f}")