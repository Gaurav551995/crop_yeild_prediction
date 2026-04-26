import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
le_crop = joblib.load("crop_encoder.pkl")
le_dist = joblib.load("dist_encoder.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Crop Prediction", layout="centered")

# =========================
# STYLING
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# TITLE
# =========================
st.title("🌾 Crop Recommendation System")

st.write("Enter details to predict best crops")

# =========================
# INPUT FIELDS
# =========================
year = st.number_input("Year", min_value=2000, max_value=2030, value=2018)

district = st.selectbox("Select District", le_dist.classes_)

area = st.number_input("Area (1000 ha)", value=100.0)
rainfall = st.number_input("Total Rainfall", value=1000.0)
temp = st.number_input("Avg Temperature", value=25.0)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict Crop"):
    
    dist_encoded = le_dist.transform([district])[0]
    
    input_data = pd.DataFrame([[year, dist_encoded, area, rainfall, temp]],
                              columns=['Year', 'Dist Name', 'Area(1000 ha)', 'Total Rainfall', 'Avg Temp'])
    
    input_scaled = scaler.transform(input_data)
    
    probs = model.predict_proba(input_scaled)[0]
    classes = le_crop.inverse_transform(model.classes_)
    
    top3 = np.argsort(probs)[-3:][::-1]
    
    st.subheader("🌱 Top 3 Crops:")
    
    for i in top3:
        st.write(f"✅ {classes[i]} ({probs[i]*100:.2f}%)")