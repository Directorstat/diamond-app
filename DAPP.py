import os
import joblib

base_dir = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_dir, "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
encoder = joblib.load(os.path.join(base_dir, "encoder.pkl"))

import os
import streamlit as st
import pandas as pd
import joblib


base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "random_forest_model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")
encoder_path = os.path.join(base_dir, "encoder.pkl")

# LOAD FILES SAFELY

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

if not os.path.exists(scaler_path):
    st.error(f"Scaler file not found at: {scaler_path}")
    st.stop()

if not os.path.exists(encoder_path):
    st.error(f"Encoder file not found at: {encoder_path}")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

# APP UI

st.title("Diamond Price Prediction App")
st.write("Enter the features of the diamond to predict its price.")

carat = st.number_input("Carat", 0.0, 5.0, step=0.01)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.slider("Depth", 0.0, 100.0)
table = st.number_input("Table", 0.0, 100.0)
x = st.number_input("Length (x)", 0.0, 100.0)
y = st.number_input("Width (y)", 0.0, 100.0)
z = st.number_input("Depth (z)", 0.0, 100.0)

# PREDICTION

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    # Split
    cat_cols = ["cut", "color", "clarity"]
    num_cols = ["carat", "depth", "table", "x", "y", "z"]

    input_num = input_data[num_cols]
    input_cat = input_data[cat_cols]

    # Transform
    num_transformed = scaler.transform(input_num)
    cat_transformed = encoder.transform(input_cat)

    # Combine
    import numpy as np
    final_input = np.hstack((num_transformed, cat_transformed))

    # Predict
    prediction = model.predict(final_input)

    st.success(f"Predicted Price: ${prediction[0]:,.2f}")