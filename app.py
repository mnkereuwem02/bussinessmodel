import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")  # dictionary of LabelEncoders

st.title("Sales Profit Prediction App")

# Input widgets
amount = st.number_input("Amount", min_value=0)
quantity = st.number_input("Quantity", min_value=1)

category = st.selectbox("Category", ['Electronics', 'Office Supplies', 'Furniture'])
sub_category = st.selectbox("Sub-Category", [
    'Electronic Games', 'Printers', 'Pens', 'Laptops', 'Tables',
    'Chairs', 'Markers', 'Sofas', 'Paper', 'Binders', 'Phones', 'Bookcases'
])
payment_mode = st.selectbox("Payment Mode", ['UPI', 'Debit Card', 'EMI', 'Credit Card', 'COD'])
state = st.selectbox("State", ['Florida', 'Illinois', 'New York', 'California', 'Texas', 'Ohio'])
city = st.selectbox("City", [
    'Miami', 'Chicago', 'Buffalo', 'Orlando', 'Los Angeles',
    'New York City', 'Springfield', 'Rochester', 'Dallas', 'San Diego',
    'Austin', 'San Francisco', 'Columbus', 'Cincinnati', 'Cleveland',
    'Houston', 'Tampa', 'Peoria'
])
order_month = st.selectbox("Order Month", list(range(1, 13)))
order_year = st.selectbox("Order Year", [i for i in range(2020, 2051)])

# Predict button
if st.button("Predict Profit"):
    input_data = pd.DataFrame({
        'Amount': [amount],
        'Quantity': [quantity],
        'Category': [category],
        'Sub-Category': [sub_category],
        'PaymentMode': [payment_mode],
        'State': [state],
        'City': [city],
        'Order Month': [order_month],
        'Order Year': [order_year]
    })

    # Encode each categorical column using its own LabelEncoder
    for col in label_encoders:
        if col in input_data.columns:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Profit: ${prediction[0]:,.2f}")
