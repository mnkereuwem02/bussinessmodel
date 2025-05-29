import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")  # dictionary of LabelEncoders

state_cities ={'Florida': ['Miami', 'Orlando', 'Tampa'], 'Illinois': ['Chicago', 'Springfield', 'Peoria'], 
 'New York': ['Buffalo', 'New York City', 'Rochester'], 'California': ['Los Angeles', 'San Diego', 'San Francisco'], 
 'Texas': ['Dallas', 'Austin', 'Houston'], 'Ohio': ['Columbus', 'Cincinnati', 'Cleveland']}

Cat_sub= {'Electronics': ['Electronic Games', 'Printers', 'Laptops', 'Phones'], 
 'Office Supplies': ['Pens', 'Markers', 'Paper', 'Binders'],
 'Furniture': ['Tables', 'Chairs', 'Sofas', 'Bookcases']}

st.title("Sales Profit Prediction App")

# Input widgets
amount = st.number_input("Amount", min_value=0)
quantity = st.number_input("Quantity", min_value=1)

category = st.selectbox("Category", list(Cat_sub.keys()))
sub_category = st.selectbox("Sub-Category", Cat_sub[category])
payment_mode = st.selectbox("Payment Mode", ['UPI', 'Debit Card', 'EMI', 'Credit Card', 'COD'])
state = st.selectbox("State", list(state_cities.keys()))
city = st.selectbox("City", state_cities[state])
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
