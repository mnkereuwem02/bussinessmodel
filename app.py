


import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
import streamlit as st

# Load pre-saved objects; make sure the following files are in your working directory:
#   model.pkl, scaler.pkl, and encoders.pkl
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('encoders.pkl')

def encode_feature(feature_name, value):
    try:
        # If the value is already numeric, return it as an int.
        return int(value)
    except (ValueError, TypeError):
        if feature_name in label_encoders:
            encoder = label_encoders[feature_name]
            # Preprocess value if needed
            value_processed = value.strip() if isinstance(value, str) else value
            # Check if the processed value is among the known classes:
            if value_processed not in encoder.classes_:
                raise ValueError(f"Value '{value_processed}' not seen during training for feature '{feature_name}'. Known classes: {encoder.classes_}")
            # Return the encoded value
            return int(encoder.transform([value_processed])[0])
        return value


# Sidebar for navigation
st.sidebar.title("Navigation")
operation = st.sidebar.radio("Select Operation", ("Predict Sales", "Optimize Pricing"))

# Main App Title
st.title("Sales Prediction & Pricing Optimization App")

if operation == "Predict Sales":
    st.header("Predict Sales Quantity")
    st.write("Enter the required parameters to predict the sales quantity.")
    


    col1, col2 = st.columns(2)
    with col1:
        price_x = st.number_input("Your Product's Price", min_value=0.0, value=500.0, step=1.0)
        price_y = st.number_input("Competitor's Price", min_value=0.0, value=200.0, step=1.0)
        order_year = st.number_input("Order Year", min_value=2000, max_value=2100, value=2022, step=1)
        
    
    state_cities ={'Florida': ['Miami', 'Orlando', 'Tampa'], 'Illinois': ['Chicago', 'Springfield', 'Peoria'], 
 'New York': ['Buffalo', 'New York City', 'Rochester'], 'California': ['Los Angeles', 'San Diego', 'San Francisco'], 
 'Texas': ['Dallas', 'Austin', 'Houston'], 'Ohio': ['Columbus', 'Cincinnati', 'Cleveland']}

    Cat_sub= {'Electronics': ['Electronic Games', 'Printers', 'Laptops', 'Phones'], 
 'Office Supplies': ['Pens', 'Markers', 'Paper', 'Binders'],
 'Furniture': ['Tables', 'Chairs', 'Sofas', 'Bookcases']}
    with col2:
        payment_mode = st.selectbox("PaymentMode", ['UPI', 'Debit Card', 'EMI', 'Credit Card', 'COD'])
        state_val =  st.selectbox("State", list(state_cities.keys()))
        city_val = st.selectbox("City", state_cities[state_val])
        category  = st.selectbox("Category", list(Cat_sub.keys()))
        sub_category = st.selectbox("Sub-Category", Cat_sub[category])
        customer_segment = st.text_input("Customer Segment (e.g., 0)", value="0")

    if st.button("Predict Sales"):
        # Encode categorical features
        payment_mode_enc = encode_feature("PaymentMode", payment_mode)
        state_enc = encode_feature("State", state_val)
        city_enc = encode_feature("City", city_val)
        customer_segment_enc = encode_feature("Customer_Segment", customer_segment)

        # Derived feature: Price_Ratio
        price_ratio = price_x / price_y if price_y != 0 else 0.0

        # Order of features as used during training:
        # ['Price_x', 'Price_y', 'Price_Ratio', 'Customer_Segment', 'PaymentMode', 'State', 'City', 'Order Year']
        features = np.array([[price_x, price_y, price_ratio, customer_segment_enc,
                              payment_mode_enc, state_enc, city_enc, order_year]])
        features_scaled = scaler.transform(features)

        # Predict the sales quantity using the pre-trained model
        predicted_quantity = model.predict(features_scaled)[0]
        st.success(f"Predicted Sales Quantity: {predicted_quantity:.2f}")

elif operation == "Optimize Pricing":
    st.header("Optimize Pricing")
    st.write("Enter constant parameters and a candidate price range to find the optimal pricing based on maximum revenue.")

    col1, col2 = st.columns(2)
    with col1:
        price_y = st.number_input("Competitor's Price", min_value=0.0, value=200.0, step=1.0)
        order_year = st.number_input("Order Year", min_value=2000, max_value=2100, value=2022, step=1)
        min_price = st.number_input("Minimum Price (min_price)", min_value=0.0, value=400.0, step=1.0)
        
    state_cities ={'Florida': ['Miami', 'Orlando', 'Tampa'], 'Illinois': ['Chicago', 'Springfield', 'Peoria'], 
 'New York': ['Buffalo', 'New York City', 'Rochester'], 'California': ['Los Angeles', 'San Diego', 'San Francisco'], 
 'Texas': ['Dallas', 'Austin', 'Houston'], 'Ohio': ['Columbus', 'Cincinnati', 'Cleveland']}

    Cat_sub= {'Electronics': ['Electronic Games', 'Printers', 'Laptops', 'Phones'], 
 'Office Supplies': ['Pens', 'Markers', 'Paper', 'Binders'],
 'Furniture': ['Tables', 'Chairs', 'Sofas', 'Bookcases']}
    with col2:
        
        payment_mode = st.selectbox("PaymentMode",['UPI', 'Debit Card', 'EMI', 'Credit Card', 'COD'])
        state_val = st.selectbox("State", list(state_cities.keys()))
        city_val =  st.selectbox("City", state_cities[state_val])
        category  = st.selectbox("Category", list(Cat_sub.keys()))
        sub_category = st.selectbox("Sub-Category", Cat_sub[category])
        customer_segment = st.text_input("Customer Segment (e.g., 0)", value="0")
        max_price = st.number_input("Maximum Price (max_price)", min_value=0.0, value=600.0, step=1.0)

    step = st.number_input("Step Size", min_value=0.1, value=5.0)

    if st.button("Optimize Price"):
        # Encode constant categorical features
        payment_mode_enc = encode_feature("PaymentMode", payment_mode)
        state_enc = encode_feature("State", state_val)
        city_enc = encode_feature("City", city_val)
        customer_segment_enc = encode_feature("Customer_Segment", customer_segment)

        best_price = None
        best_quantity = None
        max_revenue = -np.inf
        candidate_results = []
        
        # Suppose 'payment_mode_enc' is the encoded value (e.g., 1)
        original_payment_mode = label_encoders["PaymentMode"].inverse_transform([payment_mode_enc])[0]


        # Iterate through the candidate prices
        for price_x in np.arange(min_price, max_price + step, step):
            price_ratio = price_x / price_y if price_y != 0 else 0.0
            features = np.array([[price_x, price_y, price_ratio, customer_segment_enc,
                                  payment_mode_enc, state_enc, city_enc, order_year]])
            features_scaled = scaler.transform(features)
            predicted_qty = model.predict(features_scaled)[0]
            revenue = price_x * predicted_qty

            candidate_results.append({
                "Price_x": price_x,
                "Predicted Quantity": predicted_qty,
                "Revenue": revenue
            })

            if revenue > max_revenue:
                max_revenue = revenue
                best_price = price_x
                best_quantity = predicted_qty

        st.success(f"Optimal Price: {best_price:.2f}, \n"
                   f" Predicted Quantity: {best_quantity:.2f}, \n"
                   f" Expected Revenue: {max_revenue:.2f}")
        st.write("Candidate Price Results:")
        st.dataframe(pd.DataFrame(candidate_results))
        
    
