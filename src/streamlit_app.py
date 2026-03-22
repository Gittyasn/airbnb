import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Airbnb Price Predictor", layout="wide", page_icon="🏠")

# Custom Styling (Premium Look)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF5A5F;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF4449;
        color: white;
    }
    h1 {
        color: #FF5A5F;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('src/model.pkl')

# Load Data (for filters/EDA)
@st.cache_data
def load_data():
    df = pd.read_csv('data/AirBNB.csv', low_memory=False)
    # Basic cleaning corresponding to the analysis script
    df['log_price'] = pd.to_numeric(df['log_price'], errors='coerce')
    df.dropna(subset=['log_price'], inplace=True)
    return df

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error("Error loading models or data. Please ensure 'src/model.pkl' and 'data/AirBNB.csv' exist.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=150)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home & Predictor", "Data Analysis", "About"])

if menu == "Home & Predictor":
    st.title("🏠 Airbnb Price Prediction")
    st.subheader("Predict the estimated price of your listing using Machine Learning.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Property Details")
        room_type = st.selectbox("Room Type", df['room_type'].dropna().unique())
        city = st.selectbox("City", df['city'].dropna().unique() if 'city' in df.columns else ['NYC', 'SF', 'LA', 'DC', 'Chicago', 'Boston'])
        accommodates = st.number_input("Accommodates", min_value=1, max_value=16, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=8.0, value=1.0, step=0.5)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1)
        beds = st.number_input("Beds", min_value=1, max_value=18, value=1)

    with col2:
        st.markdown("### Extra Information")
        cancellation_policy = st.selectbox("Cancellation Policy", df['cancellation_policy'].dropna().unique() if 'cancellation_policy' in df.columns else ['strict', 'flexible', 'moderate'])
        cleaning_fee = st.selectbox("Cleaning Fee Included?", ["True", "False"])
        instant_bookable = st.selectbox("Instant Bookable?", ["t", "f"])
        number_of_reviews = st.slider("Number of Reviews", 0, 500, 20)
        review_scores_rating = st.slider("Review Scores Rating", 20, 100, 95)
        host_identity_verified = st.selectbox("Host Identity Verified?", ["t", "f"])
        host_has_profile_pic = st.selectbox("Host Profile Pic?", ["t", "f"])

    # Prediction Button
    if st.button("Calculate Predicted Price"):
        # Prepare Input Data
        input_dict = {
            'accommodates': [accommodates],
            'bathrooms': [bathrooms],
            'bedrooms': [bedrooms],
            'beds': [beds],
            'number_of_reviews': [number_of_reviews],
            'review_scores_rating': [review_scores_rating],
            'room_type': [room_type],
            'cancellation_policy': [cancellation_policy],
            'city': [city],
            'host_has_profile_pic': [host_has_profile_pic],
            'host_identity_verified': [host_identity_verified],
            'instant_bookable': [instant_bookable],
            'cleaning_fee': [cleaning_fee]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Ensure types match the training data prep (astype(str) for categorical)
        cat_cols = ['room_type', 'cancellation_policy', 'city', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'cleaning_fee']
        for col in cat_cols:
            input_df[col] = input_df[col].astype(str)

        # Predict
        log_pred = model.predict(input_df)[0]
        actual_price = np.exp(log_pred)
        
        st.success(f"### 🎯 Estimated Price: ${actual_price:.2f} per night")
        st.info(f"The log-transformed price predicted by the model is {log_pred:.4f}")

elif menu == "Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    st.markdown("#### Distribution of Prices")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['log_price'], bins=30, kde=True, color='#FF5A5F')
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Price by Room Type")
        fig2, ax2 = plt.subplots()
        sns.barplot(x='room_type', y='log_price', data=df, palette='viridis')
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        
    with col2:
        st.markdown("#### Accommodates vs Price")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='accommodates', y='log_price', data=df, alpha=0.1, color='#FF5A5F')
        st.pyplot(fig3)

elif menu == "About":
    st.title("ℹ️ About the Project")
    st.write("""
    This project is a COMPLETE Machine Learning case study to predict Airbnb listing prices.
    
    **Features used for prediction:**
    - Property size (bedrooms, bathrooms, beds, accommodates)
    - Host verification status
    - Local market (city)
    - Guest ratings & reviews
    - Cancellation & cleaning policies
    
    **Model:** Multiple Linear Regression
    **Developed by:** Antigravity (Professional Data Scientist)
    """)

st.sidebar.markdown("---")
st.sidebar.info("Airbnb Price Case Study")
