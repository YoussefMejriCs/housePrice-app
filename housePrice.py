import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# --- PAGE SETUP ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 Simple House Price Predictor")
st.write("This app predicts the **Median House Value** in California based on neighborhood data.")

# --- LOAD AND TRAIN MODEL (On the fly) ---
# We train it right here so you don't have to manage multiple files.
@st.cache_resource
def train_model():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    # Use 3 simple features
    X = df[['MedInc', 'HouseAge', 'AveRooms']]
    y = df['MedHouseVal']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

# --- USER INPUTS (SIDEBAR) ---
st.sidebar.header("Specify Input Parameters")

# Sliders for user to adjust
income = st.sidebar.slider("Median Income (in $10k)", 0.5, 15.0, 3.0)
age = st.sidebar.slider("House Age (Years)", 1, 52, 20)
rooms = st.sidebar.slider("Avg Rooms per Household", 1.0, 10.0, 5.0)

# --- PREDICTION ---
# Create a dataframe for the input
input_data = pd.DataFrame({
    'MedInc': [income],
    'HouseAge': [age],
    'AveRooms': [rooms]
})

# Display User Input
st.subheader("User Input:")
st.write(input_data)

# Button to predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    # The target is in units of $100,000
    price = prediction[0] * 100000 
    
    st.success(f"Estimated House Price: ${price:,.2f}")
    
    # Simple logic explanation
    if price > 200000:
        st.write("📈 This is considered a **High Value** property.")
    else:
        st.write("📉 This is considered an **Affordable** property.")