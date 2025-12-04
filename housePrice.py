import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# --- PAGE SETUP ---
st.set_page_config(
    page_title="House Price Predictor", 
    page_icon="🏠",
    layout="wide"
)

# Custom CSS to make the result look like a card
# ADDED: "color: black" to force text visibility in Dark Mode
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #d1d1d1;
        color: black !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #31333F !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 California Real Estate Estimator")
st.write("Adjust the parameters below to estimate property value.")

# --- LOAD AND TRAIN MODEL ---
@st.cache_resource
def train_model():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df[['MedInc', 'HouseAge', 'AveRooms']]
    y = df['MedHouseVal']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")

# 1. ADDED FLAGS to the options
currency_option = st.sidebar.selectbox(
    "Select Currency",
    ("🇺🇸 USD ($)", "🇪🇺 EUR (€)", "🇹🇳 TND (DT)")
)

# Exchange Rates (Approximate)
rates = {
    "🇺🇸 USD ($)": 1.0,
    "🇪🇺 EUR (€)": 0.95,
    "🇹🇳 TND (DT)": 2.94
}

# 2. SYMBOL MAPPING to get clean symbols
symbols = {
    "🇺🇸 USD ($)": "$",
    "🇪🇺 EUR (€)": "€",
    "🇹🇳 TND (DT)": "DT"
}

currency_rate = rates[currency_option]
currency_symbol = symbols[currency_option]

# --- MAIN INPUTS ---
st.subheader("Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    income_input = st.number_input(
        "Annual Income (local currency)", 
        min_value=5000, 
        max_value=150000, 
        value=30000, 
        step=1000
    )
    processed_income = income_input / 10000

with col2:
    age = st.slider("House Age (Years)", 1, 50, 20)

with col3:
    rooms = st.slider("Number of Rooms", 1, 10, 5)

# --- PREDICTION ---
if st.button("Calculate Estimate", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'MedInc': [processed_income],
        'HouseAge': [age],
        'AveRooms': [rooms]
    })
    
    # 1. Predict ($100k units)
    prediction_raw = model.predict(input_data)[0]
    
    # 2. Convert to USD
    price_usd = prediction_raw * 100000
    
    # 3. Convert to selected currency
    price_final = price_usd * currency_rate
    
    # 4. Make it a Pure INT
    price_final_int = int(price_final)
    
    # 5. Display Result
    # FORMAT: Symbol in front (e.g., "DT 100,000")
    st.metric(
        label=f"Estimated Price in {currency_option.split(' ')[2]}", 
        value=f"{currency_symbol} {price_final_int:,.0f}"
    )

    if price_usd > 200000:
        st.info("📈 Market Analysis: High Value Property")
    else:
        st.success("📉 Market Analysis: Affordable Entry-Level Property")