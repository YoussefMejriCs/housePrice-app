import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# --- PAGE SETUP ---
st.set_page_config(
    page_title="House Price Predictor", 
    page_icon="🏠",
    layout="wide"  # Makes the app use the full width of the screen
)

# Custom CSS to make it look nicer
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #d1d1d1;
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
currency_option = st.sidebar.selectbox(
    "Select Currency",
    ("USD ($)", "EUR (€)", "TND (DT)")
)

# Exchange Rates (Approximate as of Late 2025)
rates = {
    "USD ($)": 1.0,
    "EUR (€)": 0.95,   # 1 USD = 0.95 Euro
    "TND (DT)": 2.94   # 1 USD = 2.94 Tunisian Dinar
}
currency_rate = rates[currency_option]
currency_symbol = currency_option.split(" ")[1].replace("(", "").replace(")", "")

# --- MAIN INPUTS (Designed with Columns) ---
st.subheader("Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    # We multiply by 10k so user sees "30,000" (Int) instead of "3.0" (Float)
    income_input = st.number_input(
        "Annual Income (in local currency)", 
        min_value=5000, 
        max_value=150000, 
        value=30000, 
        step=1000
    )
    # Convert back to the scale the model expects (units of $10k)
    # We assume the input matches the currency scale relative to USD for simplicity,
    # or just treat it as "Standard of Living" units.
    # For this simple demo, we convert the raw input back to the 3.0 scale roughly.
    processed_income = income_input / 10000

with col2:
    age = st.slider("House Age (Years)", 1, 50, 20)

with col3:
    rooms = st.slider("Number of Rooms", 1, 10, 5)

# --- PREDICTION ---
input_data = pd.DataFrame({
    'MedInc': [processed_income],
    'HouseAge': [age],
    'AveRooms': [rooms]
})

# Display User Input Table (Without the "Useless Column 1")
st.write("### Summary of Inputs")
st.dataframe(input_data, hide_index=True)

if st.button("Calculate Estimate", type="primary"):
    # 1. Predict (Result is in units of $100,000 USD)
    prediction_raw = model.predict(input_data)[0]
    
    # 2. Convert to actual USD value
    price_usd = prediction_raw * 100000
    
    # 3. Convert to selected currency
    price_final = price_usd * currency_rate
    
    # 4. Make it a Pure INT (No floats!)
    price_final_int = int(price_final)
    
    # 5. Display with a nice Metric Card
    st.metric(
        label=f"Estimated Price in {currency_option}", 
        value=f"{currency_symbol} {price_final_int:,.0f}"
    )

    # Logic explanation
    if price_usd > 200000:
        st.info("📈 Market Analysis: High Value Property")
    else:
        st.success("📉 Market Analysis: Affordable Entry-Level Property")