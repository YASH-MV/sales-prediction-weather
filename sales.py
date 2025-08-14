import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("product_weather_sales.csv")

# Features and target
X = data[["Temperature", "Humidity", "WindSpeed", "Pressure"]]
y = data[["ColdDrinks", "UmbrellaRaincoat", "HotDrinks", "Medicines"]]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌦️ Weather-based Product Sales Prediction")
st.write("Predict recommended product stock levels based on future weather conditions.")

# User inputs
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)

if st.button("Predict Sales"):
    future_data = pd.DataFrame({
        "Temperature": [temp],
        "Humidity": [humidity],
        "WindSpeed": [wind],
        "Pressure": [pressure]
    })

    future_pred = model.predict(future_data)[0]

    st.subheader("📦 Recommended Stock")
    st.write(f"🥤 **Cold Drinks**: {round(future_pred[0])} units")
    st.write(f"🌂 **Umbrella/Raincoat**: {round(future_pred[1])} units")
    st.write(f"☕ **Hot Drinks**: {round(future_pred[2])} units")
    st.write(f"💊 **Medicines**: {round(future_pred[3])} units")
