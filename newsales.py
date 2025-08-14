import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load dataset
data = pd.read_csv("product_weather_sales.csv")

# Weather model
weather_X = data[["Temperature", "Humidity", "WindSpeed", "Pressure"]]
weather_y = data["Weather"]

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    weather_X, weather_y, test_size=0.2, random_state=42
)
weather_model = LogisticRegression(max_iter=200)
weather_model.fit(Xw_train, yw_train)

# Sales model
sales_X = data[["Temperature", "Humidity", "WindSpeed", "Pressure"]]
sales_y = data[["ColdDrinks", "UmbrellaRaincoat", "HotDrinks", "Medicines"]]

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    sales_X, sales_y, test_size=0.2, random_state=42
)
sales_model = LinearRegression()
sales_model.fit(Xs_train, ys_train)

# Streamlit UI
st.title("🌦️ Weather-based Product Sales & Stock Recommendation")
st.write("Predict the weather and get product stock recommendations based on conditions.")

# User inputs
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)

if st.button("Predict Weather & Sales"):
    # Prepare input
    future_data = pd.DataFrame({
        "Temperature": [temp],
        "Humidity": [humidity],
        "WindSpeed": [wind],
        "Pressure": [pressure]
    })

    # Predict weather
    predicted_weather = weather_model.predict(future_data)[0]

    # Predict sales
    future_pred = sales_model.predict(future_data)[0]
    future_pred = [max(0, round(val)) for val in future_pred]  # No negative values

    # Prepare recommendation text
    recommendation_text = (
        f"🌤 Predicted Weather: {predicted_weather}\n"
        f"📦 Recommended Stock Based on Weather\n"
        f"-----------------------------------\n"
        f"🥤 Cold Drinks: {future_pred[0]} units\n"
        f"🌂 Umbrella/Raincoat: {future_pred[1]} units\n"
        f"☕ Hot Drinks: {future_pred[2]} units\n"
        f"💊 Medicines: {future_pred[3]} units\n"
    )

    # Save to .txt file
    with open("recommendation.txt", "w", encoding="utf-8") as f:
        f.write(recommendation_text)

    # Show on Streamlit
    st.subheader("🌤 Predicted Weather")
    st.write(predicted_weather)

    st.subheader("📦 Recommended Stock")
    st.text(recommendation_text)
    st.success("Recommendation saved to recommendation.txt ✅")
