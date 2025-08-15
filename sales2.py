import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load dataset
data = pd.read_csv("product_weather_sales.csv")

# Weather model
weather_X = data[["Temperature", "Humidity", "WindSpeed", "Precipitation"]]
weather_y = data["Weather"]

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    weather_X, weather_y, test_size=0.2, random_state=42
)
weather_model = LogisticRegression(max_iter=200)
weather_model.fit(Xw_train, yw_train)

# Sales model
sales_X = data[["Temperature", "Humidity", "WindSpeed", "Precipitation"]]
sales_y = data[["ColdDrinks", "UmbrellaRaincoat", "HotDrinks", "Medicines"]]

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    sales_X, sales_y, test_size=0.2, random_state=42
)
sales_model = LinearRegression()
sales_model.fit(Xs_train, ys_train)

# Function to map numbers to "Low", "Moderate", "High"
def demand_level(value):
    if value < 20:
        return "Low"
    elif value < 50:
        return "Moderate"
    else:
        return "High"

# Streamlit UI
st.title("🌦️ Weather-based Product Demand Recommendation")
st.write("Predict the weather and get product demand levels based on conditions.")

# User inputs
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
precipitation = st.number_input("Precipitation (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)

if st.button("Predict Weather & Demand"):
    # Prepare input
    future_data = pd.DataFrame({
        "Temperature": [temp],
        "Humidity": [humidity],
        "WindSpeed": [wind],
        "Precipitation": [precipitation]
    })

    # Predict weather
    predicted_weather = weather_model.predict(future_data)[0]

    # Predict sales demand
    future_pred = sales_model.predict(future_data)[0]
    future_pred = [max(0, round(val)) for val in future_pred]  # Avoid negatives

    # Convert to High / Moderate / Low
    demand_levels = [demand_level(val) for val in future_pred]

    # Prepare recommendation text
    recommendation_text = (
        f"🌤 Predicted Weather: {predicted_weather}\n"
        f"📦 Recommended Demand Levels\n"
        f"-----------------------------------\n"
        f"🥤 Cold Drinks: {demand_levels[0]}\n"
        f"🌂 Umbrella/Raincoat: {demand_levels[1]}\n"
        f"☕ Hot Drinks: {demand_levels[2]}\n"
        f"💊 Medicines: {demand_levels[3]}\n"
    )

    # Save to .txt file
    with open("recommendation.txt", "w", encoding="utf-8") as f:
        f.write(recommendation_text)

    # Show on Streamlit
    st.subheader("🌤 Predicted Weather")
    st.write(predicted_weather)

    st.subheader("📦 Recommended Demand Levels")
    st.text(recommendation_text)
    st.success("Recommendation saved to recommendation.txt ✅")
