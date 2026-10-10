from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="Sanchaar Bus Prediction API")

# Load model
model = joblib.load("sanchaar_passenger_prediction_model.joblib")

# Load dataset used for model development
data = pd.read_csv(
    "delhi_bus_passengers_2025_weather_crowd_based_synthetic.csv"
)

# Convert date and time
data["Date"] = pd.to_datetime(data["Date"])
data["Time"] = pd.to_datetime(
    data["Time"],
    format="%H:%M:%S"
)


# -----------------------------
# Home
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Sanchaar API is running!"
    }


# -----------------------------
# Bus decision
# -----------------------------
def get_bus_decision(
    predicted_passengers,
    current_buses,
    bus_capacity=50
):

    effective_capacity = bus_capacity * 0.80

    required_buses = int(
        np.ceil(
            predicted_passengers /
            effective_capacity
        )
    )

    occupancy = (
        predicted_passengers /
        (current_buses * bus_capacity)
    )

    if occupancy < 0.50:
        crowding_level = "LOW"

    elif occupancy < 0.80:
        crowding_level = "NORMAL"

    elif occupancy < 1.00:
        crowding_level = "HIGH"

    else:
        crowding_level = "CRITICAL"

    difference = required_buses - current_buses

    if difference > 0:
        action = f"DEPLOY {difference} BUS(ES)"

    elif difference < 0:
        action = f"REMOVE {abs(difference)} BUS(ES)"

    else:
        action = "MAINTAIN CURRENT BUSES"

    return {
        "predicted_passengers": round(
            predicted_passengers, 1
        ),
        "current_buses": current_buses,
        "required_buses": required_buses,
        "occupancy_percent": round(
            occupancy * 100, 1
        ),
        "crowding_level": crowding_level,
        "action": action
    }


# -----------------------------
# Prediction
# -----------------------------
@app.post("/predict")
def predict(data_input: dict):

    location = data_input["location"]
    date = pd.to_datetime(data_input["date"])
    time = pd.to_datetime(
        data_input["time"],
        format="%H:%M"
    )

    hour = time.hour

    # Find matching location + date + hour
    matching_rows = data[
        (data["Location"] == location) &
        (data["Date"] == date) &
        (data["Time"].dt.hour == hour)
    ]

    if matching_rows.empty:
        return {
            "error": "No matching data found for this location, date and time."
        }

    row = matching_rows.iloc[0]

    # Prepare model input
    model_input = pd.DataFrame([{
        "Location": row["Location"],
        "Hour": row["Time"].hour,
        "Month": row["Date"].month,
        "Day_of_Week": row["Day_of_Week"],
        "Is_Weekend": row["Is_Weekend"],
        "temp_c": row["temp_c"],
        "humidity": row["humidity"],
        "windspeed_kph": row["windspeed_kph"],
        "aqi_index": row["aqi_index"],
        "pm2_5": row["pm2_5"],
        "pm10": row["pm10"],
        "condition_text": row["condition_text"],
        "Nearby_Crowd_Count": row["Nearby_Crowd_Count"],
        "Holiday": row["Holiday"],
        "Event": row["Event"]
    }])

    # ML prediction
    prediction = round(model.predict(model_input)[0], 1)

    # Current buses
    current_buses = {
        "Anand Vihar": 2,
        "Connaught Place": 2,
        "Dwarka": 2,
        "IGI Airport": 2,
        "Okhla Phase III": 2,
        "Rohini": 2
    }

    buses = current_buses.get(
        location,
        2
    )

    # Decision
    result = get_bus_decision(
        predicted_passengers=float(prediction),
        current_buses=buses
    )

    # Add input information
    result["location"] = location
    result["date"] = str(date.date())
    result["time"] = data_input["time"]

    return result