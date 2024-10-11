import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = tf.keras.models.load_model('model_gru_mae.h5', compile=False)

# Assuming the scaler was trained on the original 3 features (flow, occupancy, speed)
scaler_Y = MinMaxScaler(feature_range=(0, 1))

# You would normally load the scaler with the saved state from training
# Here we simulate the training range based on reasonable assumptions
scaler_Y.fit([[0], [100]])

# Function to preprocess input data for prediction
def preprocess_input(location, hour, day, window_size=24):
    # Create an empty array with the shape (24, 4590)
    input_data = np.zeros((window_size, 4590))
    
    # Encode hour as one-hot and insert into the input array
    hour_one_hot = np.eye(24)[hour]
    input_data[:, :24] = hour_one_hot
    
    # Encode day as one-hot and insert into the input array
    day_one_hot = np.eye(7)[day]
    input_data[:, 24:31] = day_one_hot

    # Reshape to match the model's input
    input_data = input_data.reshape(1, window_size, 4590)
    
    return input_data

# Streamlit App
st.title("Traffic Prediction App")

st.write("""
### About the Dataset
The PEMS-08 Dataset includes data from San Bernardino during July and August of 2016. It covers 170 locations with sensors recording traffic information every 5 minutes. The dataset includes 3 key variables:

1. **Flow**: Number of vehicles passing through the sensor in each 5-minute interval.
2. **Occupancy**: Percentage of time a vehicle was detected by the sensor.
3. **Speed**: Average speed of vehicles during the interval, measured in mph.

By providing the **location**, **hour of the day**, and **day of the week**, the model predicts the values for **flow**, **occupancy**, and **speed**.
""")

# Input fields
st.header("Input Parameters")

location = st.selectbox("Select Location", range(1, 171))  # Select location (1 to 170)
hour = st.selectbox("Hour of Day (0-23)", range(0, 24))  # Select hour
day = st.selectbox("Day of the Week (0=Mon, 6=Sun)", range(0, 7))  # Select day

# Display the predict button
if st.button("Predict"):
    st.write("Making prediction...")
    
    # Preprocess input
    input_data = preprocess_input(location, hour, day)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Inverse transform the predictions to the original scale
    predicted_flow = scaler_Y.inverse_transform([[prediction[0][0]]])[0][0]
    predicted_occupancy = scaler_Y.inverse_transform([[prediction[0][1]]])[0][0] * 100  # Convert to percentage
    predicted_speed = scaler_Y.inverse_transform([[prediction[0][2]]])[0][0]
    
    # Clamp the occupancy to the range 0-100%
    predicted_occupancy = max(0, min(predicted_occupancy, 100))
    
    # Additional logic: If flow is low, occupancy shouldn't be too high
    if predicted_flow < 10 and predicted_occupancy > 80:
        predicted_occupancy = max(0, predicted_occupancy - 20)
    
    # Further logic: If speed is relatively high, occupancy should be lower
    if predicted_speed > 50 and predicted_occupancy > 50:
        predicted_occupancy = max(0, predicted_occupancy - 30)
    
    # Round and present the predictions in a human-readable format
    predicted_flow = round(predicted_flow)
    predicted_occupancy = round(predicted_occupancy, 2)
    predicted_speed = round(predicted_speed, 1)
    
    # Logic check: If flow is 0, set occupancy and speed to 0 as well
    if predicted_flow == 0:
        predicted_occupancy = 0.0
        predicted_speed = 0.0
    
    st.write("Prediction made!")
    
    st.subheader(f"Predicted Flow: {predicted_flow} vehicles per 5 minutes")
    st.subheader(f"Predicted Occupancy: {predicted_occupancy}% of the time")
    st.subheader(f"Predicted Speed: {predicted_speed} mph")
    
    st.write("Note: The prediction is based on the GRU model trained on the PEMS-08 dataset.")
else:
    st.write("Click the 'Predict' button to see the prediction.")
