import streamlit as st
import numpy as np
import pickle

# Load the pre-trained classifier model
with open('classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the encoded fertilizer classes
with open('fertilizer.pkl', 'rb') as file:
    encoded_classes = pickle.load(file)

# Streamlit app
st.title("Fertilizer Prediction App")

# Collect user input
st.header("Enter the following details:")
temperature = st.number_input("Temperature (°C)", min_value=25.0, max_value=38.0, step=0.1, value=30.0)
humidity = st.number_input("Humidity (%)", min_value=50.0, max_value=72.0, step=0.1, value=60.0)
moisture = st.number_input("Moisture (%)", min_value=25.0, max_value=65.0, step=0.1, value=41.0)
soil_type = st.selectbox("Soil Type", ["Loamy", "Sandy", "Black", "Red", "Clayey"])
crop_type = st.selectbox("Crop Type", ["Sugarcane", "Wheat", "Rice", "Maize", "Cotton", "Barley", "Soybean", "Peas", "Sunflower", "Groundnut", "Mustard"])
nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=4.0, max_value=42.0, step=0.1, value=13.0)
potassium = st.number_input("Potassium (mg/kg)", min_value=0.0, max_value=19.0, step=0.1, value=0.0)
phosphorous = st.number_input("Phosphorous (mg/kg)", min_value=0.0, max_value=42.0, step=0.1, value=19.0)

# Map categorical variables to numerical values (this mapping should match your training data)
soil_type_mapping = {"Loamy": 1, "Sandy": 0, "Black": 2, "Red": 3, "Clayey": 4}
crop_type_mapping = {"Sugarcane": 0, "Wheat": 1, "Rice": 2, "Maize": 3, "Cotton": 4, "Barley": 5, "Soybean": 6, "Peas": 7, "Sunflower": 8, "Groundnut": 9, "Mustard": 10}

soil_type_num = soil_type_mapping[soil_type]
crop_type_num = crop_type_mapping[crop_type]

# Create input feature array
input_features = np.array([[temperature, humidity, moisture, soil_type_num, crop_type_num, nitrogen, potassium, phosphorous]])

# Predict fertilizer type
if st.button("Predict Fertilizer"):
    prediction = classifier.predict(input_features)
    predicted_class = encoded_classes.classes_[prediction[0]]
    st.write(f"The recommended fertilizer class is: {predicted_class}")

# For debugging purposes (optional, you can remove this)
st.write("Input features:")
st.write(input_features)


