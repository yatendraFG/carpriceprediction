import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Streamlit UI
st.title("Car Price Predictor")
st.write("Fill in the details below to predict the price of a used car.")

# Dropdown options
companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

# User input fields
company = st.selectbox("Select Company", ['Select Company'] + companies)
car_model = st.selectbox("Select Car Model", car_models)
year = st.selectbox("Select Year", years)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500)

# Predict button
if st.button("Predict Price"):
    if company == "Select Company" or kms_driven == 0:
        st.error("Please select a valid company and enter kilometers driven.")
    else:
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                               data=np.array([car_model, company, year, kms_driven, fuel_type]).reshape(1, 5)))
        st.success(f"Estimated Car Price: ₹{np.round(prediction[0], 2)}")
