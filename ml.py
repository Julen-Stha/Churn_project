#Gender 1 Female 0 Male
#Churn  1 yes 0 no
#scaler is exported as scaler.pkl
#model is exported as model.pkl
#order of x is 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np


scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")


st.title("Churn Predictionn")

st.divider()

st.write("Please enter values and hit the predict button to get prediction")

st.divider()

age = st.number_input("Enter age",min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure",min_value=0, max_value=130, value =10)

monthlycharge = st.number_input("Enter monethly charge",min_value=30,max_value=150)

gender = st.selectbox("Enter gender",["Male","Female"])

st.divider()



predictbutton = st.button("Predict")

st.divider()

if predictbutton:
    gender_selected = 1 if gender =="Female" else 0

    x=[age,gender_selected,tenure,monthlycharge]

    x1 = np.array(x)

    x_array = scaler.transform([x1])

    prediction = model.predict(x_array)[0]

    predicted = "Churn" if prediction ==1 else "Not Churn"

    st.write(f"Predicted: {predicted}")

else:
    st.write("Please enter the values and use predict button")