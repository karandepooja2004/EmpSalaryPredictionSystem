#Create a GUI using streamlit
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Define the input form
st.title("Employee Salary Prediction")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")
st.write("Fill in the employee details:")

# User input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
fnlwgt = st.number_input("FNLWGT", min_value=10000, max_value=1000000, value=200000)
education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)
marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                                         'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                                         'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'England'])

# Button to make prediction
if st.button("Predict Salary"):
    # Prepare input for prediction
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

    # Use same encoders or preprocessing as in training (you must replace this with actual code if you saved encoders)
    for col in input_data.select_dtypes(include='object').columns:
        input_data[col] = input_data[col].astype('category').cat.codes

    # Predict
    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Salary Category: {result}")