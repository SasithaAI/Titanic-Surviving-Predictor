#!/usr/bin/env python
import streamlit as st
import pandas as pd
import pickle
import sys
import os

# Add the directory containing the scripts to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_preprocess_data
from model_training import train_survival_model
from prediction_function import predict_survival_probability

# Load the trained model and model features
try:
    with open('survival_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        model_features = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run model_training.py first.")
    st.stop()
import streamlit as st
import pandas as pd
import pickle
import sys
import os

# Add the directory containing the scripts to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_preprocess_data
from model_training import train_survival_model
from prediction_function import predict_survival_probability

# Load the trained model and model features
try:
    with open('survival_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        model_features = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run model_training.py first.")
    st.stop()

st.title('Titanic Survival Prediction')

st.write("""
This application predicts the survival probability of a passenger on the Titanic
based on their characteristics. Please enter the passenger's details below:
""")

# Create input fields for each predictor feature
input_data = {}
for feature in model_features:
    if feature == 'Pclass':
        input_data[feature] = st.selectbox('Pclass:', [1, 2, 3])
    elif feature == 'Age':
        input_data[feature] = st.slider('Age:', 0, 100, 30)
    elif feature == 'SibSp':
        input_data[feature] = st.slider('SibSp:', 0, 8, 0)
    elif feature == 'Parch':
        input_data[feature] = st.slider('Parch:', 0, 6, 0)
    elif feature == 'Fare':
        input_data[feature] = st.slider('Fare:', 0.0, 500.0, 50.0)
    elif feature == 'Sex_male':
        input_data[feature] = st.selectbox('Sex:', ['female', 'male'])
    elif feature == 'Embarked_Q':
        input_data[feature] = st.selectbox('Embarked_Q:', [False, True])
    elif feature == 'Embarked_S':
        input_data[feature] = st.selectbox('Embarked_S:', [False, True])

# Convert categorical inputs to the format expected by the model
if input_data['Sex_male'] == 'male':
    input_data['Sex_male'] = 1
else:
    input_data['Sex_male'] = 0

if input_data['Embarked_Q'] == True:
    input_data['Embarked_Q'] = 1
else:
    input_data['Embarked_Q'] = 0

if input_data['Embarked_S'] == True:
    input_data['Embarked_S'] = 1
else:
    input_data['Embarked_S'] = 0

if st.button('Predict Survival'):
    # Make prediction
    probability = predict_survival_probability(input_data)

    if probability is not None:
        st.write(f"Predicted survival probability: {probability:.2f}%")

if st.button('Predict Survival'):
    # Make prediction
    probability = predict_survival_probability(input_data)

    if probability is not None:
        st.write(f"Predicted survival probability: {probability:.2f}%")
