import numpy as np
import pandas as pd
import joblib

loaded_kmeans = joblib.load('submission/kmeans_model.joblib')

def convert_to_attrition(cluster):
    if cluster in [0, 1]:
        return 1
    else:
        return 0

def predict_attrition(data):
    predicted_cluster = loaded_kmeans.predict(data)

    predicted_attrition = convert_to_attrition(predicted_cluster[0])

    return predicted_attrition

input_data = pd.DataFrame({
    'EmployeeId': [50],
    'Age': [45],
    'Gender': ['Male'],
    'Attrition': [0]
})

input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0})

features = ['EmployeeId', 'Age', 'Gender', 'Attrition']

X = input_data[features]

predicted_attrition = predict_attrition(X)

print("cluster untuk EmployeeId {}: {}".format(input_data['EmployeeId'][0], predicted_attrition))
