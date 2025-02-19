import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify   # request: for http request, jsonify: convert request to json

# Load the trained model
import dagshub
dagshub.init(repo_owner='MariahFerns', repo_name='dagshub-demo', mlflow=True)
import mlflow

logged_model = 'runs:/d421656a008d44cb867552e5c3be6efe/Random Forest Classifier'

# Load model
model = mlflow.sklearn.load_model(logged_model)
print(type(model))


# Initialize Flask app
app = Flask(__name__)

# Define home route
@app.route('/')
def home():
    return 'Welcome to the Diabetes Prediction API! Use the /predict endpoint to make predictions'


@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)
    
    # Convert features to dataframe
    features = np.array(data['features'])
    features_df = pd.DataFrame([features], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Make prediction
    prediction = model.predict(features_df)
    
    # Map numberic prediction to class name
    outcome = {0:'Diabetes negative', 1:'Diabetes positive'}
    predicted_outcome = outcome[int(prediction[0])]

    # Send back the prediction as JSON
    return jsonify({"Prediction": int(prediction[0]), 'Outcome': predicted_outcome})


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
