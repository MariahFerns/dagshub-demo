import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import yaml

# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Read data
df = pd.read_csv(config['data']['raw_data_path'])

# Split data into X and y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=config['data']['test_size'], 
                                                    random_state=config['data']['random_state'])
print('Sample input data:\n',X_test[:1])

# Create a sample based on above
features = np.array([6,148,72,35,0,33.6,0.627,50])
features_df = pd.DataFrame([features], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
print(features_df)


# Test predicting on logged mlflow model
import dagshub
dagshub.init(repo_owner='MariahFerns', repo_name='dagshub-demo', mlflow=True)

import mlflow
logged_model = 'runs:/d421656a008d44cb867552e5c3be6efe/Random Forest Classifier'

# Load model
loaded_model = mlflow.sklearn.load_model(logged_model)
print(type(loaded_model))

# Make prediction
prediction = loaded_model.predict(features_df)
print('Prediction:\n', prediction)




