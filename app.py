import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Read data
df = pd.read_csv('diabetes.csv')


# Split data into X and y
X = df.drop('Outcome', axis=1)
y = df['Outcome']


# Split the data into train and test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

