import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Read data
df = pd.read_csv('diabetes.csv')


# Split data into X and y
X = df.drop('Outcome', axis=1)
y = df['Outcome']



# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train models
models = [
    {
        'name': 'Logistic Regression',
        'model': LogisticRegression,
        'params':{
            "solver": "liblinear",
            "C": 1
        }
        
    },
    {
        'name': 'Random Forest Classifier',
        'model': RandomForestClassifier,
        'params' : {
            'n_estimators': 100,
            'max_depth':3,
            'random_state':42
            }
    }
]

reports=[]
for m in models:
    model_name = m['name']
    model = m['model'](**m['params'])
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    report = classification_report(y_test, y_test_pred, output_dict=True)
    reports.append(report)




# Track using DagsHub
import dagshub
dagshub.init(repo_owner='MariahFerns', repo_name='dagshub-demo', mlflow=True) # collaborators can access this repo and experiments will be tracked under the same experiment name


# Track using MLFlow
import mlflow

# Set experiment name
mlflow.set_experiment('Diabetes Prediction')

for i, m in enumerate(models):
    model_name = m['name']
    model  = m['model']
    params = m['params']
    report = reports[i]

    with mlflow.start_run(run_name = model_name):
        # log params
        mlflow.log_param('model', model_name)
        mlflow.log_params(params)

        # log metrics
        mlflow.log_metric('accuracy', report['accuracy'])
        mlflow.log_metric('recall_0', report['0']['recall'])
        mlflow.log_metric('recall_1', report['1']['recall'])
        mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score'])
        
        # log model
        mlflow.sklearn.log_model(model, f'{model_name}')


# # Get list of experiments on local mlflow client (not on dagshub - comment dagshub code)
# from mlflow.tracking import MlflowClient

# client = MlflowClient()
# print(mlflow.search_experiments())


# # Filter runs within experiments and get artifacts using code instead of UI
# from mlflow.entities import ViewType

# runs = client.search_runs(
#     experiment_ids='574783491609994501',
#     filter_string = 'metrics.accuracy > 0.77',
#     run_view_type = ViewType.ACTIVE_ONLY,
#     max_results=5
# )

# print(runs)

