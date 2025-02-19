import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import yaml

# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Read data
# df = pd.read_csv('diabetes.csv')
df = pd.read_csv(config['data']['raw_data_path'])


# Split data into X and y
X = df.drop('Outcome', axis=1)
y = df['Outcome']



# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=config['data']['test_size'], 
                                                    random_state=config['data']['random_state'])


# Mapping model names from string to actual class
model_mapping = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier
}


# Train models
models = [
    {
        'name': config['model1']['name'],
        'model': model_mapping[config['model1']['type']],
        'params':{
            "solver": config['model1']['parameters']['solver'],
            "C": config['model1']['parameters']['C']
        }
        
    },
    {
        'name': config['model2']['name'],
        'model': model_mapping[config['model2']['type']],
        'params' : {
            'n_estimators': config['model2']['parameters']['n_estimators'],
            'max_depth':config['model2']['parameters']['max_depth'],
            'random_state':config['model2']['parameters']['random_state']
            }
    }
]

metrics = config['evaluation']['metrics']
per_class_metrics = config['evaluation']['per_class_metrics']
macro_avg_metrics = config['evaluation']['macro_avg_metrics']


# Track using DagsHub
import dagshub
dagshub.init(repo_owner='MariahFerns', repo_name='dagshub-demo', mlflow=True) # collaborators can access this repo and experiments will be tracked under the same experiment name


# Track using MLFlow
import mlflow

# Set experiment name
mlflow.set_experiment(config['app']['name'])


for i, m in enumerate(models):
    model_name = m['name']
    params = m['params']
    model = m['model'](**params)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    report = classification_report(y_test, y_test_pred, output_dict=True)
    

    
    with mlflow.start_run(run_name = model_name):
        # Log params
        mlflow.log_param('model', model_name)
        mlflow.log_params(params)

        # Log metrics
        # general metrics (e.g., accuracy)
        for metric in metrics:
            if metric in report:
                mlflow.log_metric(metric, report[metric])
        # per-class metrics (e.g., recall for each class)
        for class_label in report.keys():
            if class_label.isdigit():  # Ensures we log only class-specific metrics
                for metric in per_class_metrics:
                    if metric in report[class_label]:
                        mlflow.log_metric(f"{metric}_{class_label}", report[class_label][metric])  
        # macro avg metrics
        for metric in macro_avg_metrics:
            if metric in report['macro avg']:
                mlflow.log_metric(f'{metric}_macro', report['macro avg'][metric])


        # mlflow.log_metric('accuracy', report['accuracy'])
        # mlflow.log_metric('recall_0', report['0']['recall'])
        # mlflow.log_metric('recall_1', report['1']['recall'])
        # mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score'])
        
        # Log model
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

