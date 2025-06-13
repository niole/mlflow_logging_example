import os
import time
import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow import MlflowClient

mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
mlflow.set_experiment("eval_" + str(np.random.randint(0, 1000)))
mlflow.autolog()

# Get the MLflow client
client = MlflowClient()

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# NOTE: downloading and training this model creates its own run
# I guess if you do something outside of an explicit start run call,
# mlflow will start a run for you
# Train a model (we'll use this in our function)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Define a prediction function
# traces are linked to requests, and runs. and requests have inputs and outputs
# how to link to an evaluation metric?
@mlflow.trace(name="custom_predict_function")
def predict_function(input_data):
    """Custom prediction function that can include business logic."""
    print("input_data")
    print(input_data)

    # Get base model predictions
    base_predictions = model.predict(input_data)

    # Add custom business logic
    # Example: Override predictions for specific conditions
    feature_sum = input_data.sum(axis=1)
    high_feature_mask = feature_sum > feature_sum.quantile(0.9)

    # Custom rule: high feature sum values are always class 1
    final_predictions = base_predictions.copy()
    final_predictions[high_feature_mask] = 1

    return final_predictions


# Create evaluation dataset
eval_data = pd.DataFrame(X_test)
eval_data["target"] = y_test

with mlflow.start_run() as run:
    # Evaluate function directly - no model logging needed!
    result = mlflow.evaluate(
        predict_function,  # Function to evaluate
        eval_data,  # Evaluation data
        targets="target",  # Target column
        model_type="classifier",  # Task type
    )

    print()
    print(result.artifacts)
    print(result.metrics)
    print(result.tables)
    run_id = run.info.run_id
    #mlflow.log_table(result.tables)

    #time.sleep(10)
# this logic fails to find the experiment-id
    traces = client.search_traces(run_id=run_id, experiment_ids=[run.info.experiment_id])
    for trace in traces:
        print(trace.info)
        print(trace.data)

    mlflow.log_metrics(result.metrics)

    # fake eval table
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")
