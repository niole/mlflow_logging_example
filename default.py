import os
from random import random, randint
import time
from mlflow import log_metric, log_param, log_artifacts, MlflowClient
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.set_experiment("mlflow_default_w_sklearn_"+	str(randint(0, 1000)))
mlflow.enable_system_metrics_logging()
mlflow.autolog()

client = MlflowClient()

def main():
    print("Hello from mlflow-test!")
    #mlflow.current_run(log_system_metrics=True)
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")

    # custom trace example
    root_span = client.start_trace("my_trace")

    # The request_id is used for creating additional spans that have a hierarchical association to this root span
    request_id = root_span.request_id
    child_span = client.start_span(
	name="child_span",
	request_id=request_id,
	parent_id=root_span.span_id,
	inputs={"input_key": "input_value"},
	attributes={"attribute_key": "attribute_value"},
    )

    client.end_span(
	request_id=child_span.request_id,
	span_id=child_span.span_id,
	outputs={"output_key": "output_value"},
	attributes={"custom_attribute": "value"},
    )

    # End the root span (trace)
    client.end_trace(
	request_id=request_id,
	outputs={"final_output_key": "final_output_value"},
	attributes={"token_usage": "1174"},
    )


    # default autolog example
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators = 100, max_depth = 6, max_features = 3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    autolog_run = mlflow.last_active_run()

    print("waiting 12 seconds for system metrics to be logged...")
    time.sleep(15)



if __name__ == "__main__":
    main()
