import os
from random import random, randint
import time
from mlflow import log_metric, log_param, log_artifacts
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
mlflow.set_experiment("mlflow_default_w_sklearn_"+	str(randint(0, 1000)))
mlflow.enable_system_metrics_logging()
mlflow.autolog()

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
