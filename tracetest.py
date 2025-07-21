import mlflow

mlflow.set_experiment("trace_exp")

@mlflow.trace(name="test_trace")
def f():
    return 1

with mlflow.start_run():
    f()

