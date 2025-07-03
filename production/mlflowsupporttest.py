from mlflow import MlflowClient
import mlflow

client = MlflowClient()

experiment_name = "my_experiment"
mlflow.set_experiment(experiment_name)

@mlflow.trace(name="my_trace")
def do_thing():
    return 2

def test_logged_model():
    logged_model = mlflow.create_external_model(
        model_type="AI System",
        params=params
    )
    model_id = logged_model.model_id
    mlflow.get_logged_model(model_id=model_id)

with mlflow.start_run() as run:
    # create logged model
    #  this is only mlflow 3.0, feel free to comment out
    #test_logged_model()

    # Log a parameter
    mlflow.log_param("param1", 5)

    # Log a metric
    mlflow.log_metric("metric1", 0.85)

    do_thing()

def get_experiment_id() -> str:
    exps = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")
    if len(exps) == 0:
        raise Exception("experiment not found")
    return exps[0].experiment_id

experiment_id = get_experiment_id()

traces = client.search_traces(
    experiment_ids=[experiment_id],
    filter_string="trace.name = 'my_trace'",
   # return_type="list" # this is 3.0 only
)
for trace in completion_traces:
    spans = trace.search_spans(name = "my_trace")
    span = spans[0]
    client.set_trace_tag(
        span.request_id,
        "mytracetag",
        "true"
    )
