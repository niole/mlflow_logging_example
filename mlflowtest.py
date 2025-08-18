from mlflow import MlflowClient
import mlflow

client = MlflowClient()

experiment_name = "my_experiment"
mlflow.set_experiment(experiment_name)


def get_experiment_id() -> str:
    exps = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")
    if len(exps) == 0:
        raise Exception("experiment not found")
    return exps[0].experiment_id

# should only work from execution
@mlflow.trace(name="my_trace")
def do_thing():
    return 2

def test_logged_model():
    params = { 'cat': {'dog': 1 }}

    # should only work from execution
    logged_model = mlflow.create_external_model(
        model_type="AI System",
        params=params
    )
    mlflow.create_external_model(
        model_type="AI System",
        params=params
    )
    model_id = logged_model.model_id
    mlflow.get_logged_model(model_id=model_id)

    mlflow.search_logged_models(experiment_ids=[get_experiment_id()])
    mlflow.set_active_model(model_id=model_id)

with mlflow.start_run() as run:
    client.get_run(run.info.run_id)
    # create logged model
    #test_logged_model()

    # Log a parameter
    mlflow.log_param("param1", 5)

    # Log a metric
    mlflow.log_metric("metric1", 0.85)

    #do_thing()


experiment_id = get_experiment_id()

traces = client.search_traces(
    experiment_ids=[experiment_id],
    filter_string="trace.name = 'my_trace'",
)
for trace in traces:
    spans = trace.search_spans(name = "my_trace")
    span = spans[0]
    client.get_trace(span.request_id) # mlflow 2, works against mlflow 3 BE
    #client.get_trace_info(span.request_id) # mlflow 3
    client.set_trace_tag(
        span.request_id,
        "mytracetag",
        "true"
    )
    client.set_trace_tag(
        span.request_id,
        "deleteme",
        "true"
    )
    client.delete_trace_tag(
        span.request_id,
        "deleteme",
    )

# verification steps:

# verify logged model exists and has the right parameters

# verify logged model has tag

# verify that traces are attached to one of the logged models and not the other

# verify that trace named my_trace exists

# verify tags on my_trace

# verify that deleted tag doesn't exist on my_trace

# verify that search traces retrieves specified traces

# verify that traces can be started from execution

# verify that logged model can be created from execution

# verify that get trace, search traces, get logged model, tag trace, delete trace tag, search loggedmodels, can happen from outside execution
