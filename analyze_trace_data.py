import os
import mlflow
from mlflow import MlflowClient

# https://mlflow.org/docs/latest/genai/tracing/search-traces

tracking_uri = f"http://localhost:{os.environ['REV_PROXY_PORT']}"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

"""
show that we can render the eval inputs and outputs and link to traces in the mlflow UI
"""

run_ids = ["a8e41f93bf1c47cc8175f68c8c6d2637"]
for run_id in run_ids:
    run = mlflow.get_run(run_id)
    experiment_id = run.info.experiment_id
    ts = client.search_traces(run_id=run_id, experiment_ids=[experiment_id], filter_string="trace.name = 'domino_eval_trace'")
    metrics = dict()
    for t in ts:
        for s in t.data.spans:
            if s.name == "domino_eval_trace":
                search_filter = f"compareRunsMode=TRACES&selectedTraceId={t.info.trace_id}"
                eval_trace_url = f"{tracking_uri}/#/experiments/{experiment_id}?{search_filter}"

                inputs = s.inputs['args']
                outputs = s.outputs

                # outputs must always be a dict, so we know what to call each metric
                print(inputs, outputs, eval_trace_url)

    mlflow_ts = mlflow.search_traces(run_id=run_id, experiment_ids=[experiment_id], filter_string="trace.name = 'domino_eval_trace'")
    print(mlflow_ts)

