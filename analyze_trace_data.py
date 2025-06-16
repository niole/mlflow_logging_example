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

experiment_id = 30
print(f"get domino traces from experiment {experiment_id}")
ts = client.search_traces(experiment_ids=['30'], filter_string="trace.name = 'domino_eval_trace'")
for t in ts:
    for s in t.data.spans:
        if s.name == "domino_eval_trace":
            search_filter = f"compareRunsMode=TRACES&selectedTraceId={t.info.trace_id}"
            eval_trace_url = f"{tracking_uri}/#/experiments/{experiment_id}?{search_filter}"

            inputs = s.inputs['args']
            outputs = s.outputs

            print(inputs, outputs, eval_trace_url)
