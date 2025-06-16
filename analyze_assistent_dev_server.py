import os
import mlflow
from mlflow import MlflowClient
import pandas as pd

# https://mlflow.org/docs/latest/genai/tracing/search-traces

tracking_uri = f"http://localhost:{os.environ['REV_PROXY_PORT']}"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

"""
analyzes outputs from the assistant dev server experiment
"""
def main():
    exps = mlflow.search_experiments(filter_string="name = 'assistant_dev_server'")
    if len(exps) == 0:
        raise Exception("assistant_dev_server experiment not found. Run the dev server first.")

    trace_df = pd.DataFrame({'inputs': [], 'outputs': [], 'url': [], 'evaluation_result':[], 'run_id':[] })
    for exp in exps:
        experiment_id = exp.experiment_id
        runs = client.search_runs(experiment_id)
        for run in runs:
            ts = client.search_traces(run_id=run.info.run_id, experiment_ids=[experiment_id], filter_string="trace.name = 'domino_eval_trace'")

            for t in ts:
                for s in t.data.spans:
                    if s.name == "domino_eval_trace":
                        search_filter = f"compareRunsMode=TRACES&selectedTraceId={t.info.trace_id}"
                        eval_trace_url = f"{tracking_uri}/#/experiments/{experiment_id}?{search_filter}"

                        inputs = ','.join(s.inputs['args'])

                        new_row = pd.DataFrame([{
                            'inputs': inputs,
                            'outputs': t.data.response,
                            'url': eval_trace_url,
                            'evaluation_result': t.info.tags.get('evaluation_result', None),
                            'run_id': run.info.run_id
                        }])
                        trace_df = pd.concat([trace_df, new_row], ignore_index=True)

        print(trace_df)
if __name__ == '__main__':
    main()

