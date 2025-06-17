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

The UI would have to call logic like this in order to get the trace data
per run
"""
def main():
    exps = mlflow.search_experiments(filter_string="name = 'assistant_dev_server_2'")
    if len(exps) == 0:
        raise Exception("assistant_dev_server experiment not found. Run the dev server first.")

    trace_df = pd.DataFrame({'inputs': [], 'outputs': [], 'url': [], 'evaluation_result_label': [], 'evaluation_result':[], 'run_id':[] })
    exp = exps[0]
    experiment_id = exp.experiment_id
    ts = client.search_traces(
        experiment_ids=[experiment_id],
        #filter_string="trace.name = 'domino_eval_trace' AND tag.evaluation_result_label = 'helpfulness'"
        filter_string="tag.evaluation_result_label = 'helpfulness'",
        #order_by=["timestamp_ms DESC", "tag.evaluation_result DESC"]
        order_by=["tag.evaluation_result DESC"]
    )

    for t in ts:
        s = t.data.spans[0]
        search_filter = f"compareRunsMode=TRACES&selectedTraceId={t.info.trace_id}"
        eval_trace_url = f"{tracking_uri}/#/experiments/{experiment_id}?{search_filter}"

        inputs = ','.join(s.inputs['args'])

        new_row = pd.DataFrame([{
            'inputs': inputs,
            'outputs': t.data.response,
            'url': eval_trace_url,
            'evaluation_result_label': t.info.tags.get('evaluation_result_label', None),
            'evaluation_result': t.info.tags.get('evaluation_result', None),
            'run_id': t.info.tags.get('run_id', None),
        }])
        trace_df = pd.concat([trace_df, new_row], ignore_index=True)

        print()
        print("Data for samples tab in comparisons view")
        print(trace_df)
        print()
        print("data for experiment top level view")
        # query traces by run_id
        grouped_by_run = trace_df.groupby('run_id')
        eval_result_labels = trace_df['evaluation_result_label'].unique()


if __name__ == '__main__':
    main()

