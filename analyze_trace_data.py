import os
import mlflow
from mlflow import MlflowClient
import click
from pprint import pprint

# https://mlflow.org/docs/latest/genai/tracing/search-traces

tracking_uri = f"http://localhost:{os.environ['REV_PROXY_PORT']}"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

"""
show that we can render the eval inputs and outputs and link to traces in the mlflow UI
"""
@click.command()
@click.option('--runid', help='mlflow run id', default="bee97d31ff21499d96813dedc7d40363")
def main(runid):
    run_ids = [runid]
    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        experiment_id = run.info.experiment_id
        ts = client.search_traces(run_id=run_id, experiment_ids=[experiment_id], filter_string="trace.name = 'domino_eval_trace'")

        # group traces by input
        grouped_traces = dict()
        for t in ts:
            for s in t.data.spans:
                if s.name == "domino_eval_trace":
                    search_filter = f"compareRunsMode=TRACES&selectedTraceId={t.info.trace_id}"
                    eval_trace_url = f"{tracking_uri}/#/experiments/{experiment_id}?{search_filter}"

                    inputs = ','.join(s.inputs['args'])
                    grouped_traces[inputs] = grouped_traces.get(inputs, [])
                    grouped_traces[inputs].append({ "outputs": s.outputs, "url": eval_trace_url})

        print("Grouped traces")
        pprint(grouped_traces)


        print()
        print("mlflow traces as a dataframe")
        print()
        mlflow_ts = mlflow.search_traces(run_id=run_id, experiment_ids=[experiment_id], filter_string="trace.name = 'domino_eval_trace'")
        # input, output cols, eval url, trace id
        print(mlflow_ts.columns)
        df = mlflow_ts[['trace_id', 'request', 'response']]
        print(df.head())


if __name__ == '__main__':
    main()

