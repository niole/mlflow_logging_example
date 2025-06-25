import os
import mlflow
from mlflow import MlflowClient
import pandas as pd
from datetime import datetime, timezone, timedelta

client = MlflowClient()

def _get_experiment_id(name: str) -> str:
    exps = mlflow.search_experiments(filter_string=f"name = '{name}'")
    if len(exps) == 0:
        raise Exception(f"{name} experiment not found. Run the dev server first.")
    return exps[0].experiment_id

def build_dataset():
    """
    This function would be run in a scheduled job by the user.
    builds an evaluation dataset csv
    from historical question and answers found in the
    "rag_response" trace
    """
    experiment_name = "all_knowing_rag_agent_analysis"
    experiment_id = _get_experiment_id(experiment_name)
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)

    # NOTE: user could use the best practice of interpolating extraction start time timestamp or use their own
    # timewindow (past day)
    extraction_start_ts = datetime.fromisoformat(os.getenv("DOMINO_EVAL_EXTRACT_START_TS", str(yesterday))).timestamp() * 1000
    print(extraction_start_ts)

    # NOTE: recommended best practice of setting eval dataset name env var. User could just hardcode the
    # dataset name in their script
    evaluation_ds_name = os.getenv("DOMINO_EVAL_DATASET_NAME", None)

    if not evaluation_ds_name:
        raise Exception("Must set DOMINO_EVAL_DATASET_NAME in order to build the production eval dataset")

    completion_traces = mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string=f"trace.name = 'rag_response' AND attributes.timestamp > {extraction_start_ts}",
        #filter_string=f"trace.name = 'rag_response'",
        return_type="list"
    )

    # create a datafrom for this slice of the traces and save as an additional file
    # in the dataset
    df = pd.DataFrame({
        'name': [],
        'inputs': [],
        'outputs': [],
        'evaluation_score': [], # we may add or overwrite in our evaluation post processing job
    })

    for trace in completion_traces:
        spans = trace.search_spans(name="rag_response")
        if len(spans) == 0:
            continue

        span = spans[0]
        # TODO all tags
        row = pd.DataFrame([{
            'name': span.name,
            'inputs': span.inputs,
            'outputs': span.outputs,
            'evaluation_result': trace.info.tags.get('domino.evaluation_result', None),
        }])
        df = pd.concat([df, row], ignore_index=True)


    # TODO what if there are a lot of traces? How do we expect users to work with the datset files?
    # do we expect them to load into a database?

    save_csv_to_domino_dataset(str(extraction_start_ts), df)


def save_csv_to_domino_dataset(df_name: str, df: pd.DataFrame):
    # TODO write to domino dataset scratch space
    evaluation_ds_name = os.environ["DOMINO_EVAL_DATASET_NAME"]
    print(df.head())


if __name__ == "__main__":
    """
    how to call from command line on macos, where tracking server is local
    if running on domino, use -u argument for UTC and date command:

     MLFLOW_TRACKING_URI="http://localhost:4040" \
     DOMINO_EVAL_DATASET_NAME="myds" \
     DOMINO_EVAL_EXTRACT_START_TS=$(gdate -d "2 hours ago" "+%Y-%m-%d %H:%M:%S") \
     uv run production/build_production_evaluation_dataset.py
    """
    build_dataset()
