import os
import mlflow
from mlflow import MlflowClient
import pandas as pd
from production.domino_eval_trace import domino_log_evaluation_data

# https://mlflow.org/docs/latest/genai/tracing/search-traces

tracking_uri = f"http://localhost:{os.environ['REV_PROXY_PORT']}"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

def get_experiment_id() -> str:
    exps = mlflow.search_experiments(filter_string="name = 'assistant_dev_server_3'")
    if len(exps) == 0:
        raise Exception("assistant_dev_server experiment not found. Run the dev server first.")
    return exps[0].experiment_id

"""
add evaluation results to evaluation traces
"""
def log_eval_metrics_to_autologged_traces():
    experiment_id = get_experiment_id()

    # search for Completion spans
    # this doesn't actually return all spans, i guess just traces?
    completion_traces = mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string="trace.name = 'Completions'",
        return_type="list"
    )

    # log evaluation metrics to the Completion spans
    # if a user writes this evaluation code and then they change the code path to have multiple Completions
    # traces, open ai may modify the Completion trace names in the mlflow ui to make them unique
    # how will a user know what span name to look for if they don't look at the mlflow dashboard?
    # btw we don't expose the dashboard by default in the domino UI
    for trace in completion_traces:
        spans = trace.search_spans(name = "Completions")
        span = spans[0]
        domino_log_evaluation_data(
            span,
            eval_result_label="helpfulness",
            eval_result=1, # fake eval result
            extract_input_field="messages.1.content"
        )


"""
analyzes outputs from the assistant dev server experiment

The "Completions" evaluations were collected by evaluating traces created with autolog
after they were collected by calling  "domino_log_evaluation_data"

The "domino_eval_trace" evaluations were collected by using the domino decorator

The UI would have to call logic like this in order to get the trace data
per run
"""
def main():
    experiment_id = get_experiment_id()

    # get all traces that are domino eval traces in the experiment
    ts = client.search_traces(
        experiment_ids=[experiment_id],
        filter_string="tags.domino.is_eval = 'True'"
    )

    trace_df = pd.DataFrame({'span_name': [], 'inputs': [], 'outputs': [],  'evaluation_result_label': [], 'evaluation_result':[] })

    for t in ts:
        s = t.data.spans[0]
        search_filter = f"compareRunsMode=TRACES&selectedTraceId={t.info.trace_id}"
        eval_trace_url = f"{tracking_uri}/#/experiments/{experiment_id}?{search_filter}"
        extract_input_field = t.info.tags.get('domino.extract_input_field', None),

        inputs = s.inputs

        # The UI would write code like this to extract the subfields of an input or output
        # on a trace if "extract_input_field" is set
        if extract_input_field[0] is not None:
            subpaths = extract_input_field[0].split('.')
            for path in subpaths:
                try:
                    i = int(path)
                    inputs = inputs[i]
                    continue
                except:
                    # it's not an index
                    pass

                inputs = inputs[path]


        new_row = pd.DataFrame([{
            'span_name': s.name,
            'inputs': inputs,
            'outputs': t.data.response,
            'evaluation_result_label': t.info.tags.get('domino.evaluation_result_label', None),
            'evaluation_result': t.info.tags.get('domino.evaluation_result', None),
        }])
        trace_df = pd.concat([trace_df, new_row], ignore_index=True)

    print(trace_df.head())

if __name__ == '__main__':
    log_eval_metrics_to_autologged_traces()
    main()
