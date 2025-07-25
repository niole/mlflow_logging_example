import os
import re
import mlflow
from mlflow import MlflowClient
import pandas as pd
from production.domino_eval_trace import domino_log_evaluation_data, find_spans
import time

# https://mlflow.org/docs/latest/genai/tracing/search-traces

client = MlflowClient()

def get_experiment_id() -> str:
    exps = mlflow.search_experiments(filter_string="name = 'all_knowing_rag_agent_analysis2'")
    if len(exps) == 0:
        raise Exception("assistant_dev_server experiment not found. Run the dev server first.")
    return exps[0].experiment_id

"""
add evaluation results to evaluation traces
"""
def log_eval_metrics_to_autologged_traces():
    experiment_id = get_experiment_id()

    spans = find_spans(
            experiment_id,
            parent_trace_name="rag_response", span_names=["Completions_2", "Completions_1"])

    # log evaluation metrics to the Completion spans
    # if a user writes this evaluation code and then they change the code path to have multiple Completions
    # traces, open ai may modify the Completion trace names in the mlflow ui to make them unique
    # how will a user know what span name to look for if they don't look at the mlflow dashboard?
    # btw we don't expose the dashboard by default in the domino UI
    for span in spans:
        sample = span.inputs['messages'][0]
        domino_log_evaluation_data(
            span,
            sample=sample,
            eval_result_label="helpfulness",
            eval_result=1, # fake eval result
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
        filter_string="tag.domino.internal.is_eval = 'true'"
    )

    evaluation_labels = set()
    p = "domino.evaluation.result.(.*)$"

    for t in ts:
        for k, _ in t.info.tags.items():
            m = re.search(p, k)
            if m:
                evaluation_labels.add(m.group(1))

    df_content = {'span_name': [], 'inputs': [], 'outputs': [],  'evaluation_result_label': [], 'evaluation_result':[] }
    print(evaluation_labels)
    for l in evaluation_labels:
        df_content[l] = []
    trace_df = pd.DataFrame(df_content)

    for t in ts:
        s = t.data.spans[0]
#        eval_trace_url = f"{tracking_uri}/#/experiments/{experiment_id}?{search_filter}"
        extract_input_field = t.info.tags.get('domino.internal.extract_input_field', None),

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
            'outputs': s.outputs,
            'evaluation_result_label': t.info.tags.get('domino.internal.evaluation_result_label', None),
            'evaluation_result': t.info.tags.get('domino.evaluation_result', None),
        }])
        trace_df = pd.concat([trace_df, new_row], ignore_index=True)

    print(trace_df.head())

if __name__ == '__main__':
    log_eval_metrics_to_autologged_traces()
    #main()
