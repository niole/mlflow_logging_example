import os
from random import randint
import mlflow
from contextlib import contextmanager
from mlflow import MlflowClient

# can't find exp if tracking uri not set in file
mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
client = MlflowClient()
mlflow.autolog()

"""
BEST

This creates a decorator that the end user decorates the code
for their evaluation calculation with. The decorator creates a trace for each
evaluation. The user can have as many eval traces per run as they want.

Their evaluation function must take an input and return their evaluation
metrics as an output, which is a dictionary, where the labels are the
metric names and the values are the metrics. Metric names must be unique
across traces. Each trace could output multiple metrics. There may be
multiple traces with different inputs.

e.g. run1 -> eval run input1 -> trace1 -> { "helpfullness": 1, "relevance": 0.8 }
                               trace2 -> { "correctness": 0.5 }

UI:
evaluation runs
run1
---------------------------------------------------------
input            | helpfulllness | relevance | correctness
---------------------------------------------------------
 <eval run input |     | 1             | 0.8       | 0.5

If the user turns on autolog for mlflow/their chosen framework, then downstream
traces may automatically be linked to domino eval trace.
"""
def domino_eval_trace(func):

    @mlflow.trace(name="domino_eval_trace")
    def wrapper(*args, **kwargs):
        # create group id
        group_id = "domino_eval_group_id" + str(randint(0, 1000))

        # tag parent trace
        parent_span = mlflow.get_current_active_span()
        parent_span.set_attributes({
            "domino_eval_group_id": group_id,
            "is_root_eval_span": True
        })

        result = func(*args, **kwargs)

        # tag child traces
        run = mlflow.active_run()
        traces = client.search_traces(
            run_id=run.info.run_id,
            experiment_ids=[run.info.experiment_id]
        )

        for trace in traces:
            spans = trace.data.spans
            for span in spans:
                client.set_trace_tag(span.request_id, "domino_eval_group_id", group_id)

        return result

    return wrapper

"""
A domino eval run links all traces executed inside the run to
a domino eval trace
the user is asked to log whatever eval metrics to the run
the domino eval run links one input to one output
input is expected to be a primitive data type, e.g. string, int, bool, etc.
input is not expected to be a complex data type, e.g. list, dict, etc.
"""

@contextmanager
@mlflow.trace(name="domino_eval_trace")
def domino_eval_run(input_v):
    run = mlflow.start_run()
    run_id = run.info.run_id
    mlflow.log_param("input", input_v)
    parent_trace = mlflow.get_last_active_trace_id()

    yield run

    traces = client.search_traces(
        run_id=run_id,
        experiment_ids=[run.info.experiment_id]
    )
    # set same session_id on each trace
    print(parent_trace)
    for trace in traces:
        spans = trace.data.spans
        print(spans)

    return "trace res"

def domino_eval_run_dec(func):

    def wrapper():
        run = mlflow.start_run()
        run_id = run.info.run_id

        root_span = client.start_trace(
            name="domino_eval_trace",
            inputs={},
            attributes={},
        )

        parent_trace = root_span.span_id

        func()

        client.end_trace(request_id=root_span.request_id)

        traces = client.search_traces(
            run_id=run_id,
            experiment_ids=[run.info.experiment_id]
        )
        # set same session_id on each trace
        print(parent_trace)
        for trace in traces:
            spans = trace.data.spans
            print(spans)

        return "trace res"

    return wrapper
