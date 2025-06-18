from mlflow import MlflowClient
from random import random
import mlflow
import os

# can't find exp if tracking uri not set in file
mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
client = MlflowClient()
mlflow.autolog()
mlflow.openai.autolog()

def domino_log_evaluation_data(span, eval_result, eval_result_label: str, is_prod: bool = False):
    client.set_trace_tag(
        span.request_id,
        "domino.evaluation_result",
        str(eval_result)
    )
    client.set_trace_tag(
        span.request_id,
        "domino.evaluation_result_label",
        eval_result_label
    )
    client.set_trace_tag(
        span.request_id,
        "domino.is_production",
        str(is_prod)
    )
    client.set_trace_tag(
        span.request_id,
        "domino.is_eval",
        str(True) # TODO this gets stringified as upper case "True"
    )


"""
This would be defined in a domino utility library

the end user uses this as a decorator on the AI system call, which they want to
trace and then evaluate the inputs/outputs of

if the app is running in production, the evaluation is not run inline, but the inputs and outputs are
instead saved somwehere. I don't know where yet: a dataset?
"""
def domino_eval_trace_2(evaluator):
    # trace_parent_id? w3c context propagation with mlflow

    def decorator(func):

        @mlflow.trace(name="domino_eval_trace")
        def do_trace(*args, **kwargs):
            return func(*args, **kwargs)


        def wrapper(*args, **kwargs):
            with mlflow.start_run() as run:
                inputs = { 'args': args, 'kwargs': kwargs }

                result = do_trace(*args, **kwargs)

                ts = client.search_traces(
                    run_id=run.info.run_id,
                    experiment_ids=[run.info.experiment_id],
                    filter_string="trace.name = 'domino_eval_trace'"
                )
                if len(ts) == 0:
                    print("no trace was found for the current run, not running the evaluator")
                    return result

                trace = ts[0].data.spans[0]

                is_production = os.getenv("PRODUCTION", "false") == "true"
                if is_production:
                    # TODO if prod mode, save the evaluation inputs and outputs:
                    # (experiment_id, run_id, inputs, result, eval_result)
                    # and user can use their evaluator utility library in a job
                    # where they would run it on that data
                    return result

                # if dev mode, run the evaluation inline
                eval_result = evaluator(trace.inputs, trace.outputs)

                # tag trace with the evaluation inputs, outputs, and result
                # or maybe assessment?
                eval_label = list(eval_result.keys())[0]
                eval_value = eval_result[eval_label]
                domino_log_evaluation_data(
                    trace,
                    eval_result_label=eval_label,
                    eval_result=eval_value,
                    is_prod=is_production
                )

                return result

        return wrapper
    return decorator

