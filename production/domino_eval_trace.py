from mlflow import MlflowClient
from random import random
from typing import Optional
import mlflow
import os

# can't find exp if tracking uri not set in file
mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
client = MlflowClient()

"""
This decorator lets users add a named span in the middle of a trace
This is meant to add custom spans to autologged traces
This will only add spans if there is an active trace

TODO maybe add optional inline evaluators?
"""
def append_domino_span(span_name: str, is_prod: bool = False, extract_input_field: Optional[str] = None, extract_output_field: Optional[str] = None):
    def decorator(func):

        def wrapper(*args, **kwargs):
            with mlflow.start_span(span_name) as parent_span:
                inputs = { 'args': args, 'kwargs': kwargs }
                parent_span.set_inputs(inputs)
                result = func(*args, **kwargs)
                parent_span.set_outputs(result)

                _add_domino_tags(parent_span, is_prod, extract_input_field, extract_output_field)

                return result

        return wrapper
    return decorator

def init_domino_tracing(experiment_name: str):
    mlflow.openai.autolog()
    mlflow.langchain.autolog()
    mlflow.autolog()

    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")

    # NOTE: user provides the model name if they want to link traces to a
    # specific AI System. We will recommend this for production, but not for dev
    # evaluation
    model_name = os.getenv("DOMINO_AI_SYSTEM_MODEL_NAME", None)

    model = mlflow.create_external_model(
        name=model_name,
        model_type="AI System",
        params={} # TODO configuration will likely be the config.yaml contents
    )

    mlflow.set_active_model(model_id=model.model_id)


"""
This logs evaluation data and metdata to a span. These spans can be used to
evaluate the AI System's performance

extract_input_field: an optional dot separated string that specifies what subfield to access in the input
when it is rendered in the Domino UI
extract_output_field: an optional dot separated string that specifies what subfield to access in the output
when it is rendered in the Domino UI, e.g. "messages.1.content"
"""
def domino_log_evaluation_data(
        span,
        eval_result,
        eval_result_label: str,
        is_prod: bool = False,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None
    ):
    # can only do this if the span is status = 'OK' or 'ERROR'
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
    _add_domino_tags(span, is_prod, extract_input_field, extract_output_field)

def _add_domino_tags(
        span,
        is_prod: bool = False,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None
    ):
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

    if extract_input_field:
        client.set_trace_tag(
            span.request_id,
            "domino.extract_input_field",
            extract_input_field
        )

    if extract_output_field:
        client.set_trace_tag(
            span.request_id,
            "domino.extract_output_field",
            extract_output_field
        )

"""
This would be defined in a domino utility library

the end user uses this as a decorator on the AI system call, which they want to
trace and then evaluate the inputs/outputs of

if the app is running in production, the evaluation is not run inline, but the inputs and outputs are
instead saved somwehere. I don't know where yet: a dataset?

user can provide input and output formatter for formatting what's on the trace
and the evaluation result inputs

TODO maybe make inline evaluators optional?
"""
def domino_eval_trace_2(evaluator, input_formatter = lambda x: x, output_formatter = lambda x: x):
    # trace_parent_id? w3c context propagation with mlflow

    def decorator(func):

        def wrapper(*args, **kwargs):
            inputs = { 'args': args, 'kwargs': kwargs }

            parent_trace = client.start_trace("domino_eval_trace", inputs=input_formatter(inputs))
            result = func(*args, **kwargs)
            client.end_trace(parent_trace.trace_id, outputs=output_formatter(result))

            # TODO error handling?
            trace = client.get_trace(parent_trace.trace_id).data.spans[0]

            # if dev mode, run the evaluation inline
            eval_result = evaluator(trace.inputs, trace.outputs)

            # TODO can we make assumptions about proudction env var?
            is_production = os.getenv("PRODUCTION", "false") == "true"

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

