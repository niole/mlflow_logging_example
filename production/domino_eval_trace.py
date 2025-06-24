import os
from mlflow import MlflowClient
from typing import Optional, Callable, Any
import mlflow
import json
import yaml
import logging

# can't find exp if tracking uri not set in file
mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
client = MlflowClient()

def init_domino_tracing(
        experiment_name: str,
        ai_frameworks: list[str] = list(),
        is_production: bool = False,
        ai_system_config_path: str = "./ai_system_config.yaml"):
    """Initialize code based Domino tracing for an AI System component.
    If in dev mode, it is expected that a run has been intialized. All traces will be linked to that run and a
    LoggedModel will be created, which represents the AI System component and will contain the configuration
    defined in the ai_system_config.yaml.

    If in prod mode, a DOMINO_AI_SYSTEM_MODEL_ID is required and represents the production AI System
    component. All traces will be linked to that model. No run is required.

    Args:
        experiment_name: the name of the Mlflow experiment to log traces to
        ai_frameworks: the ai frameworks to initialize autologging for, see https://mlflow.org/docs/latest/ml/tracking/autolog#supported-libraries
        is_production: whether or not this component is running in production mode
        ai_system_config_path: the path to the ai system configuration file
    """

    # set production environment variable
    os.environ["DOMINO_EVALUATION_LOGGING_IS_PROD"] = json.dumps(is_production)

    # initialize autologging
    mlflow.autolog()
    for fw in ai_frameworks:
        try:
            getattr(mlflow, fw).autolog()
        except Exception as e:
            logging.warning(f"Failed to call mlflow autolog for {fw} ai framework", e)

    mlflow.set_experiment(experiment_name)

    # NOTE: user provides the model name if they want to link traces to a
    # specific AI System. We will recommend this for production, but not for dev
    # evaluation
    if is_production:
        model_id = os.getenv("DOMINO_AI_SYSTEM_MODEL_ID", None)

        if not model_id:
            raise Exception("DOMINO_AI_SYSTEM_MODEL_ID environment variable must be set in production mode")

        model = mlflow.get_logged_model(model_id=model_id)
        mlflow.set_active_model(model_id=model.model_id)
    else:
        if not mlflow.active_run():
            raise Exception("No active run found. Please start a run when using Domino tracing in development mode.")

        # save configuration file for the AI System
        params = {}
        try:
            with open(ai_system_config_path, "r") as f:
                params = yaml.safe_load(f)
        except Exception as e:
            logging.warning("Failed to read ai system config yaml: ", e)

        # if dev mode, we create a model, which represents the AI System
        # but traces will not be linked to it
        mlflow.create_external_model(
            model_type="AI System",
            params=params
        )

"""
This would be defined in a domino utility library

the end user uses this as a decorator on the AI system call, which they want to
trace and then evaluate the inputs/outputs of

if the app is running in production, the evaluation is not run inline, but the inputs and outputs are
instead saved somwehere. I don't know where yet: a dataset?

user can provide input and output formatter for formatting what's on the trace
and the evaluation result inputs
"""
def start_domino_trace(
        name: str,
        evaluator: Optional[Callable[[Any, Any], Any]] = None,
        evaluation_label: Optional[str] = None,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None
    ):
    # trace_parent_id? w3c context propagation with mlflow

    def decorator(func):

        def wrapper(*args, **kwargs):
            is_production = _is_production()
            inputs = { 'args': args, 'kwargs': kwargs }

            parent_trace = client.start_trace(name, inputs=inputs)
            result = func(*args, **kwargs)
            client.end_trace(parent_trace.trace_id, outputs=result)

            # TODO error handling?
            trace = client.get_trace(parent_trace.trace_id).data.spans[0]

            eval_result = do_evaluation(trace, evaluator, is_production)
            if eval_result:
                for (k, v) in eval_result.items():

                    # tag trace with the evaluation inputs, outputs, and result
                    # or maybe assessment?
                    domino_log_evaluation_data(
                        trace,
                        eval_result_label=k,
                        eval_result=v,
                        is_production=is_production,
                        extract_input_field=extract_input_field,
                        extract_output_field=extract_output_field
                    )
            else:
                _add_domino_tags(trace, is_production, extract_input_field, extract_output_field, is_eval=False)

            return result

        return wrapper
    return decorator

"""
This decorator lets users add a named span in the middle of a trace
This is meant to add custom spans to autologged traces
This will only add spans if there is an active trace

we might want data from multiple spans in one evaluation, multiple arguments to evaluator
"""
def append_domino_span(
        name: str,
        evaluation_result_label: Optional[str] = None,
        evaluator: Optional[Callable[[Any, Any], Any]] = None,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None
    ):
    def decorator(func):

        def wrapper(*args, **kwargs):
            is_production = _is_production()
            with mlflow.start_span(name) as parent_span:
                inputs = { 'args': args, 'kwargs': kwargs }
                parent_span.set_inputs(inputs)
                result = func(*args, **kwargs)
                parent_span.set_outputs(result)

                eval_result = do_evaluation(parent_span, evaluator, is_production)

                if eval_result:
                    for (k, v) in eval_result.items():
                        domino_log_evaluation_data(
                            parent_span,
                            eval_result=v,
                            eval_result_label=k,
                            is_production=is_production,
                            extract_input_field=extract_input_field,
                            extract_output_field=extract_output_field,
                        )
                else:
                    _add_domino_tags(parent_span, is_production, extract_input_field, extract_output_field, is_eval=False)
                return result

        return wrapper
    return decorator

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
        eval_result_label: Optional[str] = None,
        is_production: bool = False,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None
    ):
    # can only do this if the span is status = 'OK' or 'ERROR'
    if eval_result:
        label = eval_result_label or "evaluation_result"

        client.set_trace_tag(
            span.request_id,
            f"domino.evaluation_result",
            eval_result
        )
        client.set_trace_tag(
            span.request_id,
            "domino.evaluation_result_label",
            label
        )
    _add_domino_tags(span, is_production, extract_input_field, extract_output_field, is_eval=eval_result is not None)

def _add_domino_tags(
        span,
        is_prod: bool = False,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None,
        is_eval: bool = False
    ):
    client.set_trace_tag(
        span.request_id,
        "domino.is_production",
        str(is_prod)
    )
    client.set_trace_tag(
        span.request_id,
        "domino.is_eval",
        json.dumps(is_eval)
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

def do_evaluation(
        span,
        evaluator: Optional[Callable[[Any, Any], Any]],
        is_production: bool) -> Optional[dict]:

        if not is_production and evaluator:
            return evaluator(span.inputs, span.outputs)
        return None

def _is_production() -> bool:
    return os.getenv("DOMINO_EVALUATION_LOGGING_IS_PROD", "false") == "true"
