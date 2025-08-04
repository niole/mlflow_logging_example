import os
from mlflow import MlflowClient
from typing import Optional, Callable, Any
import mlflow
import json
import time
import yaml
import logging
from datetime import datetime, timezone

client = MlflowClient()

def _do_evaluation(
        span,
        evaluator: Optional[Callable[[Any, Any], dict[str, Any]]] = None,
        is_production: bool = False) -> Optional[dict]:

        if not is_production and evaluator:
            return evaluator(span.inputs, span.outputs)
        return None

def read_ai_system_config(path: str = "./ai_system_config.yaml") -> dict:
    params = {}
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
    except Exception as e:
        # TODO make this quiet in production
        logging.warning("Failed to read ai system config yaml: ", e)
    return params

def _add_domino_tags(
        span,
        is_prod: bool = False,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None,
        is_eval: bool = False,
        sample: Optional[Any] = None,
    ):
    client.set_trace_tag(
        span.request_id,
        "domino.internal.aisystem.is_production",
        json.dumps(is_prod)
    )
    client.set_trace_tag(
        span.request_id,
        "domino.internal.is_eval",
        json.dumps(is_eval)
    )

    raw_sample = sample or [span.inputs, span.outputs]

    # TODO I want to get rid of these
    if not sample and extract_input_field:
        raw_sample = [extract_subfield(span.inputs, extract_input_field), raw_sample[1]]

    if not sample and extract_output_field:
        raw_sample = [raw_sample[0], extract_subfield(span.outputs, extract_output_field)]

    if is_eval:
        tag_sample = None
        if sample:
            tag_sample = json.dumps(sample)
        else:
            tag_sample = '|'.join([json.dumps(s) for s in raw_sample])

        # TODO validate that sample is < 5 kb https://mlflow.org/docs/latest/api_reference/rest-api.html#request-structure
        client.set_trace_tag(
            span.request_id,
            f"domino.internal.{span.name}.sample",
            tag_sample
        )


def extract_subfield(field: dict[str, Any], extract_field: str):
    inputs = field
    subpaths = extract_field.split('.')
    for path in subpaths:
        try:
            i = int(path)
            inputs = inputs[i]
            continue
        except:
            # it's not an index
            pass

        inputs = inputs[path]

    return inputs

def _is_production() -> bool:
    return os.getenv("DOMINO_EVALUATION_LOGGING_IS_PROD", "false") == "true"

def _get_prod_model_id() -> str:
    model_id = os.getenv("DOMINO_AI_SYSTEM_MODEL_ID", None)
    if not model_id:
        raise Exception("DOMINO_AI_SYSTEM_MODEL_ID environment variable must be set in production mode")

    return model_id

def _get_prod_logged_model():
    model_id = _get_prod_model_id()

    return mlflow.get_logged_model(model_id=model_id)

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
        model = _get_prod_logged_model()
        mlflow.set_active_model(model_id=model.model_id)
    else:
        # save configuration file for the AI System
        params = read_ai_system_config(ai_system_config_path)

        # if dev mode, we create a model, which represents the AI System
        # but traces will not be linked to it
        mlflow.create_external_model(
            model_type="AI System",
            params=params
        )

def start_domino_trace(
        name: str,
        evaluator: Optional[Callable[[Any, Any], dict[str, Any]]] = None,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None
    ):
    """A decorator that starts an mlflow trace for the function it decorates.
    It also enables the user to run an evaluation inline in the code is run in development mode on
    the inputs and outputs of the wrapped function call.
    The user can provide input and output formatters for formatting what's on the trace
    and the evaluation result inputs, which can be used by client's to extract relevant data when
    analyzing a trace.

    @start_domino_trace(
        name="assistant_chat_bot",
        evaluator=evaluate_helpfulness,
        extract_output_field="answer"
    )
    def ask_chat_bot(user_input: str) -> dict:
        ...

    Args:
        name: the name of the trace to create

        evaluator: an optional function that takes the inputs and outputs of the wrapped function and returns
        a dictionary of evaluation results. The evaluation result will be added to the trace as tags.

        extract_input_field: an optional dot separated string that specifies what subfield to access in the trace input

        extract_output_field: an optional dot separated string that specifies what subfield to access in the trace output

    Returns:
        A decorator that wraps the function to be traced.
    """
    def decorator(func):

        def wrapper(*args, **kwargs):
            is_production = _is_production()
            inputs = { 'args': args, 'kwargs': kwargs }

            parent_trace = client.start_trace(name, inputs=inputs)
            result = func(*args, **kwargs)
            client.end_trace(parent_trace.trace_id, outputs=result)

            # TODO error handling?
            trace = client.get_trace(parent_trace.trace_id).data.spans[0]

            eval_result = _do_evaluation(trace, evaluator, is_production)
            if eval_result:
                for (k, v) in eval_result.items():

                    # tag trace with the evaluation inputs, outputs, and result
                    # or maybe assessment?
                    domino_log_evaluation_data(
                        trace,
                        eval_result_label=k,
                        eval_result=v,
                        extract_input_field=extract_input_field,
                        extract_output_field=extract_output_field
                    )
            else:
                _add_domino_tags(trace, is_production, extract_input_field, extract_output_field, is_eval=False)

            return result

        return wrapper
    return decorator

def append_domino_span(
        name: str,
        evaluator: Optional[Callable[[Any, Any], dict[str, Any]]] = None,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None
    ):
    """A decorator that starts an mlflow span for the function it decorates. If there is an existing trace
    this span will be appended to it.

    It also enables the user to run an evaluation inline in the code is run in development mode on
    the inputs and outputs of the wrapped function call.
    The user can provide input and output formatters for formatting what's on the trace
    and the evaluation result inputs, which can be used by client's to extract relevant data when
    analyzing a trace.

    @append_domino_span(
        name="assistant_chat_bot",
        evaluator=evaluate_helpfulness,
        extract_output_field="answer"
    )
    def ask_chat_bot(user_input: str) -> dict:
        ...

    Args:
        name: the name of the trace to create

        evaluator: an optional function that takes the inputs and outputs of the wrapped function and returns
        a dictionary of evaluation results. The evaluation result will be added to the trace as tags.

        extract_input_field: an optional dot separated string that specifies what subfield to access in the trace input

        extract_output_field: an optional dot separated string that specifies what subfield to access in the trace output

    Returns:
        A decorator that wraps the function to be traced.
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            is_production = _is_production()
            with mlflow.start_span(name) as parent_span:
                inputs = { 'args': args, 'kwargs': kwargs }
                parent_span.set_inputs(inputs)
                result = func(*args, **kwargs)
                parent_span.set_outputs(result)

                eval_result = _do_evaluation(parent_span, evaluator, is_production)

                if eval_result:
                    for (k, v) in eval_result.items():
                        domino_log_evaluation_data(
                            parent_span,
                            eval_result=v,
                            eval_result_label=k,
                            extract_input_field=extract_input_field,
                            extract_output_field=extract_output_field,
                        )
                else:
                    _add_domino_tags(parent_span, is_production, extract_input_field, extract_output_field, is_eval=False)
                return result

        return wrapper
    return decorator

def domino_log_evaluation_data(
        span,
        eval_result,
        eval_result_label: Optional[str] = None,
        sample: Optional[Any] = None,
        extract_input_field: Optional[str] = None,
        extract_output_field: Optional[str] = None,
    ):
    """This logs evaluation data and metdata to a parent trace. This is used to log the evaluation of a span
    after it was created. This is useful for analyzing past performance of an AI System component.

    Args:
        span: the span to evaluate

        eval_result: optional, the evaluation result to log. This must be a primitive type in order to enable
        more powerful data analysis

        eval_result_label: an optional label for the evaluation result. This is used to identify the evaluation result
        sample: An optional sample representing what was evaluated. It must be JSON serializable. The sample will default to the inputs and outputs of the span.
        extract_input_field: an optional dot separated string that specifies what subfield to access in the trace input
        extract_output_field: an optional dot separated string that specifies what subfield to access in the trace output
    """

    is_production = _is_production()
    # TODO can only do this if the span is status = 'OK' or 'ERROR'
    if eval_result:
        label = eval_result_label or "evaluation_result"

        client.set_trace_tag(
            span.request_id,
            f"domino.prog.metric.{label}",
            json.dumps(eval_result),
        )
    _add_domino_tags(span, is_production, extract_input_field, extract_output_field, is_eval=eval_result is not None, sample=sample)

def log_summary_metric(evaluation_label: str, aggregation: Callable[[list], Any]):
    """Use this to log an aggregation metric for an evaluation at the end of a run or when
    doing analyziz in production

    TODO add summary window

    Args:
        evaluation_label: The label of the evaluation result that you returned from your evaluator
        aggregation: a funciton that aggregates a list of evaluation results. Should be a list of primitive values
    """
    label = f"domino.evaluation_result.{evaluation_label}"
    filter_string = f"tags.domino.evaluation_label.{evaluation_label} = 'true'"
    is_production = _is_production()
    traces = None
    if  is_production:
        # logs a summary metric to LoggedModel
        model = _get_prod_logged_model()
        model_id = model.model_id
        experiment_id = model.experiment_id

        traces = client.search_traces(
            experiment_ids=[experiment_id],
            model_id=model_id,
            filter_string=filter_string
        )

    else:
        # logs a summary metric to the run
        run = mlflow.active_run()
        experiment_id = run.info.experiment_id

        traces = client.search_traces(
            run_id = run.info.run_id,
            experiment_ids=[experiment_id],
            filter_string=filter_string
        )

    if traces:
        aggregate = aggregation([t.info.tags.get(label, None) for t in traces])
        mlflow.log_metric(label, aggregate)

class DominoEvaluation:
    def __init__(self, name: str, value: Any, sample: str, inputs: Any, outputs: Any, trace_id: str, span_name: str):
        self.name = name
        self.value = value
        self.sample = sample
        self.inputs = inputs
        self.outputs = outputs
        self.trace_id = trace_id
        self.span_name = span_name

"""
helps user find spans that they want to evaluate in a post-hoc fashion
By default returns spans in from the last day
TODO add run_id?
evaluation data is all on the trace, but the span could be a subset of the trace
"""
def find_spans(
        experiment_id: str,
        parent_trace_name: Optional[str] = None,
        span_names: Optional[list[str]] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        evaluations_only: bool = False,
    ) -> list:
    default_timestamp = int(time.time() * 1000)
    start_time = start_time_ms or (default_timestamp - (24*60*60*1000))
    end_time = end_time_ms or default_timestamp

    limit = 100
    filter_string = f"status = 'OK' AND trace.timestamp > {start_time} AND trace.timestamp < {end_time}"
    if parent_trace_name:
        filter_string += f" AND trace.name = '{parent_trace_name}'"

    if evaluations_only:
        filter_string += f" AND tags.domino.internal.is_eval = 'true'"

    traces = mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=limit,
        return_type="list",
    )

    all_span_names = span_names or [parent_trace_name] if parent_trace_name else []
    if len(all_span_names) > 0:
        spans = []
        for trace in traces:
            opt_spans =  [trace.search_spans(name=name) for name in all_span_names]
            spans.extend([item for sl in opt_spans for item in sl])
        return spans
    else:
        # get top span, which is the parent trace's span
        return [t.data.spans[0] for t in traces]
