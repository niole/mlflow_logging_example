import mlflow
from random import random, randint
from mlflow.entities import SpanType
from openai import OpenAI
import os
from  domino_eval_trace import start_domino_trace, append_domino_span, read_ai_system_config
import evaluators
from langchain.chat_models import init_chat_model
from tools import tools, tools_table
from rag import query_docs

# must invole autolog in the file with the functions that you want to trace
mlflow.openai.autolog()
mlflow.langchain.autolog()
mlflow.autolog()

ai_system_config = read_ai_system_config("./production/ai_system_config.yaml")

client = OpenAI()
llm = init_chat_model(
    ai_system_config["llm"]["tool_model"],
    model_provider="openai"
)
llm_with_tools = llm.bind_tools(tools, tool_choice="any")

@append_domino_span("rag_response", evaluator=evaluators.question_fullfillment_evaluator)
def answer_question_with_context(question: str) -> str:
    """
        users asks questions and this function should be able to answer anything
        in a conversational way with good context.
    """
    question_context = query_docs(question)
    system_content = f"Please answer the question. Here is some context that may be helpful in answering the question: {question_context}"
    messages = [
        {"role": "system", "content": system_content },
        {"role": "user", "content": question }
    ]
    # openai autolog example
    # Inputs and outputs of the API request will be logged in a trace
    response = client.chat.completions.create(model=ai_system_config["llm"]["chat_model"], messages=messages)

    return response.choices[0].message.content or "Sorry, I couldn't answer that question"


@start_domino_trace("domino_eval_trace", evaluator=evaluators.assistant_evaluator)
def ask_assistant(question: str) -> str:
    # is very unhelpful half of the time
    content = llm_with_tools.invoke(question)
    tool_call = content.tool_calls[0]

    # NOTE: I messed up the tool call definition earilier
    # and used the tracing to figure out the bug
    return tools_table.get(tool_call["name"], lambda x: "I couldn't help with that").invoke(tool_call["args"])
