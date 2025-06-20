import mlflow
from random import random, randint
from mlflow.entities import SpanType
from openai import OpenAI
import os
from  domino_eval_trace import domino_eval_trace_2
import evaluators
from langchain.chat_models import init_chat_model
from tools import tools, tools_table

# must invole autolog in the file with the functions that you want to trace
mlflow.openai.autolog()
mlflow.langchain.autolog()
mlflow.autolog()

client = OpenAI()
llm = init_chat_model(
    "gpt-3.5-turbo",
    model_provider="openai"
)
llm_with_tools = llm.bind_tools(tools, tool_choice="any")


@domino_eval_trace_2(evaluator=evaluators.assistant_evaluator, input_formatter=lambda x: x['args'][0])
def ask_assistant(question: str) -> str:
    # is very unhelpful half of the time
    content = llm_with_tools.invoke(question)
    tool_call = content.tool_calls[0]

    # NOTE: I messed up the tool call definition earilier
    # and used the tracing to figure out the bug
    return tools_table.get(tool_call["name"], lambda x: "I couldn't help with that").invoke(tool_call["args"])

@mlflow.trace(name="RAG Pipeline", span_type=SpanType.CHAIN)
def answer_question(question: str) -> str:
    """A simple RAG pipeline with manual tracing."""

    context = retrieve_context(question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": question},
        ],
        max_tokens=150,
    )

    return response.choices[0].message.content


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_context(question: str) -> str:
    """Simulate context retrieval."""
    # Simulate retrieval logic
    return f"Relevant context for: {question}"
