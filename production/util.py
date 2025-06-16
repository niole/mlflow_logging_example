import mlflow
from random import random, randint
from mlflow.entities import SpanType
from openai import OpenAI
import os
from  domino_eval_trace import domino_eval_trace_2
import evaluators

client = OpenAI()

@domino_eval_trace_2(evaluator=evaluators.assistant_evaluator)
def ask_assistant(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant who messes up sometimes, but you try your best"},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    content = response.choices[0].message.content

    if content is None:
        return "I couldn't help with that"
    return content

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
