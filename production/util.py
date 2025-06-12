import mlflow
from random import random, randint
from mlflow.entities import SpanType
from openai import OpenAI
import os


@mlflow.trace(name="RAG Pipeline", span_type=SpanType.CHAIN)
def answer_question(question: str) -> str:
    """A simple RAG pipeline with manual tracing."""

    # Step 1: Retrieve context (manually traced)
    context = retrieve_context(question)

    # Step 2: Generate answer (automatically traced by OpenAI autolog)
    client = OpenAI()
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


# Execute the traced pipeline
result = answer_question("What is MLflow Tracing?")
print(result)
