from openai import OpenAI
import mlflow
from domino_eval_trace import *
from util import answer_question_with_context
from db import collection

"""
This is an example of how you would evaluate the performance of a rag application in dev
"""

# add docs here in order to improve query context
collection.add(documents=["this week I have to do DOM-1234"], ids=["domino1234"])

def average(vs: list[str]) -> float:
    ns = [float(v) for v in vs]
    return sum(ns)/len(ns)

if __name__ == "__main__":
    init_domino_tracing("all_knowing_rag_agent_analysis", ["openai", "langchain"], False)

    # example questions
    questions = [
        "What is the capital of France?",
        "What tickets do I have to finish this week at work?",
        "is it warm enough to wear a t-shirt today?",
    ]

    with mlflow.start_run():

        # evaluating the answer question with context function
        for question in questions:
            answer_question_with_context(question)

        log_summary_metric("fullfilled", average)
