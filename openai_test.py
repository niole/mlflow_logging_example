import os
from random import random, randint
import mlflow
from openai import OpenAI
from domino_eval_run import domino_eval_run, domino_eval_run_dec, domino_eval_trace

mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
mlflow.set_experiment("openai_"+        str(randint(0, 1000)))

mlflow.openai.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
client = OpenAI()

def content_run(eval_input):
    print("Hello from openai_test")


    messages = [
        {"role": "system", "content": "You are an llm judge for llm assistants who knows how to evaluate helpfulness of the assistant. You will be given an assistant's response and you will return a 1 if it was helpful and 0 if it was not. You will only reply with 1 or 0"},
        {"role": "user", "content": eval_input}
    ]
    # openai autolog example
    # Inputs and outputs of the API request will be logged in a trace
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    content = response.choices[0].message.content

    if content is not None:
        eval_response = int(content)
        mlflow.log_metric("output", eval_response)
        return eval_response

@domino_eval_trace
def content_run_trace_example(eval_input):
    print("Hello from openai_test")


    messages = [
        {"role": "system", "content": "You are an llm judge for llm assistants who knows how to evaluate helpfulness of the assistant. You will be given an assistant's response and you will return a 1 if it was helpful and 0 if it was not. You will only reply with 1 or 0"},
        {"role": "user", "content": eval_input}
    ]
    # openai autolog example
    # Inputs and outputs of the API request will be logged in a trace
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    content = response.choices[0].message.content

    if content is not None:
        eval_response = int(content)
        return { "helpfulness": eval_response }
"""
BEST
"""
def main3():
    print("Hello from openai_test main3")

    with mlflow.start_run():

        more_eval_inputs = [
            "I am too busy to help and I am not sorry",
            "I am happy to help",
            "I am not sure how to help"
        ]

        for e in more_eval_inputs:
            content_run_trace_example(e)

@domino_eval_run_dec
def main2():
    print("Hello from openai_test eval dec")

    eval_input = "I am too busy to help and I am not sorry"
    mlflow.log_param("input", eval_input)
    content_run(eval_input)

def main():
    print("Hello from openai_test")

    eval_input = "I am too busy to help and I am not sorry"

    with domino_eval_run(eval_input) as eval_run:
        content_run(eval_input)


if __name__ == "__main__":
    #main()
    #main2()
    main3()
