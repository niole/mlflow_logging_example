from openai import OpenAI
from  domino_eval_trace import read_ai_system_config
import mlflow

client = OpenAI()

ai_system_config = read_ai_system_config("./production/ai_system_config.yaml")

def assistant_evaluator(inputs, result) -> dict:
    eval_input = f"the question was: {inputs}, and the answer was {result}"
    messages = [
        {"role": "system", "content": "You are an llm judge for llm assistants who knows how to evaluate helpfulness of the assistant. You will be given an assistant's response and you will return a 1 if it was helpful and 0 if it was not. You will only reply with 1 or 0"},
        {"role": "user", "content": eval_input}
    ]
    # openai autolog example
    # Inputs and outputs of the API request will be logged in a trace
    response = client.chat.completions.create(model=ai_system_config["llm"]["chat_model"], messages=messages)
    content = response.choices[0].message.content

    if content is not None:
        eval_response = int(content)
        return {"helpfulness": eval_response}
    return {"helpfulness": 0}

"""
I have a RAG application, which is supposed to answer questions. When I ask it a question,
it sometimes doesn't know the answer. I need to evaluate its performance on a set of
questions in order to understand what I still need to add to its knowledge base.
"""
def question_fullfillment_evaluator(question: str, answer: str) -> dict[str, float]:
    """
    returns number from 0 - 1, where 0 means the answer is completely wrong
    and 1 means the answer is completely correct.
    """

    eval_input = f"the question was: {question}, and the answer was {answer}"
    messages = [
        {"role": "system", "content": """
            You are an llm judge for llm assistants who knows how to evaluate whether a question
            was fulfilled or not. You will be given an assistant's response and you will
            return a number from 0 - 1, where 0.0 means the answer is completely wrong or doesn't contain relevant information
            and 1.0 means the answer is completely correct and .5 means it was ok, but could have been more helpful. ONLY responsd with a float from 0.0 to 1.0
        """},
        {"role": "user", "content": eval_input}
    ]
    # openai autolog example
    # Inputs and outputs of the API request will be logged in a trace
    response = client.chat.completions.create(model=ai_system_config["llm"]["chat_model"], messages=messages)
    content = response.choices[0].message.content

    if content is not None:
        eval_response = float(content)
        return { "fullfilled": eval_response }
    return { "fullfilled": 0.0 }
