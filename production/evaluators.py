from openai import OpenAI
import mlflow

client = OpenAI()

def assistant_evaluator(inputs, result):
    eval_input = f"the question was: {inputs['args'][0]}, and the answer was {result}"
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
        return {"helpfulness": eval_response}

