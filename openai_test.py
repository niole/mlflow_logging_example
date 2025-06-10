import os
from random import random, randint
import mlflow
from openai import OpenAI

mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
mlflow.set_experiment("openai_"+	str(randint(0, 1000)))

mlflow.openai.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
client = OpenAI()

messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"}
]


def main():
    print("Hello from openai_test")
    mlflow.start_run()
    # openai autolog example
    # Inputs and outputs of the API request will be logged in a trace
    client.chat.completions.create(model="gpt-4o-mini", messages=messages)

if __name__ == "__main__":
    main()

