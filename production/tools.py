from random import random, randint
from openai import OpenAI
import mlflow
from langchain_core.tools import tool

client = OpenAI()
mlflow.openai.autolog()

@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

@tool
def ask_chat_bot_assistant(question: str) -> str:
    """As a chat bot assistant a question
    Args:
            question: the question to ask the chat bot
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant who messes up sometimes, but you try your best"},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    content = response.choices[0].message.content

    if content is None:
        return "I couldn't help with that"

    return content

tools = [add, multiply, ask_chat_bot_assistant]
tools_table = {
    "add": add,
    "multiply": multiply,
    "ask_chat_bot_assistant": ask_chat_bot_assistant
}
