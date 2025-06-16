import mlflow
from random import random, randint
from mlflow.entities import SpanType
from openai import OpenAI
import os
from fastapi import FastAPI
from pydantic import BaseModel
import util
from pydantic import BaseModel

class Question(BaseModel):
	content: str

app = FastAPI()

if os.getenv("PRODUCTION", "false") != "true":
    # only define an experiment if running in dev mode
    mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
    mlflow.set_experiment("assistant_dev_server")

# Enable automatic tracing for OpenAI
mlflow.openai.autolog()


@app.get("/")
async def answer_question():
    return util.answer_question("what is mlflow tracing?")

@app.post("/assistant")
async def assistant(question: Question):
    return util.ask_assistant(question.content)
