import mlflow
from random import random, randint
from mlflow.entities import SpanType
from openai import OpenAI
import os
from fastapi import FastAPI
from pydantic import BaseModel
import util
from pydantic import BaseModel
from dotenv import load_dotenv
from domino_eval_trace import init_domino_tracing

load_dotenv()

class Question(BaseModel):
	content: str

app = FastAPI()

init_domino_tracing("assistant_dev_server_2", os.getenv("PRODUCTION", "false") != "true")

@app.get("/")
async def answer_question():
    return util.answer_question("what is mlflow tracing?")

@app.post("/assistant")
async def assistant(question: Question):
    return util.ask_assistant(question.content)
