import os

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4317/v1/traces"
os.environ["OTEL_SERVICE_NAME"] = "mlflowtest"

import mlflow

from random import random, randint
from mlflow.entities import SpanType
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import util
from pydantic import BaseModel
from dotenv import load_dotenv
from domino_eval_trace import init_domino_tracing
import logging

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

class Question(BaseModel):
	content: str

app = FastAPI()

# all traces go to the same experiment, but may be linked to different ai system external models
init_domino_tracing("assistant_dev_server_3", is_production=os.getenv("PRODUCTION", "false") == "true")

@app.get("/")
async def answer_question():
    return util.answer_question("what is mlflow tracing?")

@app.post("/assistant")
async def assistant(question: Question):
    return util.ask_assistant(question.content)
