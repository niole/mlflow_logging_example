import mlflow
from random import random, randint
from mlflow.entities import SpanType
from openai import OpenAI
import os
from fastapi import FastAPI
from pydantic import BaseModel
import util

app = FastAPI()

# Set up MLflow tracking
mlflow.set_tracking_uri(f"http://localhost:{os.environ['REV_PROXY_PORT']}")
mlflow.set_experiment("prod_tracing_"+	str(randint(0, 1000)))

# Enable automatic tracing for OpenAI
mlflow.openai.autolog()


@app.get("/")
async def answer_question():
    return util.answer_question("what is mlflow tracing?")
