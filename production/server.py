import os
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

logging.basicConfig(level=logging.WARNING)

load_dotenv()

class Question(BaseModel):
        content: str

app = FastAPI()

init_domino_tracing(
    "all_knowing_rag_agent_analysis2",
    is_production=os.getenv("PRODUCTION", "false") == "true",
    ai_frameworks=["openai", "langchain"],
    ai_system_config_path="./production/ai_system_config.yaml"
)

@app.post("/assistant")
async def assistant(question: Question):
    return util.answer_question_with_context(question.content)
