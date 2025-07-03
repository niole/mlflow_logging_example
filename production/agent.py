#"""
#a pyfunc chatagentmodel
#
#https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent
#"""
#
##from mlflow.pyfunc import ChatAgent
#from mlflow import pyfunc
#from mlflow.types.responses import *
#
#class SimpleOllamaModel(pyfunc.ResponsesAgent):
#    def __init__(self):
#        self.model_name = "llama3.2:1b"
#        self.client = None
#        self.client = ollama.Client()
#
#    def predict(
#        self,
#		request: ResponsesAgentRequest,
#    ) -> ChatAgentResponse:
#        ollama_messages = self._convert_messages_to_dict(messages)
#        response = self.client.chat(model=self.model_name, messages=ollama_messages)
#        return ChatAgentResponse(**{"messages": [response["message"]]})
