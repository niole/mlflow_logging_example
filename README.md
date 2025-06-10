prove that the endpoints that we need to proxy are /api, /ajax-api, /get-artifact, and /model-versions/get-artifact

what can get logged in mlflow?
- this page lists what can be logged: https://www.mlflow.org/docs/1.26.1/tracking.html#automatic-logging: "Automatic logging allows you to log metrics, parameters, and models without the need for explicit log statements."
- data sources also get logged
- openai example logs a trace
- default mlflow example logs system metrics, model metrics, params, model artifacts, and a dataset

run:
```sh
export OPENAI_API_KEY=<key>
uv run ./trackingserver.sh
uv run default.py
```
