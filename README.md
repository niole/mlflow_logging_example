requests for logging and accessing logged information go through the tracking server, except sometimes for artifacts.
By default, these requests are sent directly to the backing storage. https://mlflow.org/docs/latest/tracking/server#tracking-server-no-proxy

I think the endpoints that we need to proxy are /api, /ajax-api, /get-artifact. I read somewhere that /model-versions/get-artifact must get
proxied, too. However, I didn't see it in the logs.

if we use proxied artifact storage e.g. --serve-artifacts, this makes it so that request for saving artifacts are
proxied through the tracking server.


what can get logged in mlflow?
- this page lists what can be logged: https://www.mlflow.org/docs/1.26.1/tracking.html#automatic-logging: "Automatic logging allows you to log metrics, parameters, and models without the need for explicit log statements."
- data sources also get logged
- openai example logs a trace
- default mlflow example logs system metrics, model metrics, params, model artifacts, and a dataset

# how to use

## run dev mode

- initialize the backend.db: `sqlite3 backend.db`, then `create table cat(cat int);`
- run tracking server
- run caddy reverse proxy
- run experiment scripts default.py and openai.py
- view the proxied UI at localhost:3030

tracking server runs at port 4040
reverse proxy runs at port 3030

run the tracking server and an mlflow logging example:
```sh
export REV_PROXY_PORT=3030
export OPENAI_API_KEY=<key>
uv run ./trackingserver.sh
uv run default.py
```

run a caddy reverse proxy which logs all traffic to the tracking server:
```sh
caddy reverse-proxy \
--from :3030 --to :4040 \
--access-log
```

run a caddy reverse proxy which blocks all traffic except some:
```sh
#TODO
caddy reverse-proxy \
--from :3030 --to :4040 \
--access-log
```

## run  prod mode

run the tracking server
set the reverse proxy env var
run the prod server:
```sh
cd production
uv run fastapi dev server.py

curl localhost:8000
```

## test domino eval logging example

- run tracking server
- set environment variables
- run `uv run openai_test.py`
- get ID of run that you want to view
- run analyze traces script `analyze_trace_data.py --runid <runid>`

# Accessed URLs

URLs accessed when logging metrics:
/api, e.g. /api/2.0/mlflow/runs/log-inputs, /api/2.0/mlflow/runs/log-model, /api/2.0/mlflow/runs/log-batch, /api/2.0/mlflow/runs/log-inputs

URLs accessed when viewing UI:

run UI:
/ajax-api, e.g. /ajax-api/2.0/mlflow/runs/search

traces:
/ajax-api/2.0/mlflow/traces?experiment_ids=15&order_by=timestamp_ms%20DESC&filter=re

artifact viewing: /get-artifact?path=model%2FMLmodel&run_uuid=219ba0a944254b66a7e47b690df16d9f

overview, model metrics, and system metrics seem to be fetched always on page load:
/ajax-api/2.0/mlflow/model-versions/search
/ajax-api/2.0/mlflow/registered-models/search
/ajax-api/2.0/mlflow/runs/get?run_id=219ba0a944254b66a7e47b690df16d9f

# questions

- can we get an example script that logs everything important to customers?
