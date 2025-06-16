this server contains some code examples using mlflow and openai.

# domino eval tracing for ai system example

- set env vars in .envrc
- start server: `uv run fastapi dev server.py`
- start mlflow server (see main readme)
- run ask assistant script: `./ask_assistant.sh "is oblivion remastered good?"`
- run trace analysis script: `uv run analyze_assistent_dev_server.py`

## what a user must know in order to use domino evaluations
- the server which contains what they want to evaluate must initialize an dev-mode experiment into which the evaluations
will be logged in dev mode
- must set the `PRODUCTION` environment variable, so that the domino evaluation library doesn't execute inline in production mode

## todos
- how to save production data and where to send it?
- what if there are multiple components in the system, like an agent with
mcp servers that it interacts with? can we make the tracing contain all components?
