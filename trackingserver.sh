#!/bin/sh

mlflow server \
    --backend-store-uri sqlite:///:memory \
    --default-artifact-root file:///Users/niole.nelson/mlflow-test/mydata \
    --host 0.0.0.0 \
    --port 4040
