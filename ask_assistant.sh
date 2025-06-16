#!/bin/bash

curl -X POST -H "Content-Type: application/json" \
-d "{\"content\":\"$1\"}" localhost:8000/assistant
