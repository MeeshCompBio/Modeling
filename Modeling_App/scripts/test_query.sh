#!/usr/bin/env bash
curl -X POST \
  http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'cache-control: no-cache' \
  -d '{
  "VALUE": [0.1234],
}'