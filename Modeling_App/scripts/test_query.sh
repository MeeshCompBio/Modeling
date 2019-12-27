#!/usr/bin/env bash
curl -X POST \
  http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'cache-control: no-cache' \
  -d '{
    "SL": [4.7],
    "SW": [3.0],
    "PL": [1.4],
    "PW": [0.2],
}'