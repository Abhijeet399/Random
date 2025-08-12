#!/bin/sh
set -e

ROLE=${ROLE:-head}
RAY_PORT=${RAY_PORT:-6379}
DASH_PORT=${DASH_PORT:-8265}

if [ "$ROLE" = "head" ]; then
  echo "Starting Ray head..."
  ray start --head --port=${RAY_PORT} --dashboard-host 0.0.0.0 --dashboard-port=${DASH_PORT}
  echo "Ray head started; running aggregator..."
  # run aggregator (this script expects app/aggregator.py present)
  python /app/aggregator.py
else
  echo "Starting Ray worker and connecting to head..."
  # wait for ray-head port to be open
  while ! nc -z ray-head ${RAY_PORT}; do
    echo "Waiting for ray-head to be available..."
    sleep 1
  done
  ray start --address=ray-head:${RAY_PORT}
  echo "Ray worker started. Sleeping to keep container alive."
  tail -f /dev/null
fi
