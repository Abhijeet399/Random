FROM python:3.9-slim

RUN apt-get update && apt-get install -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY client/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app

CMD ["python", "/app/client_sim.py"]
