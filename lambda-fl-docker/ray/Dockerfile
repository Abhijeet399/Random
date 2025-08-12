FROM python:3.9-slim

# install netcat (used by entrypoint wait logic) and build deps
RUN apt-get update && apt-get install -y netcat-openbsd gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy ray requirements and install
COPY ray/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app code (from project root app/)
COPY app /app

# copy entrypoint
COPY ray/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
