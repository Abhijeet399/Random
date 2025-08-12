import os

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC_UPDATES = os.environ.get("TOPIC_UPDATES", "job_updates")   # clients -> aggregator
TOPIC_GLOBAL = os.environ.get("TOPIC_GLOBAL", "job_global")      # aggregator -> clients
AGG_BATCH = int(os.environ.get("AGG_BATCH", "5"))
AGG_TIMEOUT = float(os.environ.get("AGG_TIMEOUT", "10.0"))  # seconds
