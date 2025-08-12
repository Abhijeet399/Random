import time
import json
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import ray
from kafka.errors import NoBrokersAvailable
from config import KAFKA_BOOTSTRAP, TOPIC_UPDATES, TOPIC_GLOBAL, AGG_BATCH, AGG_TIMEOUT

# Connect to Ray head (head node is started in container, ray.init auto connects with address="auto")
# For safety, allow local init if no remote cluster is found
try:
    ray.init(address="auto", ignore_reinit_error=True)
except Exception as e:
    print("ray.init(address='auto') failed, trying local ray.init() fallback:", e)
    ray.init(ignore_reinit_error=True)

@ray.remote
def aggregate_batch(updates):
    total_n = sum(u['n_samples'] for u in updates)
    if total_n == 0:
        return [0.0] * len(updates[0]['weights'])
    agg = np.zeros_like(np.array(updates[0]['weights'], dtype=np.float64))
    for u in updates:
        w = np.array(u['weights'], dtype=np.float64)
        agg += (u['n_samples'] / total_n) * w
    return agg.tolist()

def connect_kafka_consumer():
    # retry until Kafka broker reachable
    while True:
        try:
            consumer = KafkaConsumer(
                TOPIC_UPDATES,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=False,
                consumer_timeout_ms=1000,
                group_id="lambda-fl-aggregators"
            )
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            return consumer, producer
        except NoBrokersAvailable:
            print("Kafka broker not available yet, retrying in 1s...")
            time.sleep(1)

def run_aggregator_loop():
    consumer, producer = connect_kafka_consumer()
    buffer = []
    last_trigger = time.time()
    round_id = 0
    print("Aggregator listening for updates on topic:", TOPIC_UPDATES)
    try:
        while True:
            for msg in consumer:
                update = msg.value
                buffer.append(update)
            now = time.time()
            if len(buffer) >= AGG_BATCH or (len(buffer) > 0 and now - last_trigger >= AGG_TIMEOUT):
                batch = buffer.copy()
                buffer.clear()
                last_trigger = now
                round_id += 1
                print(f"[round {round_id}] triggering aggregation: {len(batch)} updates")
                fut = aggregate_batch.remote(batch)
                agg_weights = ray.get(fut)
                payload = {'round': round_id, 'weights': agg_weights, 'timestamp': time.time()}
                producer.send(TOPIC_GLOBAL, payload)
                producer.flush()
                consumer.commit()
                print(f"[round {round_id}] published aggregated model to {TOPIC_GLOBAL}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Aggregator interrupted. Exiting...")
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    run_aggregator_loop()
