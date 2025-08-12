import time
import json
import random
import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from config import KAFKA_BOOTSTRAP, TOPIC_UPDATES

def connect_producer():
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            return producer
        except NoBrokersAvailable:
            print("Kafka producer not available yet, retrying in 1s...")
            time.sleep(1)

def fake_weights(size=100):
    return np.random.rand(size).tolist()

def send_updates(client_id=None, delay_range=(0.5, 3.0)):
    if client_id is None:
        client_id = random.randint(0, 1000000)
    producer = connect_producer()
    print(f"Client {client_id} starting to send updates")
    while True:
        w = fake_weights()
        n_samples = random.randint(1, 100)
        msg = {'client': client_id, 'weights': w, 'n_samples': n_samples}
        producer.send(TOPIC_UPDATES, msg)
        producer.flush()
        print(f"Client {client_id} sent update n={n_samples}")
        time.sleep(random.uniform(*delay_range))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, default=None)
    args = parser.parse_args()
    send_updates(client_id=args.client_id)
