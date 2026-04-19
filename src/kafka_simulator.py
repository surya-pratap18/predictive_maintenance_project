import json
import time
import random
import queue
import threading
import pandas as pd
from datetime import datetime

# ============================================
# Simulated Kafka Topics (Python Queues)
# ============================================
topics = {
    'raw-sensor-data'  : queue.Queue(),
    'cleaned-features' : queue.Queue(),
    'anomalies-flagged': queue.Queue()
}

# ============================================
# Load Dataset
# ============================================
cols = ['engine_id', 'cycle'] + \
       [f'setting_{i}' for i in range(1, 4)] + \
       [f'sensor_{i}' for i in range(1, 22)]

train_df = pd.read_csv(
    'data/train_FD001.txt',
    sep='\s+',
    header=None,
    names=cols,
    engine='python'
)
train_df.dropna(axis=1, how='all', inplace=True)

max_cycle = train_df.groupby('engine_id')['cycle'].max().reset_index()
max_cycle.columns = ['engine_id', 'max_cycle']
train_df = train_df.merge(max_cycle, on='engine_id')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df.drop(columns=['max_cycle'], inplace=True)

print(f"✅ Dataset loaded: {train_df.shape}")

# ============================================
# Producer (simulates Kafka producer)
# ============================================
def producer(num_messages=100, delay=0.5):
    print("\n🚀 Producer started - streaming sensor data...")
    messages_sent = 0

    for _ in range(num_messages):
        row = train_df.sample(1).iloc[0]

        record = {
            'timestamp'          : datetime.now().isoformat(),
            'equipment_id'       : int(row['engine_id']),
            'cycle'              : int(row['cycle']),
            'RUL'                : float(row['RUL']),
            'setting_1'          : float(row['setting_1']),
            'setting_2'          : float(row['setting_2']),
            'is_injected_anomaly': False
        }

        # Add sensors
        for i in range(1, 22):
            col = f'sensor_{i}'
            if col in row.index:
                record[col] = float(row[col])

        # Inject random anomaly (5% chance)
        if random.random() < 0.05:
            sensor = f'sensor_{random.randint(2, 15)}'
            if sensor in record:
                record[sensor] *= random.uniform(2.0, 4.0)
                record['is_injected_anomaly'] = True

        # Put message in topic queue
        topics['raw-sensor-data'].put(record)
        messages_sent += 1

        if messages_sent % 10 == 0:
            print(f"📤 Sent {messages_sent} messages")

        time.sleep(delay)

    print(f"\n✅ Producer done! Total sent: {messages_sent}")

# ============================================
# Consumer (simulates Kafka consumer)
# ============================================
def consumer(max_messages=100):
    print("📡 Consumer started - listening for messages...\n")
    received    = 0
    anomalies   = 0

    while received < max_messages:
        try:
            # Get message from queue
            record = topics['raw-sensor-data'].get(timeout=5)
            received += 1

            is_anomaly = record.get('is_injected_anomaly', False)
            if is_anomaly:
                anomalies += 1

            status = "🚨 ANOMALY" if is_anomaly else "✅ Normal"
            print(f"[{received:4d}] Equipment: {record['equipment_id']:3d} | "
                  f"Cycle: {record['cycle']:4d} | "
                  f"RUL: {record['RUL']:6.1f} | "
                  f"{status}")

            # Forward anomalies to anomalies topic
            if is_anomaly:
                topics['anomalies-flagged'].put(record)

        except queue.Empty:
            print("⏳ Waiting for messages...")
            continue

    print(f"\n✅ Consumer done!")
    print(f"📊 Received: {received} | Anomalies: {anomalies} "
          f"({anomalies/received*100:.1f}%)")

# ============================================
# Run Producer & Consumer Together
# ============================================
if __name__ == "__main__":
    print("="*55)
    print("🔄 KAFKA STREAM SIMULATOR")
    print("="*55)

    # Run producer in background thread
    producer_thread = threading.Thread(
        target=producer,
        kwargs={'num_messages': 50, 'delay': 0.3}
    )

    # Run consumer in background thread
    consumer_thread = threading.Thread(
        target=consumer,
        kwargs={'max_messages': 50}
    )

    # Start both
    producer_thread.start()
    consumer_thread.start()

    # Wait for both to finish
    producer_thread.join()
    consumer_thread.join()

    print("\n" + "="*55)
    print("✅ Streaming simulation complete!")
    print("="*55)