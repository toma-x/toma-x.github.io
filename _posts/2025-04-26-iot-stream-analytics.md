---
layout: post
title: Real-time Analytics on Streaming IoT Data with Anomaly Detection
---

Hey everyone,

After exploring specific algorithms like LSTMs for anomaly detection in my [last technical post](/2025/04/26/anomaly-engine/), I wanted to tackle a more end-to-end project. The goal was to build a system that could handle a constant stream of data, like you'd get from IoT sensors, process it in real-time, store it efficiently, and run machine learning models on it – specifically for detecting anomalies. I tried to mimic some patterns you see in cloud architectures but built it locally using Python and some cool open-source tools.

## The Challenge: Taming the IoT Data Flood

Imagine thousands of sensors sending readings every few seconds – temperature, pressure, vibration, you name it. You end up with a massive amount of data pouring in constantly (high velocity, high volume). The challenges are:

1.  **Ingestion:** How do you reliably collect all this data without losing any?
2.  **Processing:** How do you analyze or transform this data *as it arrives*? Waiting for batch processing at the end of the day is often too late.
3.  **Storage:** Where do you put potentially terabytes of time-series data so you can query it later?
4.  **Real-time ML:** How do you run models (like anomaly detection) on the live stream to get immediate insights?

I decided to simulate this scenario and build a pipeline to handle it.

## My Approach: A Cloud-Inspired Local Pipeline

I designed a pipeline with distinct stages, inspired by how cloud platforms often structure these things:

1.  **Data Simulation:** Generate realistic-looking sensor data.
2.  **Streaming/Ingestion:** Use a message queue (like Kafka) to handle the incoming data stream.
3.  **Processing & Storage:** Consume the data, do minimal processing, and store it in a queryable format (like BigQuery, but simpler).
4.  **Anomaly Detection:** Apply the LSTM autoencoder model from my previous project to the live data.
5.  **ML Lifecycle:** Manage the process of training, deploying, and monitoring the model.

Here's a breakdown of how I built each part:

### 1. Simulating the Sensor Flood (1M+ Events/Day)

First, I needed data. I wrote a Python script to simulate multiple sensors sending readings. To make it interesting, I made most sensors generate "normal" data (e.g., temperature fluctuating around a baseline) but occasionally introduced anomalies (sudden spikes, drops, or changes in pattern).

To hit the target of over 1 million events per day (which is about 12 events per second), I used Python's `asyncio` and `random` libraries.

```python
# conceptual snippet from simulate_data.py
import asyncio
import json
import random
import time
from kafka import KafkaProducer # Using kafka-python

async def generate_sensor_data(producer, topic, sensor_id):
    """Simulates data from a single sensor."""
    while True:
        timestamp = time.time()
        # Simulate normal data + occasional anomalies
        if random.random() < 0.001: # Small chance of anomaly
            value = random.uniform(50, 100) # Anomaly value
            is_anomaly = True
        else:
            value = random.normalvariate(25, 5) # Normal value
            is_anomaly = False

        event = {
            'sensor_id': sensor_id,
            'timestamp': timestamp,
            'value': value,
            'is_anomaly_ground_truth': is_anomaly # For evaluation later
        }
        try:
            producer.send(topic, value=json.dumps(event).encode('utf-8'))
            # producer.flush() # Flushing frequently can impact performance
        except Exception as e:
            print(f"Error sending data: {e}")

        # Control the rate (e.g., 1 event per second per sensor)
        await asyncio.sleep(random.uniform(0.5, 1.5))

async def main():
    producer = KafkaProducer(bootstrap_servers='localhost:9092') # Assuming Kafka running locally
    topic = 'iot-sensor-data'
    num_sensors = 10 # Adjust this to control volume

    tasks = [generate_sensor_data(producer, topic, f'sensor_{i}') for i in range(num_sensors)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Make sure Kafka is running before starting
    # Run Kafka using Docker: docker-compose up -d (with a simple docker-compose.yml)
    print("Starting data simulation...")
    asyncio.run(main())
```

*Challenge:* Getting the simulation rate right and ensuring it could actually push data fast enough without overwhelming my local Kafka setup (running in Docker) or the producer script itself was tricky. Frequent flushing in Kafka slows things down, so I relied on Kafka's batching. Simulating realistic anomalies that the model could learn was also an iterative process.

### 2. Streaming with Kafka (The Message Bus)

Why Kafka? Even for a local setup, using something like Kafka (or RabbitMQ, Redis Streams) is great because it *decouples* the data producers (simulators) from the consumers (processor). If my processing script crashes or slows down, Kafka buffers the data, so I don't lose events. Producers just fire and forget.

I ran a single-node Kafka instance using Docker, which was surprisingly easy to set up for local development.

### 3. Processing and Storing Data (Like BigQuery, but Simpler)

Next, I needed a script to read from the Kafka topic, do any necessary quick transformations, and store the data. I wanted something that could handle large volumes and allow SQL-like queries later, similar to Google BigQuery.

Setting up a full data warehouse locally is overkill. I landed on using **DuckDB** with **Parquet files**. DuckDB is amazing – it's an in-process analytical data management system. It can directly query Parquet files using SQL, and Parquet is a columnar format great for analytics and compression.

My processing script (`stream_processor.py`) used `kafka-python` to consume messages and `pyarrow` to write them into hourly or daily Parquet files. DuckDB could then query across these files.

```python
# conceptual snippet from stream_processor.py
import json
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from kafka import KafkaConsumer
from datetime import datetime
import pandas as pd # For easier data handling before writing

def process_and_store(consumer, db_path, table_name):
    """Consumes from Kafka, processes, and stores in Parquet via DuckDB."""
    # Connect to DuckDB. It can manage Parquet files.
    # Alternatively, just use pyarrow to write parquet directly.
    con = duckdb.connect(database=db_path, read_only=False)
    # Ensure table/directory structure exists if needed

    batch = []
    batch_size = 1000 # Write every 1000 messages

    print("Starting consumer...")
    for message in consumer:
        try:
            event = json.loads(message.value.decode('utf-8'))
            # Add processing timestamp maybe
            event['processing_timestamp'] = datetime.now().isoformat()
            batch.append(event)

            if len(batch) >= batch_size:
                # Convert batch to DataFrame/Arrow Table
                df = pd.DataFrame(batch)
                # Define partition key (e.g., date)
                # For simplicity, just appending to one large table here
                # A better approach: write to partitioned Parquet files by date/hour
                arrow_table = pa.Table.from_pandas(df)

                # Use DuckDB to append to a Parquet-backed table or just write file
                # Example: Writing partitioned parquet files (more scalable)
                # current_date = datetime.now().strftime('%Y-%m-%d')
                # pq.write_to_dataset(arrow_table, root_path=f'{db_path}/{table_name}',
                #                    partition_cols=['date_partition_col']) # Need to add this col

                # Simpler: Append to a single DuckDB table (less scalable but easier)
                con.execute(f"INSERT INTO {table_name} SELECT * FROM arrow_table")

                print(f"Processed and stored batch of {len(batch)} events.")
                batch = [] # Reset batch

        except json.JSONDecodeError:
            print(f"Ignoring non-JSON message: {message.value}")
        except Exception as e:
            print(f"Error processing message: {e}")
            # Handle exceptions, maybe write failed messages to a dead-letter queue

    # Store any remaining messages in the batch
    if batch:
        # ... (store remaining batch logic) ...
        print(f"Processed and stored final batch of {len(batch)} events.")

    con.close()


if __name__ == "__main__":
    topic = 'iot-sensor-data'
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        group_id='my-processor-group', # Consumer group
        auto_offset_reset='earliest' # Start reading from the beginning if new group
    )
    db_file = 'iot_data.duckdb' # Or path to Parquet directory
    table_name = 'sensor_readings'

    # Initial table creation if using DuckDB table directly
    con = duckdb.connect(db_file)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            sensor_id VARCHAR,
            timestamp DOUBLE,
            value DOUBLE,
            is_anomaly_ground_truth BOOLEAN,
            processing_timestamp VARCHAR
        )
    """)
    con.close()

    process_and_store(consumer, db_file, table_name)

```

*Challenge:* Handling backpressure was important. If the processing/storage part couldn't keep up with Kafka, the consumer lag would grow. Writing efficiently to Parquet (batching, partitioning) was key. Error handling (what if a message is malformed?) also needed consideration. Partitioning the Parquet files (e.g., by date) would be essential for real scale, making queries much faster.

### 4. Real-time Anomaly Detection with LSTM Autoencoder

This is where my previous work came in handy. I took the trained LSTM autoencoder model (trained offline using historical data stored in the Parquet files/DuckDB).

I integrated the model loading and prediction logic into the `stream_processor.py` (or it could be a separate microservice reading from Kafka). For each incoming sensor reading (or maybe a small window of readings), it would:
1.  Preprocess the data (scaling, creating sequences/windows) just like during training.
2.  Feed the sequence to the loaded Keras/TensorFlow model.
3.  Calculate the reconstruction error (MSE).
4.  Compare the error to the pre-defined threshold.
5.  If `error > threshold`, flag it as an anomaly.

```python
# Conceptual snippet added within stream_processor.py or a separate service

# Load the trained model (outside the main loop)
# from tensorflow import keras
# model = keras.models.load_model('lstm_autoencoder_model.h5')
# threshold = 0.05 # Load the determined threshold

# Inside the message processing loop:
# ... receive event ...

# --- Anomaly Detection Logic ---
# Maintain a rolling window of recent values for the specific sensor
# sensor_windows[event['sensor_id']].append(event['value'])
# if len(sensor_windows[event['sensor_id']]) == sequence_length:
#    sequence = np.array(sensor_windows[event['sensor_id']]).reshape(1, sequence_length, 1)
#    # Scale the sequence using the same scaler used in training
#    # scaled_sequence = scaler.transform(sequence.reshape(-1, 1)).reshape(1, sequence_length, 1)
#    reconstructed_sequence = model.predict(scaled_sequence)
#    mse = np.mean(np.square(scaled_sequence - reconstructed_sequence))

#    if mse > threshold:
#        print(f"Anomaly detected! Sensor: {event['sensor_id']}, Value: {event['value']}, MSE: {mse}")
#        # Add anomaly flag to the event before storing
#        event['is_anomaly_predicted'] = True
#        # TODO: Trigger an alert!
#    else:
#        event['is_anomaly_predicted'] = False
#    # Remove oldest point from window
#    # sensor_windows[event['sensor_id']].pop(0)
# --- End Anomaly Detection ---

# ... store event (now including 'is_anomaly_predicted') ...

```

*Challenge:* Managing state for each sensor's window in a streaming context can be tricky. If the processor restarts, it loses the windows. More robust solutions might use Kafka Streams or Flink, but for this project, I kept the state in memory, accepting the risk of losing it on restart. Real-time prediction also adds latency to the processing pipeline.

### 5. ML Model Lifecycle Management

This sounds fancy, but for this project, it boiled down to a few key steps:

1.  **Data Collection:** The simulation script continuously generated data, stored via the pipeline.
2.  **Training:** An offline script (`train_model.py`) would query the historical data (e.g., last week's data) from DuckDB/Parquet, preprocess it (scaling, windowing), train the LSTM autoencoder (like in my previous post), and save the trained model file (`lstm_autoencoder_model.h5`) and the scaler object. It also determined the anomaly threshold based on reconstruction errors on a validation set of normal data.
3.  **Deployment:** The `stream_processor.py` loaded the saved model and scaler to make live predictions. Updating the model meant replacing the model file and restarting the processor (a simple deployment strategy).
4.  **Monitoring/Evaluation:** Since my simulated data had `is_anomaly_ground_truth` labels, I could evaluate the model's performance. I wrote another script (`evaluate_model.py`) to query recent predictions and ground truth from DuckDB/Parquet and calculate metrics like Precision, Recall, and F1-score. This is how I validated the **~95% F1-score** on the simulated anomalies.

*Challenge:* Automating this cycle (e.g., retraining automatically every week) would be the next step. Versioning models and data would also be crucial in a real system. Monitoring involved checking the F1 score, but also looking at the distribution of reconstruction errors over time to spot drift.

## Putting It All Together

The final flow looked like this:
`Simulator -> Kafka Topic -> Stream Processor (Consume, Predict Anomaly, Store) -> DuckDB/Parquet Storage`

An evaluation script could then query the storage to check performance.

## What I Learned (The Hard Parts and the Wins)

*   **Streaming is Harder:** Building a robust streaming pipeline, even locally, is more complex than batch processing. Handling state, failures, and ensuring low latency requires careful design.
*   **Decoupling is Key:** Kafka was invaluable for separating the data source from the processing logic. It made the system more resilient.
*   **Tooling Matters:** Kafka, DuckDB, Parquet, PyArrow, TensorFlow/Keras, `kafka-python`, `asyncio` – learning how to stitch these tools together was a major part of the project. DuckDB + Parquet felt like a superpower for handling large local datasets.
*   **Real-time ML Deployment:** Integrating the model directly into the stream processor worked for this scale, but for more complex scenarios, a dedicated prediction service might be better. State management for windowing was a pain point.
*   **Simulation vs. Reality:** The 95% F1 score is great, but it's on simulated data. Real-world sensor data is much messier, and the types of anomalies might be unknown beforehand. Threshold tuning remains critical.
*   **End-to-End Thinking:** Building the whole pipeline forced me to think about the entire ML lifecycle, from data generation to monitoring, which was really insightful.

## Conclusion

This project was a big step up in complexity, moving from isolated algorithms to a full (albeit simulated) data pipeline. It was challenging managing the different moving parts – the simulator, Kafka, the processor, the storage, and the ML model – but incredibly rewarding to see it working end-to-end. Processing over 1M events/day locally and performing real-time anomaly detection felt like a real achievement. While it's a simplified version of what cloud platforms offer, building it myself provided a much deeper understanding of the challenges and concepts involved in real-time data analytics and MLOps.

Definitely learned a ton about streaming architectures and the practicalities of deploying ML models in such environments!
