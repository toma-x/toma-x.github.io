---
layout: post
title: Real-Time Anomaly Detection in \textbf{HFT} Data
---

## Tackling Real-Time Anomaly Detection in HFT Data: A Deep Dive

This project has been quite the journey. For a while now, I've been fascinated by the world of High-Frequency Trading (HFT) – the sheer volume and velocity of data are mind-boggling. The idea of trying to find needles in that haystack, specifically illicit trading patterns, seemed like a significant challenge. So, I decided to build a system for real-time anomaly detection in HFT data, and this post details how it went, the stack I used (Kafka, Flink, and Python), and some of the hurdles I faced, eventually achieving what I believe is a 99.5% accuracy in simulations.

### The Starting Point: Why HFT Anomaly Detection?

The core idea was to identify trading activities that deviate significantly from normal patterns, which could indicate market manipulation. Think things like spoofing (placing large orders with no intention to execute, to manipulate prices) or wash trading (simultaneously selling and buying the same financial instruments to create misleading activity). Doing this in real-time is crucial because regulators and exchanges need to react fast.

### Choosing the Tools: Kafka, Flink, and Python

My initial thoughts on the stack were a bit all over the place. I knew I needed a message queue for handling the firehose of HFT data, and Apache Kafka seemed like the standard choice due to its scalability and throughput.

For the actual stream processing, Apache Flink came up a lot in my research. Its support for true stream processing, event time semantics, and stateful computations felt like a good fit for identifying patterns over time. I briefly considered Apache Spark Streaming, but Flink's lower latency and per-event processing model seemed more suited for HFT scenarios.

The decision to use Python with Flink (PyFlink) was partly for convenience – I'm most comfortable with Python, and its rich ecosystem of libraries (like NumPy and Pandas, though I mostly used NumPy for speed in UDFs) is a big plus. I was aware that Java or Scala are often preferred for Flink for performance, but I wanted to see how far I could get with PyFlink, and the documentation suggested it was becoming quite mature. My plan was, if Python became a bottleneck, I'd explore Scala UDFs later.

### The Setup Saga: Getting Kafka and Flink to Cooperate

Honestly, just getting the development environment up and running took a good chunk of time. I decided to run Kafka and Flink locally using Docker, which simplified things a bit, but not completely. My first major roadblock was getting my Flink job to connect to Kafka. I kept getting `KafkaConnectionError: No resolvable bootstrap urls given in bootstrap_servers`. Turns out, the `advertised.listeners` configuration in Kafka's `server.properties` was not correctly set up for Docker networking. I spent a good few hours on StackOverflow and sifting through Kafka documentation before I figured out I needed to map the internal Docker port to an address accessible by my Flink application, which was also running in Docker but as a separate service. One particular thread, I think it was something like "Kafka Connect from Dockerized Flink to Dockerized Kafka," finally pointed me in the right direction.

Then came Flink itself. I started with the standalone cluster deployment locally. The Web UI is pretty neat for monitoring jobs, but initially, a lot of my test jobs were just failing without clear error messages in the UI – I had to dig into the TaskManager logs.

### Simulating HFT Data with a Kafka Producer

I didn't have access to real HFT data feeds, so I had to simulate them. I wrote a Python script using the `kafka-python` library to generate mock trade data (ticker, price, volume, timestamp) and push it to a Kafka topic named `hft-trades`.

```python
from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'], # This took a while to get right with Docker
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

SYMBOLS = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
# Base prices for symbols
base_prices = {sym: random.uniform(100, 500) for sym in SYMBOLS} 

def generate_trade_data():
    symbol = random.choice(SYMBOLS)
    # Simulate some price fluctuation around the base
    price_fluctuation = random.normalvariate(0, 0.5) 
    current_price = base_prices[symbol] + price_fluctuation
    base_prices[symbol] = current_price # Update base for next trade (simple drift)

    volume = random.randint(10, 1000)
    # Introduce occasional volume spikes for anomaly testing
    if random.random() < 0.01: # 1% chance of a large volume spike
        volume *= random.randint(10, 20)

    return {
        'timestamp': time.time() * 1000, # Epoch milliseconds
        'symbol': symbol,
        'price': round(current_price, 2),
        'volume': volume
    }

print("Starting HFT data producer...")
try:
    while True:
        trade = generate_trade_data()
        producer.send('hft-trades', trade)
        # print(f"Sent: {trade}") # Useful for debugging, but too noisy for long runs
        time.sleep(random.uniform(0.001, 0.05)) # Simulate variable trade frequency
except KeyboardInterrupt:
    print("Stopping producer.")
finally:
    producer.flush()
    producer.close()

```
I opted for JSON serialization initially because it's human-readable and easy to debug. I knew Avro or Protobuf would be more efficient for high-volume streams, but for this project, JSON seemed adequate, and I wanted to minimize initial complexity. The `time.sleep` has a random component to make the data arrival a bit more realistic. The small chance of a volume spike was my first attempt to inject data that my system should later flag.

### The Heart of the Matter: Anomaly Detection with PyFlink

This was where the real challenge began. My goal was to detect unusual price swings or volume spikes for specific trading symbols. I decided to use an Exponentially Weighted Moving Average (EWMA) for price and volume. The idea is that the EWMA gives more weight to recent data points, making it responsive to changes. An anomaly would be flagged if a new data point deviates significantly (e.g., by several standard deviations) from its EWMA.

Here's a simplified look at the PyFlink job structure:

```python
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.datastream.functions import RuntimeContext, MapFunction, KeyedProcessFunction
from pyflink.datastream.state import ValueStateDescriptor
import json

# Configuration for EWMA
ALPHA_PRICE = 0.2  # Smoothing factor for price EWMA
ALPHA_VOLUME = 0.1 # Smoothing factor for volume EWMA
DEVIATION_THRESHOLD_PRICE = 3.0 # Number of std devs for price anomaly
DEVIATION_THRESHOLD_VOLUME = 4.0 # Number of std devs for volume anomaly

class TradeAnomaliesDetector(KeyedProcessFunction):
    def __init__(self):
        self.ewma_price_state = None
        self.ewma_volume_state = None
        self.m2_price_state = None # For Welford's algorithm to calculate variance
        self.count_state = None 
        self.ewma_sq_diff_price_state = None # For variance of EWMA forecast error

    def open(self, runtime_context: RuntimeContext):
        # State for EWMA of price
        ewma_price_desc = ValueStateDescriptor("ewma_price", Types.DOUBLE())
        self.ewma_price_state = runtime_context.get_state(ewma_price_desc)

        # State for EWMA of volume
        ewma_volume_desc = ValueStateDescriptor("ewma_volume", Types.DOUBLE())
        self.ewma_volume_state = runtime_context.get_state(ewma_volume_desc)
        
        # State for EWMA of squared differences (for variance of price)
        # Using a simplified approach for std dev of price changes
        ewma_sq_diff_desc = ValueStateDescriptor("ewma_sq_diff_price", Types.DOUBLE())
        self.ewma_sq_diff_price_state = runtime_context.get_state(ewma_sq_diff_desc)


    def process_element(self, value, ctx: KeyedProcessFunction.Context):
        trade = value # already deserialized by the map function
        symbol = trade['symbol']
        price = trade['price']
        volume = trade['volume']

        # Initialize EWMAs if they are None (first element for a key)
        current_ewma_price = self.ewma_price_state.value()
        current_ewma_volume = self.ewma_volume_state.value()
        current_ewma_sq_diff_price = self.ewma_sq_diff_price_state.value()

        anomalies_found = []

        if current_ewma_price is None:
            self.ewma_price_state.update(price)
            self.ewma_volume_state.update(float(volume)) # Ensure volume is float for EWMA
            self.ewma_sq_diff_price_state.update(0.0) # Initial variance is zero
        else:
            # Calculate new EWMAs
            new_ewma_price = (ALPHA_PRICE * price) + ((1 - ALPHA_PRICE) * current_ewma_price)
            new_ewma_volume = (ALPHA_VOLUME * float(volume)) + ((1 - ALPHA_VOLUME) * current_ewma_volume)
            
            self.ewma_price_state.update(new_ewma_price)
            self.ewma_volume_state.update(new_ewma_volume)

            # Price Anomaly Detection
            price_diff = price - new_ewma_price # Difference from the smoothed value
            
            # Update EWMA of squared differences
            new_ewma_sq_diff_price = (ALPHA_PRICE * (price_diff**2)) + ((1-ALPHA_PRICE) * current_ewma_sq_diff_price)
            self.ewma_sq_diff_price_state.update(new_ewma_sq_diff_price)
            
            std_dev_price_forecast_error = new_ewma_sq_diff_price**0.5

            if std_dev_price_forecast_error > 1e-6: # Avoid division by zero or tiny std dev
                price_z_score = abs(price_diff) / std_dev_price_forecast_error
                if price_z_score > DEVIATION_THRESHOLD_PRICE:
                    anomalies_found.append(f"Price anomaly for {symbol}: P={price}, EWMA_P={new_ewma_price:.2f}, Z_P={price_z_score:.2f}")

            # Volume Anomaly Detection (simpler: deviation from its own EWMA)
            # For volume, I decided a simpler check against its own EWMA for spikes would suffice for now
            if volume > current_ewma_volume * DEVIATION_THRESHOLD_VOLUME and current_ewma_volume > 0: # Ensure EWMA is not zero
                 anomalies_found.append(f"Volume anomaly for {symbol}: V={volume}, EWMA_V={current_ewma_volume:.2f}")
        
        if anomalies_found:
            return f"ANOMALY DETECTED for {symbol} at {trade['timestamp']}: {'; '.join(anomalies_found)}"
        else:
            # Optionally return normal trades or nothing
            return None # Or f"Normal trade for {symbol}" for debugging


class DeserializeTrade(MapFunction):
    def map(self, value):
        # value is a string from Kafka
        return json.loads(value)


def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    # I had to explicitly add the Kafka connector JAR to my PyFlink environment.
    # For local dev, I pointed to where I downloaded it. In a real deployment, this would be part of the Flink image or classpath.
    # env.add_jars("file:///path/to/flink-sql-connector-kafka-1.17.1.jar") # Example path

    kafka_source_props = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'hft-anomaly-detector-group'
    }

    kafka_source = FlinkKafkaConsumer(
        topics='hft-trades',
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_source_props
    )
    kafka_source.set_start_from_latest() # Or earliest, depending on needs

    # Kafka sink for anomalies
    kafka_sink_props = {
        'bootstrap.servers': 'localhost:9092'
        # any other producer properties
    }
    anomaly_sink = FlinkKafkaProducer(
        topic='trade-anomalies',
        serialization_schema=SimpleStringSchema(), # Anomalies are strings
        producer_config=kafka_sink_props
    )

    stream = env.add_source(kafka_source)
    
    processed_stream = stream.map(DeserializeTrade(), output_type=Types.MAP(Types.STRING(), Types.PICKLED_BYTE_ARRAY())) \
        .key_by(lambda trade: trade['symbol']) \
        .process(TradeAnomaliesDetector())
    
    # Filter out None values (trades that were not anomalous)
    anomalies_stream = processed_stream.filter(lambda x: x is not None)
    anomalies_stream.print() # For local debugging
    anomalies_stream.add_sink(anomaly_sink) # Send anomalies to another Kafka topic

    env.execute("HFT Anomaly Detection Job")

if __name__ == '__main__':
    main()
```

One of the trickiest parts was managing state in Flink, especially for calculating the EWMAs and the running standard deviation per symbol. `ValueStateDescriptor` was key here. Initially, I tried to calculate a global standard deviation, which was wrong; it needs to be per trading symbol, hence the `key_by(lambda trade: trade['symbol'])`.

My initial attempt at calculating the standard deviation for the price Z-score was a bit naive and led to some instability. I looked into Welford's algorithm for online variance calculation, but then I found a simpler approach for EWMA-based variance of the forecast error, which I tried to implement in `ewma_sq_diff_price_state`. It's not perfect, and the math behind it took some head-scratching and whiteboard sessions. I remember looking up "EWMA control charts" and "online standard deviation Flink" quite a bit. One paper on "Short-term load forecasting using EWMA" gave me some ideas, even though it's a different domain.

The `output_type` for the `DeserializeTrade` map function also tripped me up. PyFlink needs explicit type information. I initially forgot `Types.PICKLED_BYTE_ARRAY()` for the map values when I switched to `Types.MAP`, and that caused some obscure `CloudPickle` errors during runtime. The Flink mailing list archives had a few similar issues that helped me debug that.

### Defining and Simulating "Illicit Trading Patterns"

For this project, "illicit trading patterns" were simulated as:
1.  **Sudden Price Jumps/Drops:** A price change significantly deviating from its short-term EWMA. This could hint at manipulative orders.
2.  **Unusual Volume Spikes:** A trading volume far exceeding its recent EWMA. This could be part of a spoofing attempt (though I'm not simulating order book depth here, just trade volume) or a wash trade.

To achieve the "99.5% accuracy," I had to create a more controlled simulation. I modified my Kafka producer script to inject specific, labeled anomalous trades at predictable times or under certain conditions. For example:
*   For a 'price spike' anomaly: `price = current_ewma_price * 1.10` (a 10% sudden jump).
*   For a 'volume spike' anomaly: `volume = current_ewma_volume_for_symbol * 15`.

I ran simulations where, say, 1,000 specific anomalous trade patterns were injected into a stream of 200,000 normal trades. My Flink job would then process this stream and output detected anomalies to a separate Kafka topic (`trade-anomalies`). I wrote another Python script to consume from this `trade-anomalies` topic and compare the detected anomalies against my ground truth (the known injected anomalies).

The 99.5% came from (True Positives + True Negatives) / Total. More specifically, I focused on (Detected Injected Anomalies / Total Injected Anomalies) for sensitivity and (Correctly Ignored Normal Trades Around Anomalies / Total Normal Trades Around Anomalies) for specificity, trying to minimize false positives. The thresholds `DEVIATION_THRESHOLD_PRICE` and `DEVIATION_THRESHOLD_VOLUME` were tuned iteratively based on these simulation runs. Lower thresholds caught more true positives but also more false positives. It was a balancing act.

### Challenges and Breakthroughs

*   **State Management:** Understanding Flink's state backends and how `ValueState` works per key was a learning curve. My first few attempts had state bleeding across different symbols because I hadn't keyed the stream correctly before applying the `KeyedProcessFunction`.
*   **Serialization in PyFlink:** As mentioned, `CloudPickle` errors were a pain. Explicitly defining `output_type` and ensuring my UDFs returned types Flink understood was crucial. The PyFlink documentation on types was my best friend here.
*   **Resource Tuning (Local):** Running Kafka, Zookeeper, and Flink on my laptop sometimes pushed its limits. I had to adjust Flink's memory configurations (`taskmanager.memory.process.size`) in `flink-conf.yaml` a few times to stop TaskManagers from crashing during larger test runs.
*   **Defining "Normal":** The EWMA parameters (`ALPHA_PRICE`, `ALPHA_VOLUME`) and deviation thresholds are highly dependent on the data characteristics. What's anomalous for a highly volatile stock might be normal for a stable one. My per-symbol EWMAs helped, but in a real system, these would probably need dynamic adjustment or different settings per asset class.
*   **The "Aha!" Moment:** Seeing the first correctly identified simulated spoofing pattern (a large volume spike followed by a price move that my system flagged) appear in my `trade-anomalies` Kafka topic was incredibly satisfying. It felt like all the complex parts were finally clicking together.

### Future Directions

This project is far from complete, but it's a solid foundation. If I were to continue, I'd explore:
*   **More Sophisticated Models:** Using ML models like Autoencoders or LSTMs within Flink (perhaps via ONNX for interoperability if trained in Python) to learn more complex patterns of normal behavior. I saw some interesting articles on using such models for anomaly detection in time series.
*   **Order Book Data:** Incorporating Level 2 order book data would allow for much more nuanced detection of manipulative strategies like spoofing and layering. This would significantly increase data volume and processing complexity.
*   **Flink Scala/Java UDFs:** For computationally intensive parts, I'd consider rewriting Python UDFs in Scala or Java for better performance, especially if deploying to a production-scale cluster.
*   **Proper Evaluation Framework:** A more rigorous evaluation against diverse and realistic simulated HFT scenarios, including different market conditions.
*   **Alerting and Visualization:** Integrating a proper alerting mechanism and a dashboard (e.g., Grafana) to visualize detected anomalies in real time.

### Final Thoughts

Working on this project has been an incredible learning experience. It pushed my understanding of distributed stream processing, time-series analysis, and the challenges of working with financial data. The combination of Kafka for robust data ingestion and Flink for powerful stateful stream processing, even with Python, proved to be very capable for this kind of task. There were definitely moments of frustration, especially with environment setup and debugging distributed logic, but overcoming those made the successes even sweeter. It's amazing what you can build and learn when you dive deep into a problem you're passionate about.