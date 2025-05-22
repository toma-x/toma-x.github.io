---
layout: post
title: Real-time Anomaly Detection HFT
---

## Real-time Anomaly Detection in HFT: A Deep Dive into my Kafka-Flink Project

This project has been quite a journey. The goal was to build a system capable of detecting anomalies in high-frequency trading (HFT) data, ideally in sub-second timeframes. The idea itself sounded incredibly challenging, especially given the sheer volume and velocity of data in HFT. But, I was determined to see if I could pull it off using some of the stream processing tools I'd been reading about.

### The Initial Stack: Kafka and Flink

Right from the start, I knew I needed a robust pipeline for data ingestion and a powerful stream processor. **Apache Kafka** seemed like the obvious choice for the data pipeline. Its ability to handle high-throughput, persistent message streams is well-documented, and it's pretty much industry standard for this kind of thing. I needed something that wouldn't buckle under the pressure of potentially thousands of trade messages per second.

For the processing layer, I settled on **Apache Flink**. The main draw here was its promise of true stream processing with low latency and strong support for event time processing and stateful computations. I’d read about its capabilities in handling out-of-order events using watermarks, which I suspected would be crucial for HFT data. The alternative, Spark Streaming, felt more like micro-batching, and I really wanted to aim for that sub-second detection.

Setting them up together wasn't trivial. Getting the Flink Kafka connector configured correctly, ensuring the JARs were in the right place, and understanding how Flink would partition its consumption from Kafka topics took a fair bit of trial and error. I remember wrestling with classpath issues for a good day or two.

### Getting the Data Flowing: Kafka Producer and Trade Data

Before processing, I needed data. I simulated HFT data with a simple Python script. The trade messages were JSON objects containing a timestamp, ticker symbol, price, and volume.

```python
# trade_producer.py
import json
import time
import random
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

tickers = ['ALPHA', 'BETA', 'GAMMA', 'DELTA']

def generate_trade_data():
    trade = {
        'timestamp': time.time(), # epoch timestamp
        'ticker': random.choice(tickers),
        'price': round(random.uniform(100, 200), 2),
        'volume': random.randint(10, 1000)
    }
    return trade

if __name__ == "__main__":
    print("Starting HFT data producer...")
    try:
        while True:
            trade = generate_trade_data()
            producer.send('hft-trades', trade)
            # print(f"Sent: {trade}")
            time.sleep(random.uniform(0.001, 0.05)) # simulate high frequency
    except KeyboardInterrupt:
        print("Stopping producer.")
    finally:
        producer.flush()
        producer.close()
```
Serialization was a point of consideration. JSON is human-readable and easy to work with in Python, but for ultra-high performance, Avro or Protobuf would have been better. Given this was a personal project and I wanted to iterate quickly, JSON felt like a reasonable trade-off for the initial development stages. Topic naming was simple: `hft-trades`.

### Flink at the Core: Processing Trade Streams

Once data was flowing into Kafka, the next step was to get Flink to consume and process it. I decided to use Flink's DataStream API with Java.

My first hurdle with Flink was grasping event time processing. HFT data arrives very quickly, but network latency or producer delays can mean events don't always arrive at the Flink job in the exact order they occurred. Using event time, extracted from the timestamp within my trade data, and configuring watermarks correctly, was critical. Without it, any windowed calculations would be inaccurate. I spent a lot of time on the Flink documentation pages for "Event Time and Watermarks" and found a particular StackOverflow answer (can't find the link now, unfortunately) that clarified how `AssignerWithPeriodicWatermarks` works in practice.

Here's a simplified snippet of how I started consuming from Kafka and preparing for windowing:

```java
// In my Flink Job
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.functions.timestamps.BoundedOutOfOrdernessTimestampExtractor;

// ... other imports

public class HFTAnomalyDetectorJob {

    public static void main(String[] args) throws Exception {
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("group.id", "hft-flink-consumer");

        FlinkKafkaConsumer<String> kafkaSource = new FlinkKafkaConsumer<>(
            "hft-trades",
            new SimpleStringSchema(),
            properties
        );

        // Assign timestamps and watermarks based on the 'timestamp' field in the JSON
        DataStream<TradeData> tradeStream = env
            .addSource(kafkaSource)
            .map(value -> { // Deserialize JSON string to TradeData POJO
                ObjectMapper objectMapper = new ObjectMapper();
                // This is a bit simplistic for error handling in production
                return objectMapper.readValue(value, TradeData.class);
            })
            .assignTimestampsAndWatermarks(
                new BoundedOutOfOrdernessTimestampExtractor<TradeData>(Time.milliseconds(200)) { // Allow 200ms lateness
                    @Override
                    public long extractTimestamp(TradeData element) {
                        return (long) (element.getTimestamp() * 1000); // Assuming timestamp is in seconds
                    }
                }
            );

        // Example: Calculate average price per ticker over 1-second tumbling windows
        DataStream<String> avgPriceStream = tradeStream
            .keyBy(TradeData::getTicker)
            .window(TumblingEventTimeWindows.of(Time.seconds(1)))
            .apply((WindowFunction<TradeData, String, String, TimeWindow>) (key, window, input, out) -> {
                double sum = 0;
                int count = 0;
                for (TradeData t : input) {
                    sum += t.getPrice();
                    count++;
                }
                double avg = (count > 0) ? sum / count : 0;
                out.collect(String.format("Ticker: %s, Window: %s-%s, AvgPrice: %.2f",
                    key, window.getStart(), window.getEnd(), avg));
            });
        
        // avgPriceStream.print(); // For debugging

        // ... actual anomaly detection logic would go here

        env.execute("HFT Anomaly Detection");
    }

    // Simple POJO for TradeData
    public static class TradeData {
        private double timestamp;
        private String ticker;
        private double price;
        private int volume;

        // getters and setters...
        public double getTimestamp() { return timestamp; }
        public void setTimestamp(double timestamp) { this.timestamp = timestamp; }
        public String getTicker() { return ticker; }
        public void setTicker(String ticker) { this.ticker = ticker; }
        public double getPrice() { return price; }
        public void setPrice(double price) { this.price = price; }
        public int getVolume() { return volume; }
        public void setVolume(int volume) { this.volume = volume; }
    }
}
```
Choosing tumbling windows of 1 second felt like a good starting point for aggregating features that could then be fed into the anomaly detection models. I considered sliding windows for smoother transitions, but the computational overhead seemed higher, and for a first pass, tumbling windows were simpler to reason about.

### The Anomaly Detection Models: Prophet and Isolation Forest

This was the core of the project. I decided to experiment with two different models: **Facebook Prophet** for its time-series forecasting capabilities and **Isolation Forest** for its general-purpose anomaly detection strengths.

**Prophet:**
My idea with Prophet was to predict an "expected" price range for each ticker and then flag trades that deviated significantly. The tricky part was that Prophet is fundamentally a batch prediction model, not designed for per-event streaming predictions. My initial thought of re-training and predicting for every single incoming trade was obviously not feasible.

So, I adopted a pragmatic approach: I would periodically (e.g., every 15 minutes or every hour) re-train a Prophet model for each active ticker using the recent historical data from Flink (perhaps aggregated and stored temporarily). Then, I'd use this model to forecast the expected price range for the next short interval (e.g., the next minute). These forecast boundaries (yhat_lower, yhat_upper) would then be broadcasted or made available to the Flink job, which would use them as dynamic thresholds.

A conceptual Flink `RichFlatMapFunction` might look something like this (this is simplified, the model loading/updating would be more complex):

```java
// Pseudo-code concept for using Prophet bounds
public class ProphetThresholdFunction extends RichFlatMapFunction<TradeData, AnomalyAlert> {
    private Map<String, PriceBound> currentPriceBounds; // Ticker -> {lower, upper}

    @Override
    public void open(Configuration parameters) throws Exception {
        // In a real scenario, this would load bounds, perhaps from a broadcast stream
        // or an external store updated by the Prophet training process.
        currentPriceBounds = new HashMap<>(); 
        // For example, pre-populate or fetch initial bounds
        // currentPriceBounds.put("ALPHA", new PriceBound(100.0, 110.0)); 
    }

    @Override
    public void flatMap(TradeData trade, Collector<AnomalyAlert> out) throws Exception {
        PriceBound bounds = currentPriceBounds.get(trade.getTicker());
        if (bounds != null) {
            if (trade.getPrice() < bounds.getLower() || trade.getPrice() > bounds.getUpper()) {
                out.collect(new AnomalyAlert(trade.getTicker(), "Prophet", "Price out of expected range", trade.getPrice()));
            }
        }
        // Logic to update currentPriceBounds would be needed, perhaps via a side input or broadcast state
    }
}
```
This was a compromise. It wasn't truly "Prophet predicting on the stream," but rather "stream data being checked against Prophet's periodically updated forecasts."

**Isolation Forest:**
For Isolation Forest, I envisioned it catching more general anomalies that Prophet might miss, especially those involving multiple features (e.g., unusual price *and* volume combinations). Flink ML did have some algorithms, but at the time I was working on this, I didn't find a direct Isolation Forest implementation that fit easily.

My approach here was to train an Isolation Forest model offline (using Python's scikit-learn) on a sample of historical trade features (like price changes, volume changes, spread, etc., engineered within Flink). Then, the challenge was how to apply this pre-trained model in Flink.

One option was to try and re-implement the prediction logic in Java, but that seemed error-prone. Another was to call out to a Python process, but that adds latency and complexity. I eventually settled on a simplified path for this student project: I focused on feature extraction in Flink. These features could then be scored. For a "live" feel in my tests, I would sometimes load a very lightweight model or a set of rules derived from the Isolation Forest directly into my Flink job.

For example, if Flink prepared a `FeatureVector` Pojo, a function could apply a simplified scoring logic.
```java
// In Flink, after feature engineering for Isolation Forest
public class IsolationForestScorer extends RichMapFunction<FeatureVector, AnomalyScore> {
    // private transient SomePreTrainedIsolationForestModel model; // Ideally load a serialized model

    @Override
    public void open(Configuration parameters) throws Exception {
        // Load the pre-trained model here. This was a pain point.
        // I experimented with serializing a simple model or its key parameters.
        // For instance, if it was a very simple tree structure, I could hardcode it,
        // which is not ideal but was a way to get things working for a demo.
        // For example: model = loadModelFromFile("/path/to/model.ser");
    }

    @Override
    public AnomalyScore map(FeatureVector features) throws Exception {
        // double score = model.predict(features); // This is the ideal
        // For the sake of this example, let's assume a placeholder scoring logic
        // based on some pre-calculated feature thresholds from an offline model
        double score = 0.0; // Lower scores are more anomalous for Isolation Forest
        if (features.getNormalizedPriceChange() > 0.8 && features.getNormalizedVolumeChange() > 0.7) {
            score = -0.2; // Arbitrary anomaly score
        } else if (features.getNormalizedPriceChange() < 0.1) {
            score = -0.1;
        }
        return new AnomalyScore(features.getTicker(), "IsolationForest", score, features.getTimestamp());
    }
}
```
Getting the model into the Flink tasks was tricky. Flink's documentation on distributing user code and resources was helpful, but I recall struggling with serialization and ensuring the model was accessible by all task managers. For a simpler version, I even considered just extracting key decision paths from a few trees in the forest and implementing those as rules.

### The "95% Accuracy" - A Closer Look

Achieving a high accuracy figure like 95% needs qualification. This wasn't against a perfectly labeled, massive, real-world HFT dataset, as those are hard to come by. I created a test dataset by combining some historical data with synthetically generated anomalies (sudden price spikes/dips, unusual volume bursts). The 95% referred primarily to the **precision** in detecting these specific, predefined anomalous events during my simulations. Recall was a bit lower, meaning some more subtle anomalies were likely missed.

Tuning was a significant effort. For Prophet, it involved playing with `changepoint_prior_scale`, seasonality parameters, and how far out I forecasted. For Isolation Forest (when I trained it offline with scikit-learn), `n_estimators`, `max_samples`, and the `contamination` parameter were key. I also realized early on that raw price/volume wasn't enough. I started engineering features like:
*   Price difference from a short-term moving average.
*   Volume deviation from a rolling median.
*   Volatility spikes.

One of the breakthroughs was realizing that combining the signals from both models yielded better results than either one in isolation. A simple heuristic like "flag if Prophet residual is high AND Isolation Forest score is very low" helped reduce false positives.

### Moments of Struggle and "Aha!"

Managing state in Flink, especially with keyed streams (e.g., per-ticker states for moving averages or Prophet bounds), was complex. I distinctly remember one late night trying to debug why my windowed aggregations were off. It turned out my understanding of watermark propagation in complex job graphs was flawed. The Flink Web UI became indispensable for visualizing tasks, data flow, and especially backpressure. Seeing those "high" watermark latencies was a clue. The concept of `allowedLateness` also took a while to click, but it helped handle those trades that straggled in a bit too late for a window without discarding them immediately.

Another "aha!" moment was when I started sketching out the data flow on paper. With Kafka, multiple Flink operators, different types of windowing, and then the two models, it was easy to get lost. Visualizing how `TradeData` transformed into `Features`, then into `AnomalyScores` or `Alerts`, and how state was maintained at each step, really clarified things.

Debugging distributed systems is just inherently hard. Sometimes a task would fail, and the Flink logs, while detailed, could be a torrent of information. Learning to filter them effectively and understanding the lifecycle of a Flink job and its tasks was a steep learning curve. I often resorted to scaling down my local Flink cluster to just one task manager and one slot to simplify debugging before trying to run it in a more distributed fashion.

### What I Learned and What's Next

This project was an incredible learning experience. I went from a theoretical understanding of Kafka and Flink to having practical, hands-on experience with their quirks and powers. Debugging stream processing logic, thinking about event time and state, and trying to integrate ML models into a streaming pipeline – these were all areas where I learned a ton.

If I were to continue this, I'd focus on:
1.  **More Sophisticated Feature Engineering:** Incorporating order book data, if available, could provide much richer features.
2.  **Flink ML Libraries:** Dive deeper into FlinkML or libraries like Alink to see if more models can be implemented directly within Flink for better performance and state management.
3.  **Model Management and Deployment:** A more robust way to update and deploy the Prophet and Isolation Forest models without downtime. Perhaps using Flink’s broadcast state more effectively for model parameters.
4.  **Complex Event Processing (CEP):** Flink’s CEP library looks very promising for defining more sophisticated anomaly patterns based on sequences of events, rather than just single-point deviations.
5.  **Scalability and Performance Testing:** Really pushing the system with a much larger volume of data to identify and resolve bottlenecks.

It was definitely ambitious, and the "real-time" aspect with models like Prophet required some practical compromises. But as a learning exercise in building an end-to-end streaming data application, it was immensely rewarding.