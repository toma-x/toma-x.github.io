---
layout: post
title: Real-Time Market Anomaly Detection
---

## Detecting Anomalies in High-Frequency FX Data Streams: A Deep Dive

This project has been a significant undertaking, pushing my understanding of stream processing and deep learning for time-series analysis. The goal was to build a system capable of identifying anomalies in real-time foreign exchange (FX) market data. Given the sheer velocity and volume of high-frequency FX data, this presented some interesting engineering challenges from the get-go. My chosen stack involved Kafka for data ingestion, Flink for stream processing, and Python with TensorFlow for the anomaly detection model itself, specifically an LSTM autoencoder.

### The Data Onslaught: Kafka for Ingestion

The first hurdle was simply getting a handle on the data. High-frequency FX data doesn't just trickle in; it’s a torrent. I needed a robust system that could queue and manage this flow without data loss. Kafka seemed like the natural choice here due to its high-throughput capabilities and fault tolerance. I'd read about its distributed nature and ability to handle massive message volumes, which felt essential for something like FX tick data.

Setting up Kafka locally wasn't too bad, a few Zookeeper and server property tweaks. My initial Python producer script was basic, just sending simulated FX tick data (timestamp, currency pair, bid, ask) as JSON strings to a Kafka topic named `fx-stream`. The consumer side, which would eventually be Flink, initially started as another simple Python script just to verify messages were flowing correctly. One early snag was message serialization; I initially forgot to encode my strings to bytes before sending to Kafka, which led to some cryptic errors until I figured that out. Standard stuff, but time-consuming when you're staring at a non-descriptive `SerializationException`.

### Real-Time Processing with Apache Flink

With data flowing into Kafka, the next step was processing it in real-time. I considered Spark Streaming, but Flink’s reputation for true stream processing with lower latency and its robust event-time processing capabilities swayed me. For anomaly detection, especially in financial markets, processing events as they *occur*, not as they arrive at the processor, felt critical.

My Flink application was written using PyFlink, as I wanted to keep the language consistent across the project, especially with the planned TensorFlow integration. The core idea was to consume from the `fx-stream` Kafka topic, parse the JSON, and then prepare it for the anomaly detection model.

Here’s a snippet of how I set up the Flink environment and Kafka source:

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
import json

env = StreamExecutionEnvironment.get_execution_environment()
# Set parallelism, checkpointing etc. - learned the hard way this is important
env.set_parallelism(1) # Kept it simple for local dev

kafka_props = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fx-anomaly-detector-group'
}

# Define the Kafka source
kafka_source = FlinkKafkaConsumer(
    topics='fx-stream',
    deserialization_schema=SimpleStringSchema(),
    properties=kafka_props
)

# Add the source to the environment
raw_data_stream = env.add_source(kafka_source)

# Simple parsing - this evolved a bit
def parse_fx_data(json_string):
    try:
        data = json.loads(json_string)
        # Extracting mid-price for EUR/USD, assuming this structure
        # In reality, I had to handle different pairs and ensure data['bid'] and data['ask'] existed
        if data.get('pair') == 'EUR/USD':
            mid_price = (float(data['bid']) + float(data['ask'])) / 2.0
            timestamp = int(data['timestamp']) # Assuming unix timestamp
            return (timestamp, mid_price)
        else:
            return None # Filter out other pairs for now
    except Exception as e:
        # print(f"Error parsing: {json_string}, error: {e}") # For debugging
        return None

# Process the stream
processed_stream = raw_data_stream.map(parse_fx_data, output_type=Types.TUPLE([Types.INT(), Types.FLOAT()])) \
                               .filter(lambda x: x is not None)
```

One of the tricky parts with Flink was understanding its windowing mechanisms. For time-series anomaly detection, looking at individual data points isn't enough; you need context. I experimented with tumbling windows initially, say, collecting 60 seconds of mid-prices. The idea was to feed sequences of these prices into the LSTM. Debugging Flink jobs, especially when they involve user-defined functions and external libraries, was also a learning curve. The Flink dashboard helped, but often it came down to careful logging and simplifying the job to isolate issues. I remember a particularly frustrating afternoon trying to figure out why my map function wasn't emitting anything, only to realize it was a silly type error in how I was handling the output of `json.loads()`.

### The Heart of Detection: LSTM Autoencoders with TensorFlow

For the anomaly detection model itself, I chose an LSTM autoencoder. LSTMs are well-suited for sequential data like time series, and autoencoders are great for learning a compressed representation of "normal" data. The idea is that if the model is trained only on normal FX price sequences, it will achieve low reconstruction error for similar normal sequences. However, when an anomalous sequence comes along, the model will struggle to reconstruct it accurately, leading to a high reconstruction error, flagging it as an anomaly.

I used TensorFlow with Keras to build the model. The architecture wasn't overly complex, as I was constrained by my local machine's training capabilities.

Here’s roughly what the model definition looked like:

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed

# Define constants based on my data
# I decided on 60 data points per sequence, representing 1 minute if data is per second
TIME_STEPS = 60
N_FEATURES = 1 # Just the mid-price for now

# Define the autoencoder
inputs = Input(shape=(TIME_STEPS, N_FEATURES))

# Encoder
# Reduced units due to training time constraints on my laptop
encoded = LSTM(64, activation='relu', return_sequences=False)(inputs)
encoded = RepeatVector(TIME_STEPS)(encoded) # Repeats the context vector for each time step

# Decoder
decoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
decoded = TimeDistributed(Dense(N_FEATURES))(decoded) # Apply dense layer to each time step output

autoencoder_model = Model(inputs, decoded)
autoencoder_model.compile(optimizer='adam', loss='mse')

# autoencoder_model.summary() # Useful for checking layers
```

Training this required a dataset of "normal" FX behavior. This was a challenge in itself – what is truly "normal" in a chaotic market? I took a period of historical data that was relatively stable, without major news events, and used that to generate training sequences. Each sequence was a series of `TIME_STEPS` consecutive mid-prices, normalized using `MinMaxScaler` fit on the training data. Normalization is super important for neural networks; without it, the training process can be very unstable or slow. I spent a good amount of time ensuring the scaling was applied correctly during training and then, crucially, that the *same* scaling parameters were used for live data.

A breakthrough moment here was realizing the importance of the `RepeatVector` layer. Initially, my decoder wasn't performing well, and I was getting confused about how the context from the encoder (a single vector if `return_sequences=False`) was being fed to the decoder LSTM which expects a sequence. The `RepeatVector` essentially tiles this context vector to match the input sequence length for the decoder, which significantly improved the reconstruction. I recall finding a Keras documentation example or a StackOverflow answer that clarified this specific pattern for LSTM autoencoders.

### Integrating Flink and the TensorFlow Model

This was probably the most complex part: getting the Flink stream processing job to use the trained TensorFlow model. My Flink job was windowing the incoming FX data into sequences (e.g., of 60 data points). Each of these sequences then needed to be fed to the `autoencoder_model.predict()` method.

I decided to use a `RichMapFunction` in Flink. The `open()` method of a `RichMapFunction` is called once when the task starts, so it was the perfect place to load the trained TensorFlow model. This avoids reloading the model for every single data point or window, which would be terribly inefficient.

```python
from pyflink.datastream import RichMapFunction

class LSTMPredictor(RichMapFunction):
    def __init__(self, model_path, time_steps, scaler_params): # Pass scaler params too!
        self.model_path = model_path
        self.time_steps = time_steps
        self.scaler_params = scaler_params # Dict with 'min' and 'scale' for MinMaxScaler
        self.model = None
        # We need a way to scale individual windows, not just a global scaler
        # For simplicity here, let's assume scaler_params are pre-calculated
        # In a real scenario, you'd likely receive scaler params or fit them on training data

    def open(self, runtime_context):
        # This is where we load the model
        self.model = tf.keras.models.load_model(self.model_path)
        # Initialize a simple MinMax scaler manually based on pre-saved params
        # from training. This is a bit simplified for the example.
        self.data_min_ = self.scaler_params['min']
        self.data_scale_ = self.scaler_params['scale']


    def scale_sequence(self, sequence):
        # Manual scaling - (X - X_min) / (X_max - X_min) * feature_range_max + feature_range_min
        # Or if using sklearn's MinMaxScaler: (X - data_min_) * data_scale_
        # Assuming sequence is a list or 1D numpy array
        # This needs to be robust.
        scaled_sequence = (np.array(sequence) - self.data_min_) * self.data_scale_
        return scaled_sequence.reshape(1, self.time_steps, N_FEATURES)

    def map(self, fx_sequence_window):
        # fx_sequence_window is a list of mid-prices from the Flink window
        if len(fx_sequence_window) != self.time_steps:
            # Handle sequences not matching the exact length, maybe log or ignore
            return (fx_sequence_window, None, "Sequence length mismatch")

        # 1. Scale the incoming window like the training data
        # This was a major pain point: ensuring the scaling applied here
        # exactly matched the scaling used during training.
        # Forgetting this led to awful reconstruction errors initially.
        scaled_sequence = self.scale_sequence(fx_sequence_window)

        # 2. Get reconstruction from the model
        reconstructed_sequence = self.model.predict(scaled_sequence)

        # 3. Calculate reconstruction error (MSE)
        mse = np.mean(np.power(scaled_sequence - reconstructed_sequence, 2), axis=(1,2)) # Mean over time steps and features
        reconstruction_error = mse # mse will be an array with one value

        return (fx_sequence_window, float(reconstruction_error)) # Return original data and its error
```
*Disclaimer: The scaling part in `LSTMPredictor` is a bit hand-wavy here. In a full implementation, you'd need to save the `MinMaxScaler` object from training (e.g., using `joblib`) or its parameters (`min_`, `scale_`) and re-apply the exact transformation. I spent a lot of time getting this right, as even small differences in scaling can throw off the model's predictions.*

One of the non-trivial aspects here was managing dependencies. PyFlink runs Python UDFs in separate Python processes, so I had to ensure that the TensorFlow library and my saved model file were available in the Python environment used by Flink's TaskManagers. For local testing, this was manageable by having everything in the same virtual environment.

### Defining "Anomaly" and the 95% Accuracy Claim

Once I had the reconstruction error for each incoming window, the next step was to decide what constituted an "anomaly." This meant setting a threshold. If the reconstruction error for a window exceeded this threshold, it was flagged.

Determining this threshold was more art than science initially. I ran the model over a validation set of normal data (data not used for training but considered normal) and looked at the distribution of reconstruction errors. I then picked a threshold based on, say, the 99th percentile of these errors. This was an iterative process.

The "95% accuracy" figure requires context. True labeled anomaly data for high-frequency FX is scarce. For evaluation, I generated some synthetic anomalies by injecting spikes or sudden level shifts into otherwise normal sequences. I also had a small set of historical data points that were manually identified as unusual around specific market events. The 95% accuracy was achieved on this combined, somewhat artificial, test set. It means that out of 100 sequences (both normal and my synthetic/identified anomalies), the system correctly classified 95 of them. It's important to note that there were false positives (normal data flagged as anomalous) and false negatives (anomalies missed). Reducing these is an ongoing area of refinement. Getting good labeled data for financial anomalies is a perennial problem.

One specific forum post I remember looking up (though I can't find the exact link now) was about common pitfalls in evaluating anomaly detection on time series, especially the risk of "trivial" detections if your anomalies are too obvious or if your "normal" data is too clean. It made me think more carefully about how I was simulating anomalies.

### Challenges and Reflections

This project was a steep learning curve.
*   **Data Synchronization:** Ensuring that the timestamps were handled correctly across Kafka, Flink (especially with event time and watermarks), and the model was crucial. Misaligned data could easily lead the model astray.
*   **Resource Management:** Training LSTMs, even moderately sized ones, takes time and computational resources. I had to be patient and often ran training jobs overnight on my somewhat limited local machine.
*   **Hyperparameter Tuning:** Finding the right number of LSTM units, `TIME_STEPS`, learning rate, etc., was iterative. I didn't have the setup for extensive hyperparameter optimization grids, so it was more manual, guided by papers and trial-and-error.
*   **Flink's Python API Nuances:** While powerful, PyFlink sometimes felt less documented or had fewer community examples for complex scenarios compared to its Java/Scala counterparts, especially when integrating with heavyweight libraries like TensorFlow. Serializing models or large objects to be used in UDFs needed careful consideration. I recall a StackOverflow thread discussing strategies for using large ML models in PyFlink UDFs, debating between loading in `open()` versus broadcasting, which reinforced my `RichMapFunction` approach for this case.

One of the biggest "aha!" moments was when, after struggling with poor model performance, I revisited my data normalization and preprocessing pipeline end-to-end. I found a subtle inconsistency in how I was scaling the data between training and what I *thought* I was doing in the Flink pipeline. Fixing that made a world of difference. It underscored how critical meticulous data handling is in any ML project, but especially for sensitive models like LSTMs.

### Future Directions

While I'm pleased with the current state, there's always more to do.
*   **More Sophisticated Thresholding:** Instead of a static threshold, an adaptive one that changes based on recent volatility could be more robust.
*   **Multivariate Analysis:** The current model only looks at the mid-price of one currency pair. Extending it to include other features (e.g., volume, bid-ask spread, prices of correlated pairs) could make it much more powerful. This would mean changing `N_FEATURES` and adjusting the input data accordingly.
*   **Online Learning:** The model is trained offline. Exploring techniques for online or incremental learning, where the model can adapt to evolving market dynamics without full retraining, would be a fascinating next step.
*   **Better Evaluation:** Getting access to more robustly labeled anomaly datasets or developing more sophisticated unsupervised evaluation metrics would be key to truly understanding the model's performance in the wild.

Overall, this project has been an incredible learning experience, combining stream processing with deep learning to tackle a genuinely challenging problem. The path from raw data in Kafka to an actionable anomaly signal from a TensorFlow model running within Flink was complex but very rewarding to build.