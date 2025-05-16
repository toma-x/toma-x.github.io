---
layout: post
title: Real-time Anomaly Detection Engine
---

## Building a Real-time Anomaly Detection Engine for Financial Data

This semester, I decided to dive deep into a project that combined a few areas I’ve been keen on exploring: real-time data processing, machine learning, and financial markets. The goal was to build an anomaly detection system that could analyze streaming financial tick data. It’s been quite a journey, with a fair share of headaches and a few moments of actual triumph.

### The Spark: Why Anomaly Detection in Tick Data?

The idea came about after a guest lecture in my "Data Mining" course. We touched upon fraud detection and how unusual patterns can signify critical events. Financial tick data, with its high velocity and volume, seemed like a perfect candidate for applying these concepts. Identifying sudden, anomalous price or volume movements in real-time could, in theory, signal market manipulation, system glitches, or the initial ripples of a major news event. This project felt like a practical way to apply machine learning to a dynamic dataset.

### Getting Off the Ground: Python and Initial Design Thoughts

I decided to build the system primarily in Python. My familiarity with it from other coursework, plus the rich ecosystem of libraries like Pandas, NumPy, and Scikit-learn, made it a natural choice. For the machine learning part, I knew I wanted to eventually leverage cloud capabilities for training, and I’d heard good things about Google Cloud's Vertex AI from a senior who used it for their capstone.

My initial plan for handling "streaming" data was pretty basic. I wasn't ready to tackle something like Kafka or Spark Streaming right out of the gate for a personal project. I figured I could simulate a stream by reading data line-by-line from a CSV file, or perhaps have a script appending to a file that my main engine would monitor. This felt manageable for a first iteration.

For anomaly detection, I initially considered statistical methods like Z-score or ARIMA-based approaches. While robust, I was curious about using a neural network, specifically an autoencoder. The concept of training a model to reconstruct "normal" data and then flagging data points with high reconstruction errors as anomalous seemed really elegant.

### Wrestling with Data: Ingestion and Preprocessing

The first major hurdle was getting and preparing the data. I found a publicly available dataset of mock Forex tick data. It wasn't perfect – it had some gaps and the occasional malformed line, but it was good enough to start with.

Each tick generally had a timestamp, symbol, bid price, and ask price. I decided to focus on price movements and volume (though the initial dataset only had price, so I had to synthesize volume for some experiments, or simplify my features).

My preprocessing pipeline, built with Pandas, looked something like this:
1.  Load the data, parse timestamps.
2.  Calculate features: I started simple with price changes (deltas) and moving averages over very short windows (e.g., 5-10 ticks).
3.  Normalize the features: Neural networks generally prefer input data to be scaled, so I used `MinMaxScaler` from Scikit-learn. This was crucial, and I remember spending a good hour debugging why my initial model wasn't converging, only to realize I'd forgotten to scale one of the new features I added.

Here’s a snippet of how I was handling new incoming data points in my simulation. It's not the most optimized, but it got the job done for development:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assume scaler is already fit on some historical training data
# scaler = MinMaxScaler()
# scaler.fit(historical_data[['price_diff', 'short_ma']])

# a small buffer to calculate moving averages for incoming ticks
tick_buffer = []
WINDOW_SIZE = 10 # for moving average

def preprocess_tick(tick_data_dict):
    global tick_buffer
    # tick_data_dict is like {'timestamp': 1678886400.0, 'price': 1.0800}

    current_price = tick_data_dict['price']
    
    if not tick_buffer: # first tick
        tick_buffer.append(current_price)
        return None # not enough data yet

    prev_price = tick_buffer[-1]
    price_diff = current_price - prev_price
    
    tick_buffer.append(current_price)
    if len(tick_buffer) > WINDOW_SIZE:
        tick_buffer.pop(0) # keep buffer size constrained

    if len(tick_buffer) < WINDOW_SIZE: # not enough for MA
        return None 

    # calculate a simple moving average
    short_ma = sum(tick_buffer) / len(tick_buffer)
    
    # create a DataFrame for scaling - this felt a bit clunky
    # but made it consistent with how I trained the scaler
    feature_df = pd.DataFrame([[price_diff, short_ma]], columns=['price_diff', 'short_ma'])
    
    # scaled_features = scaler.transform(feature_df)
    # for now, just returning the raw features, scaling would happen before model input
    # return scaled_features 
    return {'price_diff': price_diff, 'short_ma': short_ma, 'original_price': current_price}

# Example usage for a new tick:
# new_tick = {'timestamp': 123456789.0, 'price': 1.12345}
# processed_features = preprocess_tick(new_tick)
# if processed_features:
#     print(f"Processed features: {processed_features}")
```
I remember the `tick_buffer` logic took a few tries to get right, especially handling the initial ticks before the buffer was full. Forgetting to pop from the buffer initially led to it growing indefinitely and my moving averages being skewed. Classic mistake.

### Into the Cloud: Model Training with Vertex AI

This was the part I was most excited and nervous about. I decided to build an autoencoder using TensorFlow and Keras. My architecture was fairly simple: an input layer, a few dense encoding layers, a bottleneck layer, a few dense decoding layers, and an output layer. The goal was for the output layer to reconstruct the input features.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_autoencoder(input_dim, encoding_dim=16):
    #
    # input_dim is the number of features
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded) # Bottleneck

    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded) # Sigmoid because inputs are scaled 0-1

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse') # Mean Squared Error for reconstruction
    return autoencoder

# Later, when I had my preprocessed training data (X_train_scaled)
# INPUT_DIM = X_train_scaled.shape
# ae_model = build_autoencoder(INPUT_DIM)
# print(ae_model.summary())
```
Getting the `input_dim` right and ensuring the final activation was `sigmoid` (because my `MinMaxScaler` scaled features to [0,1]) were small but crucial details. I initially used `relu` on the output layer by mistake, and the loss just wouldn't go down meaningfully.

Training this locally was fine for small datasets, but I wanted to learn Vertex AI. The process involved:
1.  Writing my TensorFlow training script.
2.  Containerizing it with Docker. This was a bit of a learning curve. I found a helpful guide on the Google Cloud documentation for structuring the Dockerfile for custom training.
3.  Uploading my training data (a CSV of preprocessed normal tick sequences) to a Google Cloud Storage bucket.
4.  Using the `gcloud ai custom-jobs create` command to submit the training job to Vertex AI.

The first few attempts failed. I had issues with package versions in my `requirements.txt` for the Docker container. Then I had a path issue where my script couldn't find the data from the GCS bucket. The Vertex AI logs were my best friend here, even if sometimes cryptic. I recall a "permission denied" error that took me ages to solve, eventually realizing my Vertex AI service account needed "Storage Object Viewer" IAM role on the bucket. A StackOverflow thread about Vertex AI IAM roles finally pointed me in the right direction.

After a successful training run, I saved the model (the `SavedModel` format) to my GCS bucket.

### The Engine Room: Real-time Anomaly Detection Logic

With a trained autoencoder, the next step was to use it. The core idea is to feed new, preprocessed tick data into the autoencoder and calculate the reconstruction error (Mean Squared Error between input and output). If this error exceeds a certain threshold, the tick is flagged as an anomaly.

```python
import numpy as np

# ae_model would be loaded from the trained model artifact
# threshold would be determined empirically from a validation set

def is_anomaly(model, data_point_scaled, threshold):
    #
    # data_point_scaled needs to be reshaped for the model
    # it expects a batch, even if it's a batch of 1
    data_point_reshaped = np.reshape(data_point_scaled, (1, data_point_scaled.shape))
    reconstruction = model.predict(data_point_reshaped)
    mse = np.mean(np.power(data_point_scaled - reconstruction, 2))
    
    # print(f"Input: {data_point_scaled}, Reconstruction: {reconstruction}, MSE: {mse}") # for debugging
    if mse > threshold:
        return True, mse
    return False, mse

# Example:
# Assume `new_scaled_features` is the output from preprocessing and scaling a new tick
# anomaly_status, error_val = is_anomaly(ae_model, new_scaled_features, ANOMALY_THRESHOLD)
# if anomaly_status:
#     print(f"ANOMALY DETECTED! MSE: {error_val}")
```
Determining the `ANOMALY_THRESHOLD` was an iterative process. I ran my model over a validation set of "normal" data, calculated their reconstruction errors, and then chose a threshold that was, say, above the 99th percentile of these errors. This was a bit of trial and error. Too low, and I got too many false positives. Too high, and I missed actual (simulated) anomalies.

For the "streaming" part, I ended up writing a Python script that would `tail -f` a CSV file. New lines appended to this CSV (by another dummy script simulating a data feed) would be read, preprocessed, and passed to the `is_anomaly` function. It’s not enterprise-grade, but it served its purpose for the project.

### Deployment to a Unix Server

I have a small Linux VM (Ubuntu) that I use for projects, so that became my "Unix server." Deployment involved:
1.  `scp`-ing my Python scripts, the trained model file (downloaded from GCS), and the fitted `MinMaxScaler` object (saved using `joblib`) to the server.
2.  Setting up a Python virtual environment (`python3 -m venv env`) and installing dependencies from `requirements.txt`. This hit a snag because the server had an older version of `pip` that struggled with some `tensorflow` dependencies until I upgraded `pip` itself within the venv.
3.  Running the main detection script. Initially, I just ran it in the foreground. For longer tests, I used `nohup python main_detector.py &` to keep it running after I logged out of SSH. Using `screen` was also an option I considered for easier management.

One issue I ran into was file paths. My local paths were different from the server paths, so I had to make sure my scripts used relative paths or configurable absolute paths for loading the model and scaler.

### Moments of Frustration and Little Victories

There were plenty of times I felt stuck.
*   **NaNs are the Enemy:** For a while, my preprocessing pipeline was mysteriously producing `NaN` values. This, of course, made TensorFlow very unhappy. It turned out to be an edge case in my moving average calculation when the input data had consecutive identical prices, leading to a zero division somewhere if not handled carefully or if a feature relied on variance. Lots of `print()` statements and `pdb` (Python debugger) sessions helped track that down.
*   **Vertex AI IAM Maze:** As mentioned, the IAM permissions for Vertex AI to access GCS buckets. The error messages weren't always super clear about *which* service account needed *which* permission. Reading through Google Cloud documentation on service accounts and roles was essential, but felt like navigating a maze at times.
*   **The Threshold Balancing Act:** Setting the anomaly threshold felt more like an art than a science initially. I spent a lot of time plotting reconstruction errors and trying to find that sweet spot. I remember reading a blog post – can't recall where now, probably towardsdatascience.com – about dynamic thresholding, which is something I'd explore if I had more time.
*   **The Breakthrough:** The biggest "aha!" moment was seeing the first true positive anomaly get flagged correctly based on a spike I manually inserted into my test data stream. After all the setup, coding, and debugging, seeing the `ANOMALY DETECTED!` message with a high MSE was incredibly satisfying.

### Where It Stands and What's Next

The system, as it is now, can process a simulated stream of tick data, preprocess it, feed it to a Vertex AI-trained autoencoder, and flag anomalies based on reconstruction error. It’s a good proof-of-concept.

Limitations are clear:
*   The "streaming" is very basic.
*   The feature engineering is simple.
*   The anomaly threshold is static.
*   Error handling and robustness could be significantly improved.

If I were to continue developing this, I'd look into:
*   Using a proper streaming framework like Apache Kafka and Flink/Spark Streaming.
*   More sophisticated feature engineering, perhaps incorporating time-based features or features from multiple correlated symbols.
*   Exploring more advanced models or ensemble methods. Recurrent Autoencoders (using LSTMs) could be interesting for capturing temporal dependencies better.
*   Implementing dynamic thresholding.
*   Building a simple dashboard to visualize the anomalies.

### Final Thoughts

This project was a massive learning experience. From the nitty-gritty of data preprocessing and model building in Python to navigating the complexities of a cloud ML platform like Vertex AI and finally deploying it on a simple server, every step had its own set of challenges. It solidified my understanding of machine learning workflows and gave me a real appreciation for the engineering that goes into building even a relatively simple "real-time" system. Definitely one of the more demanding but rewarding projects I’ve tackled.