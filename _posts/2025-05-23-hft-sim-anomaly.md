---
layout: post
title: Real-Time Anomaly Detection in Simulated HFT Data
---

## Real-Time Anomaly Detection in Simulated HFT Data: A Deep Dive

This project has been a journey. What started as an interest in financial markets and a desire to apply some machine learning concepts quickly spiraled into a full-blown stream processing challenge. I wanted to see if I could detect anomalies in High-Frequency Trading (HFT) data in something close to real-time. Here’s how it went, the hurdles I faced, and what I managed to build.

### The Goal: Spotting the Odd Ones Out in Financial Data Streams

The core idea was to process a simulated stream of HFT data – think rapid price and volume updates – and flag suspicious patterns. These could be sudden price spikes, unusual trading volumes, or anything that deviates from the "normal" behavior of the market. Given the speed and volume of HFT data, this meant building something efficient.

### Initial Thoughts on Architecture

I knew I'd need a few key components:
1.  A way to simulate HFT data (since I don't have access to a real exchange feed!).
2.  A messaging queue to handle the data stream – Kafka seemed like the standard choice here.
3.  A stream processor that could consume data from Kafka.
4.  An anomaly detection model.
5.  A way to deploy and integrate all of this.

I decided to try a hybrid approach for the stream processor, using Rust for its performance in handling raw data throughput from Kafka and Python for its strong machine learning ecosystem, specifically for deploying the anomaly detection model.

### Simulating HFT Data

First, I needed data. I wrote a simple Python script to generate tick data – timestamp, price, and volume. To make things interesting, I programmed it to occasionally inject anomalies: sudden, unrealistic price jumps or drops, or massive volume spikes that shouldn't normally occur. This wasn't super sophisticated, mostly `numpy.random` with some conditional logic to create these outliers, but it gave me something to test with. The data was serialized as JSON strings.

### The Backbone: Kafka

Setting up Kafka locally was an experience. I used the official Docker images, which simplified things a bit, but understanding topics, partitions, producers, and consumers took some time. I settled on a single topic, `hft-data-stream`, for the simulated trades. The main challenge early on was just ensuring my producer script was correctly sending messages and that I could consume them with basic Kafka command-line tools. Lots of `kafka-console-consumer.sh` debugging!

### The Stream Processor: A Tale of Two Languages (Rust and Python)

This was the most complex part. I wanted Rust to handle the initial Kafka consumption because I'd read about its performance benefits for I/O-bound tasks and its robust `rdkafka` crate. The plan was for Rust to grab messages, do some very basic validation (e.g., check if the JSON parsing works), and then pass the relevant data to a Python process for the actual anomaly detection using the PyTorch model.

**Rust: The Kafka Consumer**

Working with `rdkafka` was a learning curve. The asynchronous nature and the configuration options were a bit daunting at first. My initial consumer was very basic:

```rust
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{StreamConsumer, Consumer};
use rdkafka::message::Message;
use std::time::Duration;
use zeromq;

async fn run_rust_consumer() {
    let group_id = "hft_anomaly_detector_group";
    let bootstrap_servers = "localhost:9092";
    let topic = "hft-data-stream";

    let consumer: StreamConsumer = ClientConfig::new()
        .set("group.id", group_id)
        .set("bootstrap.servers", bootstrap_servers)
        .set("enable.partition.eof", "false")
        .set("session.timeout.ms", "6000")
        .set("enable.auto.commit", "true")
        //.set("auto.offset.reset", "earliest") // Decided against this for live processing
        .create()
        .expect("Consumer creation failed");

    consumer
        .subscribe(&[topic])
        .expect("Can't subscribe to specified topic");

    // Set up ZeroMQ publisher to send data to Python
    let zmq_context = zeromq::Context::new();
    let zmq_publisher = zmq_context.socket(zeromq::PUB).expect("Failed to create ZMQ publisher");
    zmq_publisher.bind("tcp://127.0.0.1:5555").expect("Failed to bind ZMQ publisher");

    println!("Rust consumer started, publishing to ZMQ on tcp://127.0.0.1:5555");

    loop {
        match consumer.recv().await {
            Err(e) => eprintln!("Kafka error: {}", e),
            Ok(m) => {
                if let Some(payload_view) = m.payload_view::<str>() {
                    match payload_view {
                        Ok(payload) => {
                            // Simple validation or quick check could happen here.
                            // For now, just forward the raw payload.
                            // This was an area where I initially tried to do more complex parsing
                            // in Rust, but decided to offload it to Python to keep this part lean.
                            // My Rust skills were still developing, and JSON deserialization
                            // with varying schemas felt easier to handle flexibly in Python at the time.
                            if payload.starts_with('{') { // Basic check for JSON
                                zmq_publisher.send(payload.as_bytes(), 0).expect("ZMQ send failed");
                            } else {
                                eprintln!("Received non-JSON message: {}", payload);
                            }
                        }
                        Err(e) => {
                            eprintln!("Error viewing message payload: {:?}", e);
                        }
                    }
                }
            }
        };
    }
}

#[tokio::main]
async fn main() {
    run_rust_consumer().await;
}
```
One specific issue I remember was configuring `rdkafka` correctly regarding offset commits and group IDs to ensure messages weren't processed multiple times or lost during restarts. I spent a fair bit of time on the `rdkafka` GitHub issues page and examples. The `enable.auto.commit` setting, for instance, was something I toggled a few times. I decided to stick with `true` for simplicity in this project, accepting the small risk of reprocessing or missing a message if a crash happened exactly between processing and committing. For a real production system, I'd implement manual commits.

**Inter-Process Communication: ZeroMQ**

To get data from Rust to Python, I considered a few options: REST APIs, gRPC, or even just writing to a file. REST seemed like overkill, and gRPC added complexity with protobuf definitions I wanted to avoid for this iteration. Writing to a file felt too clunky. I landed on ZeroMQ (using the `zmq` crate in Rust and `pyzmq` in Python) with a simple PUB-SUB pattern. Rust would publish the raw JSON strings, and Python would subscribe. This seemed like a good balance of performance and simplicity. The `tcp://127.0.0.1:5555` endpoint became my little data highway.

One tricky bit with ZeroMQ was ensuring the Python subscriber connected *after* the Rust publisher had bound to the port, or handling reconnections. For this student project, I mostly just made sure to start the Rust process first.

**Python: Deserialization and Feature Engineering**

The Python component subscribed to the ZeroMQ socket, received the JSON data, and then prepared it for the model.

```python
import zmq
import json
import numpy as np
import torch
# Assume VAE_model is defined elsewhere and loaded
# from my_vae_model import VAE, load_model

# Global model variable
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vae_model = load_model('path_to_my_trained_vae.pth', input_dim=3, latent_dim=10).to(device)
# vae_model.eval() # Set to evaluation mode

# Placeholder for actual model loading and VAE class
# For this post, I'll focus on the ZMQ and data handling part.
# The model details are in the next section.
# For now, imagine vae_model is loaded and ready.
# For demonstration, let's define a placeholder function for anomaly score
def get_anomaly_score_placeholder(features):
    # In reality, this would be:
    # features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    # reconstruction, _, _ = vae_model(features_tensor)
    # loss = torch.nn.functional.mse_loss(reconstruction, features_tensor, reduction='none')
    # return loss.sum().item() # or mean
    return np.random.rand() * 10 # Placeholder value

def python_processor():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.subscribe("") # Subscribe to all messages

    print("Python processor connected to ZMQ.")

    # This threshold would be determined empirically from validation data
    anomaly_threshold = 7.5 

    while True:
        message_str = socket.recv_string()
        try:
            data_tick = json.loads(message_str)
            
            # Basic feature engineering: using price, volume, and maybe a time-based feature later.
            # For now, let's assume 'price' and 'volume' are directly available.
            # Also, I added a 'spread' if 'bid' and 'ask' were present, but my sim often just had 'price'.
            price = float(data_tick.get("price", 0.0))
            volume = float(data_tick.get("volume", 0.0))
            
            # My data simulator sometimes included bid/ask, sometimes not.
            # So I had to handle missing keys gracefully.
            bid = data_tick.get("bid")
            ask = data_tick.get("ask")
            spread = 0.0
            if bid is not None and ask is not None:
                spread = float(ask) - float(bid)

            features = np.array([price, volume, spread]) 
            # Normalization would be crucial here. I had a StandardScaler fitted on training data.
            # features_normalized = scaler.transform(features.reshape(1, -1))

            # This is where the PyTorch model would be called
            # anomaly_score = calculate_reconstruction_error(features_normalized, vae_model)
            anomaly_score = get_anomaly_score_placeholder(features) # using placeholder

            # print(f"Data: {price}, {volume}, {spread} -> Score: {anomaly_score:.4f}")
            if anomaly_score > anomaly_threshold:
                print(f"ANOMALY DETECTED! Score: {anomaly_score:.2f} Data: {data_tick}")

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {message_str}")
        except Exception as e:
            print(f"Error processing message: {e} - Data: {message_str}")

if __name__ == "__main__":
    # Need to load the actual model and scaler here
    # For example:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vae_model = VAE(input_dim=3, h_dim1=20, h_dim2=15, latent_dim=5).to(device) # Example dimensions
    # vae_model.load_state_dict(torch.load('vae_model_sim_hft_v1.pth', map_location=device))
    # vae_model.eval()
    # print("VAE model loaded.")
    # scaler = joblib.load('data_scaler_v1.pkl') # Assuming a scaler was saved during training
    # print("Scaler loaded.")
    python_processor()
```
Preprocessing the features was key. I quickly realized that feeding raw prices and volumes into a neural network wouldn't work well. I experimented with percentage changes, log returns, and standard scaling. Standard scaling (`StandardScaler` from scikit-learn) based on a sample of my simulated "normal" data seemed to give the most stable results for the VAE. I had to save this scaler and load it in the Python processor.

### The Anomaly Detector: A PyTorch Variational Autoencoder (VAE)

For the anomaly detection model, I chose a Variational Autoencoder (VAE). I'd read that VAEs are good at learning a compressed representation (the latent space) of normal data. The idea is that anomalous data points will be poorly reconstructed from this latent space, leading to a higher reconstruction error, which can then be used as an anomaly score.

I considered simpler models like Isolation Forests or One-Class SVMs, but I wanted to get more experience with PyTorch and neural networks. Plus, VAEs seemed more flexible for potentially complex patterns in HFT data, even if simulated.

**Model Architecture**
My VAE architecture was relatively simple. I didn't use a massively deep network, as my simulated data wasn't *that* complex. After some trial and error with the number of layers and neurons:
*   Input: 3 features (e.g., scaled price, scaled volume, scaled spread).
*   Encoder:
    *   Linear layer (3 -> 20 neurons) + ReLU
    *   Linear layer (20 -> 15 neurons) + ReLU
    *   Two linear layers (15 -> 5 neurons each) for mu and log_var (latent dimension of 5).
*   Decoder:
    *   Linear layer (5 -> 15 neurons) + ReLU
    *   Linear layer (15 -> 20 neurons) + ReLU
    *   Linear layer (20 -> 3 neurons) + Sigmoid (since my scaled inputs were roughly in the 0-1 range after MinMax scaling, or Tanh for standard scaled data, though I eventually used linear output and MSE loss). I ended up using a linear output layer for the decoder and relied on MSE loss, as sigmoid outputs can sometimes saturate and limit reconstruction.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_HFT(nn.Module):
    def __init__(self, input_dim, h_dim1, h_dim2, latent_dim):
        super(VAE_HFT, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc_mu = nn.Linear(h_dim2, latent_dim) # for mean
        self.fc_log_var = nn.Linear(h_dim2, latent_dim) # for log variance
        
        # Decoder
        self.fc_decode1 = nn.Linear(latent_dim, h_dim2)
        self.fc_decode2 = nn.Linear(h_dim2, h_dim1)
        self.fc_output = nn.Linear(h_dim1, input_dim) # Reconstruct original features

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # random sample from N(0, I)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_decode1(z))
        h = F.relu(self.fc_decode2(h))
        # Using a linear output layer, reconstruction quality judged by MSE loss.
        # If inputs were strictly, sigmoid could be an option here.
        return self.fc_output(h) 

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

# Loss function for VAE
def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    # MSE for reconstruction loss. Using sum instead of mean per batch item
    # felt like it gave more pronounced anomaly scores in some experiments.
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') 
    
    # KL divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + beta * kld # beta can be tuned
```

**Training Struggles and Breakthroughs**

Training the VAE was an iterative process.
1.  **Data Scaling**: Initially, I didn't scale my input features properly, and the model just wouldn't learn. Standardizing the features (to zero mean and unit variance) based on the *training data* was crucial. I used `sklearn.preprocessing.StandardScaler`.
2.  **Hyperparameter Tuning**: Learning rate, batch size, latent dimension size, and the `beta` parameter in the VAE loss (which balances reconstruction loss and KL divergence) all needed tuning. I mostly did this by hand, running lots of small experiments. A `beta` value that was too high made the reconstructions poor, while too low made the latent space less regular. I found a `beta` around 0.5 to 1.0 worked okay for my data.
3.  **Loss Function**: Understanding the VAE loss function (reconstruction error + KL divergence) took some effort. The KL divergence term acts as a regularizer on the latent space. Getting this balance right so that it learned a meaningful representation of "normal" was tricky.
4.  **Defining "Normal"**: My training data consisted only of "normal" simulated trades. The VAE learns to reconstruct these well. When an anomalous trade (which it hasn't seen) is fed in, the reconstruction error should be higher.
5.  **Threshold Setting**: After training, I fed some validation data (with known anomalies) through the VAE and calculated their reconstruction errors. This helped me set a threshold: if the error for a new data point is above this threshold, it's flagged as an anomaly. This was quite empirical. I looked at the distribution of errors for normal and anomalous data and tried to pick a threshold that gave me good recall for anomalies without too many false positives.

One specific forum post that helped me a lot with VAEs was a discussion on the PyTorch forums about implementing the reparameterization trick correctly and debugging the KL divergence term. I remember staring at the formula `0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)` and comparing it to my code multiple times.

### Integration and "Real-Time" Performance

Once the Rust consumer, ZeroMQ bridge, and Python VAE processor were built, I ran them together. The Rust component was very fast at pulling from Kafka. ZeroMQ introduced minimal latency. The main bottleneck was the PyTorch model inference in Python. For a single data point, inference was quick (milliseconds), but if data arrived very rapidly, Python's Global Interpreter Lock (GIL) could become an issue if I tried to do more complex Python-side processing concurrently. For this project's scope, processing one message at a time in the Python script was manageable with my simulated data rate.

The definition of "real-time" here is loose. It wasn't nanosecond-level HFT real-time, but it could process messages from Kafka within a few tens of milliseconds end-to-end, which felt like a decent achievement for a personal project.

### Results: Did it Work?

It did, to a reasonable extent! The system successfully flagged the more obvious anomalies I had simulated – the large price jumps and volume surges. The key was tuning the anomaly threshold based on the VAE's reconstruction error. I aimed for high recall on the anomalies, even if it meant a few more false positives. For example, if normal reconstruction errors were typically below 2.0-3.0, and my injected anomalies produced errors of 10.0+, a threshold of, say, 7.5 worked reasonably well.

The main limitations were:
*   **Simulated Data**: Real HFT data is far more complex and noisy. My anomalies were quite distinct.
*   **Model Simplicity**: A more sophisticated model might capture more subtle anomalies or adapt to changing market dynamics (concept drift). My VAE was trained once and then used statically.
*   **Thresholding**: The static threshold for anomaly detection is a bit crude. Adaptive thresholding could be an improvement.
*   **Scalability**: While Rust and Kafka are scalable, my Python VAE processor was single-threaded for inference.

### Lessons Learned and What's Next

This project was a fantastic learning experience.
*   **Rust**: Got my hands dirty with `async/await`, `rdkafka`, and error handling in Rust. It definitely has a steeper learning curve than Python, but the performance potential is clear.
*   **Python & PyTorch**: Deepened my understanding of VAEs, how to build and train them, and the importance of preprocessing.
*   **System Design**: Thinking about how different components interact, data flow, and potential bottlenecks was challenging but rewarding. Using Kafka and ZeroMQ gave me practical experience with distributed messaging.
*   **The "It Works!" Moment**: Seeing the "ANOMALY DETECTED!" message print out for data I specifically crafted to be anomalous was pretty satisfying after all the debugging.

**Future Ideas**:
*   **More Realistic Data**: Try it with more complex simulated data, or even historical tick data if I can find a good source.
*   **Advanced Models**: Explore LSTMs or Transformers within the VAE for time-series aspects, or other anomaly detection techniques.
*   **Concept Drift**: Implement mechanisms to retrain or adapt the model as the "normal" behavior of the data stream changes.
*   **Improved Deployment**: Maybe package the Python part in Docker as well, and explore something like Kafka Streams or Flink for more advanced stream processing if I were to scale this up.
*   **Better IPC**: While ZeroMQ worked, for a more robust setup, gRPC with Protobuf could be a good next step to learn, despite the initial overhead.

It's far from a production-ready system, but as a student project, it taught me an immense amount about stream processing, anomaly detection, and integrating different technologies. The journey from a simple idea to a (mostly) working system was the best part.