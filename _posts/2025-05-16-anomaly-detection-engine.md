---
layout: post
title: Real-time Anomaly Detection Engine
---

It's been a while since I've updated the blog, mostly because I've been buried in what turned out to be a pretty complex project: building a real-time anomaly detection engine for financial tick data. The idea started fairly simply – could I build something that spots unusual movements in stock prices as they happen? It turned out to be a journey with a steep learning curve, especially when it came to integrating everything.

### The Core Idea: Catching Anomalies in Financial Streams

The goal was to process a stream of financial tick data – essentially, every price change – and flag anything that looked out of the ordinary. This kind of system could be useful for algorithmic trading or just for getting a quick heads-up on weird market behavior. I decided to build it in Python because of its strong data science libraries and general ease of use for a project like this.

The initial plan was to use some kind of time-series model to learn what "normal" tick data looks like and then identify deviations. I knew I'd need something that could handle sequential data effectively.

### Model Selection and Initial Hurdles with Vertex AI

I decided to use an LSTM Autoencoder for anomaly detection. [12, 15, 26] LSTMs are good with sequences, and autoencoders are designed to reconstruct their input – the idea is that if the model is trained on normal data, it'll do a poor job reconstructing anomalous data, and the reconstruction error can be our anomaly score. [15, 26]

This is where Vertex AI came into the picture. [8, 28] I'd used it before for some smaller, more straightforward tabular models with AutoML, but this was my first real dive into custom model training on the platform. [10, 22] The documentation made it seem straightforward enough: package your training code, configure a training job, and let Vertex AI handle the rest. [9, 19]

My first big hurdle was just getting the environment right for the custom training job. Vertex AI has pre-built containers for TensorFlow, which was what I was using for the LSTM. [9, 10] But I quickly realized that managing dependencies and making sure my training script (`task.py` as they often call it in the examples) could actually find my data and write the model artifacts to the right Cloud Storage bucket was trickier than I anticipated. [18, 19]

I remember spending a good chunk of time debugging `gcloud ai custom-jobs create` commands. The official docs and a few blog posts were helpful, but a lot of it was trial and error. [10, 11] One specific issue I ran into was with the `--python-module` flag. My directory structure for the trainer package was initially a bit messy, and it took a few tries to get Vertex AI to correctly locate my `trainer.task` module.

I consulted a few resources on structuring the training application. [18] Eventually, I settled on a fairly standard layout:

```
anomaly_trainer/
├── trainer/
│   ├── __init__.py
│   ├── task.py       # Main training script
│   ├── model.py      # LSTM Autoencoder definition
│   └── utils.py      # Data preprocessing functions
└── setup.py
```

The `model.py` contained the Keras model definition. Something along these lines, though it evolved a bit:

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def create_lstm_autoencoder(timesteps, n_features):
    # Define the encoder
    inputs = Input(shape=(timesteps, n_features))
    # Enc_lstm_1 layer learns to compress the input sequence
    enc_lstm_1 = LSTM(128, activation='relu', return_sequences=True)(inputs)
    enc_lstm_2 = LSTM(64, activation='relu', return_sequences=False)(enc_lstm_1) # Only last output
    
    # The bottleneck layer, captures the compressed representation
    bottleneck = RepeatVector(timesteps)(enc_lstm_2)
    
    # Define the decoder
    dec_lstm_1 = LSTM(64, activation='relu', return_sequences=True)(bottleneck)
    dec_lstm_2 = LSTM(128, activation='relu', return_sequences=True)(dec_lstm_1)
    # Output layer reconstructs the original input shape
    output = TimeDistributed(Dense(n_features))(dec_lstm_2)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Later in task.py, I'd call this:
# TIMESTEPS = 60 # Using 1 minute of tick data (assuming 1 tick per second)
# N_FEATURES = 1 # Just using price for now
# lstm_ae_model = create_lstm_autoencoder(TIMESTEPS, N_FEATURES)
```
I initially tried a simpler architecture with fewer layers and units, but the reconstruction error on even the training data was too high. I found a few papers and some GitHub repositories that used a similar stacked LSTM approach for anomaly detection, which gave me more confidence in this direction. [12, 15, 23] One particular GitHub repo, "LSTM-Autoencoder-for-Anomaly-Detection" by BLarzalere, was a useful reference, though I had to adapt it quite a bit for my specific tick data format and Vertex AI integration. [12]

Training the model on Vertex AI was an iterative process. [9] I started with a small subset of my tick data to make sure the pipeline worked, then gradually scaled up. The ability to just change the machine type for the training job without rewriting my code was a huge plus. Monitoring the training logs in Cloud Logging was essential. I learned the hard way that not saving the model frequently enough during a long training run on a pre-emptible VM can be... frustrating.

### Dealing with Data: Streaming and Preprocessing

Financial tick data is notoriously noisy and comes in at high velocity. [1, 6] For this project, I simulated a stream using a historical dataset, but the system was designed with the idea of eventually connecting to a live feed.

The preprocessing was a significant part of the work. I had to:
1.  **Normalize the data:** Neural networks generally prefer inputs in a small range, typically 0 to 1 or -1 to 1. I used MinMaxScaler from scikit-learn.
2.  **Create sequences:** LSTMs need input in the form of sequences. I decided to use a sliding window approach, where each input sequence would be, say, 60 ticks (representing one minute if ticks were per second), and the model would try to reconstruct that sequence.

My `utils.py` had a function something like this:

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1)) # Fit this on training data only

def create_sequences(data, timesteps):
    # data is expected to be a 1D numpy array of prices
    X = []
    # scaled_data = scaler.transform(data.reshape(-1,1)) # Ensure data is 2D for scaler
    # For this project, let's assume data is already scaled and 1D for simplicity here
    # In reality, scaling would happen before this, and be fit only on training data
    for i in range(len(data) - timesteps):
        X.append(data[i:(i + timesteps)])
    return np.array(X)

# Example usage if I were processing a batch:
# raw_tick_prices = fetch_some_tick_data() [...]
# scaled_prices = scaler.fit_transform(raw_tick_prices.reshape(-1,1)).flatten() # Fit and transform
# sequences = create_sequences(scaled_prices, TIMESTEPS)
# sequences = sequences.reshape((sequences.shape, sequences.shape, N_FEATURES)) # Reshape for LSTM
```
A key moment of confusion was whether to fit the scaler on the entire dataset or just the training portion. StackOverflow and various machine learning blogs were quite clear: *only fit on the training data* and then use that *fitted* scaler to transform the validation and test sets (and future incoming data). This prevents data leakage from the test set into the training process.

Another challenge was handling the sheer volume of tick data. [1, 6] Even for backtesting, loading and preprocessing large CSV files of tick data was slow. I experimented with using `pandas` chunking to process data in smaller pieces.

### Deployment to a Unix Server and Real-time Processing

Once I had a trained model on Vertex AI, the next step was deploying it. [10, 14] I created an endpoint on Vertex AI for the model. [2, 8, 14] This was surprisingly straightforward using the `gcloud ai endpoints deploy-model` command or the Python SDK. [2, 5, 14]

The "real-time" part of the project involved a Python script running on a Unix server. This script would:
1.  Simulate receiving new tick data (in a real-world scenario, this would come from a WebSocket, a message queue like Kafka, or an API).
2.  Preprocess the latest window of ticks (normalize, create a sequence).
3.  Send this sequence to the Vertex AI endpoint for prediction. [2, 4]
4.  Get the reconstructed sequence back from the model.
5.  Calculate the Mean Squared Error (MSE) between the input sequence and the reconstructed sequence.
6.  If the MSE was above a certain threshold (determined by looking at reconstruction errors on a validation set of normal data), flag it as an anomaly.

Here’s a snippet of what the prediction client looked like (simplified, of course):

```python
from google.cloud import aiplatform
# from google.oauth2 import service_account # For auth if not on GCP, or for service accounts

# endpoint_name = "projects/YOUR_PROJECT_ID/locations/YOUR_REGION/endpoints/YOUR_ENDPOINT_ID"
# credentials = service_account.Credentials.from_service_account_file("path/to/your/keyfile.json")
# client_options = {"api_endpoint": "YOUR_REGION-aiplatform.googleapis.com"}
# prediction_client = aiplatform.gapic.PredictionServiceClient(client_options=client_options, credentials=credentials)

def get_prediction_vertexai(endpoint_id_full_path, instance_data):
    # Vertex AI SDK has simplified this a lot recently.
    # This is more aligned with direct REST/gapic, but SDK `predict` method is easier.
    # For the SDK:
    # endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id_full_path)
    # prediction = endpoint.predict(instances=[instance_data])
    # return prediction.predictions

    # This is a placeholder for the actual prediction call.
    # In a real script, I'd use the aiplatform.Endpoint.predict() method.
    # For demonstration, imagine this formats and sends the request.
    # The instance_data would need to be a list of lists, or a list of dicts
    # depending on the model's expected input format.
    # For an LSTM expecting (1, timesteps, features):
    #reshaped_instance_data = np.array(instance_data).reshape(1, TIMESTEPS, N_FEATURES).tolist()
    
    # This part is tricky because the exact format depends on how the model was exported
    # and what the Vertex AI endpoint expects. I spent a lot of time here.
    # It usually expects a JSON payload with an "instances" key.
    # payload = {"instances": reshaped_instance_data}
    
    # response = prediction_client.predict(endpoint=endpoint_id_full_path, instances=[payload]) # This needs correct formatting
    # For an LSTM autoencoder, the "prediction" would be the reconstructed sequence.
    # For the sake_of_this_example, returning the input
    print(f"Sending data to endpoint: {instance_data[:5]}...") # Log first 5 data points
    # This is where the actual call to Vertex AI would be.
    # The Vertex AI SDK's endpoint.predict() handles a lot of the boilerplate.
    # For example:
    # client = aiplatform.gapic.PredictionServiceClient(client_options={"api_endpoint": api_endpoint})
    # instances = [instance_data] # instance_data should be a list here
    # parameters_dict = {}
    # endpoint_path = client.endpoint_path(project="your-project", location="your-region", endpoint="your-endpoint-id")
    # response = client.predict(endpoint=endpoint_path, instances=instances, parameters=parameters_dict)
    # reconstructed_sequence = response.predictions
    # return reconstructed_sequence
    
    # Simulating a passthrough for now for blog post brevity
    return instance_data # In reality, this would be the reconstructed data from the model
```
Actually invoking the endpoint from an external Python script reliably took some doing. Authentication was the first step – setting up a service account with the right IAM permissions (Vertex AI User) and pointing to the JSON key file. Then, formatting the request correctly. The `predict()` method of the `aiplatform.Endpoint` class in the `google-cloud-aiplatform` SDK was what I ended up using mostly. [5, 8] It handles the JSON structuring for you, which is a relief. [2]

The Unix server deployment itself was pretty basic for this project. I used `screen` to keep the Python script running in the background. [29] A more robust setup would involve `systemd` or Docker, but for a personal project, `screen` did the job. [21, 25] I found a few StackOverflow threads discussing deploying simple Python scripts to Unix servers, which confirmed this as a reasonable starting point for my needs. [30]

### Thresholding and "What is an Anomaly?"

One of the "Aha!" moments, or maybe more of a "Oh, that's how it is" moment, was realizing that "anomaly detection" isn't magic. The model gives you a reconstruction error, but *you* have to decide what level of error constitutes an anomaly. [15, 26]

I ran the trained model on a validation set of *normal* data, calculated all the reconstruction errors, and then plotted their distribution. This helped me pick a threshold – say, anything above the 99th percentile of normal reconstruction errors would be flagged. This part felt more like an art than a science and involved some tweaking. I read a blog post on anomaly detection that mentioned using the Mean Absolute Error (MAE) and finding a threshold based on its distribution on normal data, which was a similar approach to my MSE method. [15]

### Challenges and Lessons Learned

*   **Data Quality and Volume:** Financial data is messy. [1, 7] Cleaning it and preparing it for an LSTM takes a lot of effort. The sheer volume can also be a challenge for local development. [6]
*   **Hyperparameter Tuning:** Finding the right number of LSTM layers, units, sequence length, etc., was time-consuming. Vertex AI does offer Hyperparameter Tuning jobs, which I explored briefly but didn't fully utilize for this iteration due to time constraints. That's definitely something for future exploration.
*   **Real-time Latency:** While Vertex AI endpoints are pretty fast, the round trip of sending data, getting a prediction, and then doing post-processing adds latency. [2, 13] For true high-frequency trading, this setup might be too slow, but for spotting slightly longer-term anomalies (over minutes rather than milliseconds), it seemed viable. [6]
*   **Cost Management:** Running custom training jobs and keeping endpoints live on Vertex AI costs money. [1] For a student project, this meant being mindful of machine types and deleting resources when not in use.
*   **"It works on my machine!" to "It works on Vertex AI!":** The classic. Ensuring the training environment on Vertex AI exactly matched my local setup (or at least was compatible) took a few tries. Docker knowledge is invaluable here. [18, 19]

This project was a fantastic learning experience. Moving from a theoretical understanding of LSTMs and anomaly detection to a working (albeit a bit rough around the edges) system that integrates cloud AI services was a big step. There are many ways to improve it – more sophisticated models, better data pipelines, more rigorous thresholding – but as a foundation, I'm pretty happy with how it turned out.