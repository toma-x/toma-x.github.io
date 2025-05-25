---
layout: post
title: AI Yield Curve Forecaster
---

## AI Yield Curve Forecaster: A Deep Dive into Transformers and Bloomberg Data

This project has been a significant undertaking, and I'm finally at a stage where I can document the process of building an AI-based yield curve forecaster. The goal was to explore whether a Transformer model, trained on historical interest rate data from the Bloomberg API, could effectively predict future yield curve movements. It's been a journey of steep learning curves, late-night coding sessions, and those small but incredibly rewarding moments of breakthrough.

### The Challenge: Predicting the Unpredictable?

Forecasting the yield curve – essentially predicting future interest rates across different maturities – is a notoriously difficult problem. The curve's shape is influenced by a myriad of economic factors, market sentiment, and policy decisions. Traditional econometric models have their place, but I was keen to see how a deep learning approach, specifically a Transformer, might capture some of the complex temporal dependencies. My hypothesis was that the attention mechanism inherent in Transformers could be particularly well-suited for identifying long-range patterns in interest rate data.

### Data Acquisition: Tapping into Bloomberg

The first hurdle was data. Reliable, high-frequency historical data is paramount for any time-series forecasting task. I was fortunate enough to have access to the Bloomberg API, which allowed me to pull historical daily data for a range of government bond yields. I focused on US Treasury yields for standard tenors: 3-month, 6-month, 1-year, 2-year, 5-year, 10-year, and 30-year.

Getting the data out consistently was an initial challenge. The Bloomberg API has its own syntax and limitations. I ended up writing a Python script using the `blpapi` library. Here's a simplified snippet of how I was fetching data for a single security and date range (error handling and full request details omitted for brevity):

```python
# session is a bdpapi.Session object, already started and service opened.
def fetch_historical_yields(security_ticker, start_date_str, end_date_str, session):
    refDataService = session.getService("//blp/refdata")
    request = refDataService.createRequest("HistoricalDataRequest")

    request.getElement("securities").appendValue(security_ticker)
    request.getElement("fields").appendValue("PX_LAST") # Last price/yield

    request.set("periodicityAdjustment", "ACTUAL")
    request.set("periodicitySelection", "DAILY")
    request.set("startDate", start_date_str) # "YYYYMMDD"
    request.set("endDate", end_date_str)
    request.set("maxDataPoints", 5000) # A reasonable limit

    # print("Sending Request:", request) # Useful for debugging
    session.sendRequest(request)

    # ... event handling loop to receive and process data ...
    # This part was tricky, dealing with async responses
    # and different event types (PARTIAL_RESPONSE, RESPONSE)
    # Storing data in a pandas DataFrame eventually
```
The asynchronous nature of the API calls took some getting used to. I spent a fair bit of time debugging response handling to ensure all data points were captured correctly and stitched together into a coherent time series for each tenor. Preprocessing involved aligning dates, handling any missing values (though Bloomberg data is generally quite clean), and then normalizing the yield rates, which I found crucial for stable training later on. I opted for standardization (subtracting mean and dividing by standard deviation) for each tenor's time series independently.

### Model Selection: Why a Transformer?

Before settling on a Transformer, I did consider other sequence models. LSTMs and GRUs are common choices for time-series data. I've used LSTMs in previous course projects with decent results. However, for this project, I was particularly interested in the Transformer's ability to handle long-range dependencies. Yield curve movements can be influenced by events and trends spanning considerable periods. The self-attention mechanism, which allows the model to weigh the importance of different past time steps directly, seemed theoretically more powerful than the recurrent gating mechanisms of LSTMs, especially for potentially very long input sequences.

I remember reading Vaswani et al.'s "Attention Is All You Need" paper, and while it was focused on NLP, the core concepts of self-attention seemed transferable. There were also a few emerging papers and blog posts discussing Transformers for time-series forecasting, which gave me some confidence this was a reasonable path, albeit a more complex one to implement from scratch compared to a standard LSTM. My main constraint was learning time; I knew implementing a Transformer would be more involved.

### Building the Transformer with PyTorch

I decided to use PyTorch for its flexibility and relatively intuitive API for building custom neural network modules. The core of my model is a Transformer Encoder. I didn't use the decoder part, as I framed the problem as sequence-to-vector (or sequence-to-sequence if predicting multiple steps directly).

My initial model architecture involved:
1.  An input embedding layer (a simple linear layer in my case, as my inputs are already numerical yields) to project the input features (yields for different tenors at a given time step) into a higher-dimensional space.
2.  Positional encoding, added to the embeddings to give the model information about the order of the time steps. I used the standard sinusoidal positional encoding. Getting this right, especially the dimensions, took a few tries.
3.  A stack of Transformer Encoder layers. Each layer contains a multi-head self-attention mechanism and a feed-forward neural network.
4.  A final linear layer to map the output of the Transformer Encoder to the desired number of output steps and tenors.

Here’s a rough idea of the `TransformerModel` class structure:

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe) # So it's not a model parameter

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerYieldForecaster(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_future_steps, dim_feedforward=2048, dropout=0.1):
        super(TransformerYieldForecaster, self).__init__()
        self.d_model = d_model
        self.num_future_steps = num_future_steps
        self.input_dim = input_dim # Number of tenors

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False) # PyTorch expects (seq, batch, feature)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output layer to predict num_future_steps for each of the input_dim tenors
        self.output_layer = nn.Linear(d_model, input_dim * num_future_steps) 

    def forward(self, src, src_mask=None):
        # src shape: [seq_len, batch_size, input_dim]
        src_projected = self.input_projection(src) * math.sqrt(self.d_model) # Scaling, as suggested in some implementations
        src_pos_encoded = self.pos_encoder(src_projected)
        
        # PyTorch's TransformerEncoder expects src_mask for attention,
        # e.g., to prevent attending to future tokens in a causal setting,
        # or to padding tokens. For my forecasting, I used a lookback window, so causal mask wasn't strictly needed for encoder-only.
        # However, padding masks are important if batches have variable sequence lengths.
        # For simplicity in this stage, I often worked with fixed-length sequences.
        
        output = self.transformer_encoder(src_pos_encoded, src_mask) # output shape: [seq_len, batch_size, d_model]
        
        # We are interested in the output from the last time step of the encoder sequence for prediction
        prediction_input = output[-1, :, :] # Take the output of the last element in the sequence. Shape: [batch_size, d_model]
        
        forecast = self.output_layer(prediction_input) # Shape: [batch_size, input_dim * num_future_steps]
        # Reshape to [batch_size, num_future_steps, input_dim] for easier interpretation
        forecast = forecast.view(-1, self.num_future_steps, self.input_dim)
        return forecast
```

One of the trickiest parts was ensuring the dimensions matched up correctly at each stage, especially with `batch_first=False` being the default for `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` in older PyTorch versions (or just me misreading documentation repeatedly). I recall spending an entire evening debugging shape mismatches. The `view` operation on the final forecast also required careful thought to ensure the output dimensions corresponded to `[batch_size, num_future_steps, num_tenors]`.

For the `src_mask`, when using fixed-length input sequences from my `DataLoader`, I initially didn't implement a sophisticated padding mask, assuming all sequences in a batch were padded to the same length. This is something I'd refine in a more production-ready system. For causal masking (to prevent looking ahead), it wasn't strictly necessary for the encoder-only setup if my input `src` only contained past data.

### Training: The Long Haul

For training, I created sequences of historical yield curve data. For example, using 60 days of data (across all tenors) to predict the yield curve for the next 5 days. My `Dataset` class would slide this window across the entire historical dataset.

I used Mean Squared Error (MSE) as the loss function, as it's standard for regression tasks. AdamW was my optimizer of choice, often recommended for training Transformers.

```python
# Simplified training loop structure
# model = TransformerYieldForecaster(...)
# criterion = nn.MSELoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001) # Learning rate was a key hyperparameter

# Assuming train_loader is a PyTorch DataLoader

# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0.0
#     for i, (batch_sequences, batch_targets) in enumerate(train_loader):
#         # batch_sequences shape: [seq_len, batch_size, input_dim]
#         # batch_targets shape: [batch_size, num_future_steps, input_dim]
          
#         optimizer.zero_grad()
#         
#         forecast = model(batch_sequences.to(device)) # .to(device) for GPU
#         
#         loss = criterion(forecast, batch_targets.to(device))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping helped a bit with stability
#         optimizer.step()
#         
#         epoch_loss += loss.item()
#     
#     avg_epoch_loss = epoch_loss / len(train_loader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
    # ... validation loop ...
```
One specific issue I remember grappling with was the learning rate. Too high, and the loss would explode. Too low, and training was painfully slow with minimal improvement. I ended up using a learning rate scheduler (`ReduceLROnPlateau`) which helped to some extent, reducing the LR when the validation loss stagnated. Gradient clipping was also something I added after reading some forum posts about training Transformers, as it seemed to help prevent sudden spikes in loss.

Overfitting was another concern. With a powerful model like a Transformer, it's easy to fit the training data too well and perform poorly on unseen data. I used dropout in both the positional encoding and the Transformer encoder layers. A separate validation set, held out from the training data, was crucial for monitoring overfitting and for hyperparameter tuning (like the number of encoder layers, heads, `d_model`, etc.). I spent a considerable amount of time adjusting these, often training smaller versions of the model first to get a feel for what worked.

### Multi-Step Forecasting Approach

My model is designed for direct multi-step forecasting. The final linear layer `self.output_layer = nn.Linear(d_model, input_dim * num_future_steps)` directly outputs all the `num_future_steps` for each `input_dim` (tenor). This means if I want to predict 5 steps ahead for 7 tenors, the output layer has `7 * 5 = 35` output neurons, which are then reshaped.

I considered an iterative approach, where the model predicts one step ahead, and that prediction is fed back as input to predict the next step. This can suffer from error accumulation. The direct approach attempts to predict all steps simultaneously. It felt like a more end-to-end deep learning way, though it might require a more complex model to learn the dependencies across future steps.

### Preliminary Results and Evaluation

Evaluating the model involved calculating metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on a held-out test set. The results were... interesting. The model was clearly learning *something* about the yield curve dynamics. For very short-term forecasts (e.g., 1-day ahead), the predictions were reasonably close to the actual values. As the forecast horizon extended (e.g., 5 days or 10 days ahead), the accuracy predictably decreased.

Visually inspecting the predicted curves versus the actual curves was very insightful. Sometimes the model would capture the general direction of change but miss the magnitude, or vice-versa. There were also instances where it seemed to predict a sort of "average" future curve, smoothing out some of the more volatile movements. This suggested that while the Transformer was capturing some patterns, predicting sharp, unexpected shifts remained a significant challenge – which is hardly surprising.

I don't have perfectly polished graphs to show here, but imagine plotting the actual 10-year yield against the model's 1-day, 5-day, and 10-day ahead forecasts. The 1-day forecast would track the actual quite closely. The 5-day forecast would start showing some deviations, and the 10-day forecast would be even smoother and less reactive.

One persistent issue was how to meaningfully evaluate "curve shape" predictions beyond just point-wise errors for each tenor. I explored some ideas like looking at the error in predicted spreads (e.g., 10Y-2Y spread), but didn't fully implement a comprehensive shape-based metric.

### Lessons Learned and Future Directions

This project was an immense learning experience.
1.  **Data is King (and a Pain):** The Bloomberg API is powerful, but robust data pipelines are critical and time-consuming to build. Any slight misalignment or error in data preprocessing can silently sabotage the model.
2.  **Transformers are Complex Beasts:** While PyTorch makes implementing the layers relatively straightforward, understanding all the nuances of attention mechanisms, positional encodings, and masking for time-series data takes serious effort. I definitely re-read documentation and tutorials multiple times. Debugging shape errors was a constant companion in the early stages.
3.  **Hyperparameter Tuning is an Art and a Science (and a lot of GPU time):** Finding the right combination of `d_model`, `nhead`, number of layers, learning rate, and dropout requires patience and a systematic approach. I wish I had more computational resources to run more extensive grid searches.
4.  **Realistic Expectations:** Forecasting financial time series is incredibly hard. While the model showed some promise, it's not a crystal ball. Understanding its limitations is as important as celebrating its successes.

For future work, I’d consider a few avenues:
*   **Incorporating Exogenous Variables:** Adding macroeconomic indicators (e.g., inflation, GDP growth) as additional inputs to the model could provide more context.
*   **Attention Visualization:** Trying to interpret what the self-attention layers are focusing on could provide insights into how the model is making its predictions.
*   **More Sophisticated Evaluation:** Developing better metrics for yield curve shape and dynamics.
*   **Hybrid Models:** Perhaps combining the Transformer with a more traditional econometric model.
*   **Exploring different Transformer architectures:** Looking into variants like the Informer or Autoformer, which are specifically designed for long sequence time-series forecasting, might yield better results for longer horizons.

This project pushed my understanding of deep learning, time-series analysis, and financial data. While the "AI Yield Curve Forecaster" isn't going to break the markets anytime soon, the process of building it has been incredibly valuable. There's always more to learn and improve, which is the most exciting part.