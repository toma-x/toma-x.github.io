---
layout: post
title: AI Counterparty Stress Tester
---

## Building an AI Counterparty Stress Tester: A Deep Dive

This project has been a bit of a marathon, but I’m finally at a point where I can share some of the journey. For a while now, I've been fascinated by the intersection of finance and machine learning, specifically how AI can be used to model complex financial risks. So, I decided to build a tool to simulate counterparty defaults under various market stress scenarios, using an LSTM model to forecast exposure profiles. I’m calling it the "AI Counterparty Stress Tester."

The main goal was to create a system that could take historical counterparty exposure data, train a model to predict future exposures, and then see how those exposures (and potential defaults) would behave if the market suddenly went sideways. This felt like a relevant problem, especially given how interconnected financial institutions are.

### The Core Idea and Initial Hurdles

The basic idea is to estimate potential losses from counterparties failing to meet their obligations. This means figuring out two main things: the likely Exposure at Default (EAD) and the Probability of Default (PD) for each counterparty, especially when the market is under stress. For the EAD part, I thought a time-series forecasting model would be appropriate, as exposures often have a temporal component. That’s where PyTorch and LSTMs came into the picture.

I chose Python for this project, mostly because I'm comfortable with it and the ecosystem for data science (Pandas, NumPy, Scikit-learn) and deep learning (PyTorch) is just so mature. Setting up the environment was straightforward enough with Conda.

### Forecasting Exposure at Default with PyTorch LSTMs

This was the part I was most excited and, admittedly, a bit intimidated by. I’d read a fair bit about LSTMs and their ability to capture long-range dependencies in sequential data, which seemed perfect for forecasting exposure profiles that can fluctuate based on past values and market conditions.

**Data Generation (The Not-So-Glamorous Part)**

Getting real, granular counterparty exposure data is next to impossible for a personal project. So, I had to simulate it. I wrote a Python script to generate synthetic time series data for a few hypothetical counterparties. Each series represented daily exposure, and I tried to inject some seasonality and responsiveness to simulated market factors (like a mock interest rate index and a volatility index). This wasn't perfect, but it gave me something to work with.

```python
import numpy as np
import pandas as pd

def generate_counterparty_exposure(n_days=1000, base_exposure=100000, trend_factor=0.01, seasonality_period=90, seasonality_strength=0.1, noise_level=0.05):
    days = np.arange(n_days)
    trend = trend_factor * days
    seasonality = seasonality_strength * np.sin(2 * np.pi * days / seasonality_period)
    noise = noise_level * np.random.randn(n_days)
    
    # Simulate some market factor influence
    market_factor_shock = np.zeros(n_days)
    shock_points = np.random.choice(days, size=5, replace=False)
    for shock_idx in shock_points:
        market_factor_shock[shock_idx:shock_idx+30] += np.random.randn() * 0.2 # A shock event
        
    exposure = base_exposure * (1 + trend + seasonality + noise + market_factor_shock)
    exposure = np.maximum(exposure, 0) # Exposure cannot be negative
    return pd.Series(exposure)

# Generate for a few counterparties
num_counterparties = 5
all_exposures = {}
for i in range(num_counterparties):
    all_exposures[f'CP_{i+1}'] = generate_counterparty_exposure(n_days=730) # 2 years of daily data

df_exposures = pd.DataFrame(all_exposures)
# For simplicity, let's add a mock market factor to use as a feature
df_exposures['MarketFactor1'] = np.random.randn(len(df_exposures)) * 0.1 + np.sin(np.arange(len(df_exposures))/50)
```
This script isn't exactly sophisticated, but it gave me sequences that had *some* character. I then had to prepare this data for the LSTM. This involved creating sequences of a fixed length (say, using the past 30 days of exposure and market factors to predict the next day's exposure). Normalizing the data using `MinMaxScaler` from scikit-learn was also crucial; LSTMs, like many neural nets, are sensitive to the scale of input data.

**The LSTM Model Architecture**

I decided on a relatively simple LSTM architecture to start with. I’d seen examples online, some using very deep networks, but for a first pass, I wanted something manageable.

```python
import torch
import torch.nn as nn

class ExposureLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_prob=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        # Linear layer to map LSTM output to desired output size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
        # Hidden cell initialization (optional, but can help)
        # self.hidden_cell = (torch.zeros(self.num_layers,1,self.hidden_layer_size),
        #                     torch.zeros(self.num_layers,1,self.hidden_layer_size))

    def forward(self, input_seq):
        # Initialize hidden state with zeros if not passed, or handle batching
        # For batch_first=True, lstm_out shape is (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(input_seq) # self.hidden_cell needs to be adapted for batch
        
        # We only want the output from the last time step
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
```

I went with two LSTM layers (`num_layers=2`) and a hidden size of 64. The `batch_first=True` argument was a lifesaver; I initially had it as `False` (the default) and spent a good couple of hours debugging shape mismatches because my data was naturally in `(batch_size, seq_len, features)` format. A StackOverflow thread finally pointed out my error there. The dropout was added later when I noticed some overfitting on my training data. The input size depended on the number of features (e.g., past exposure, MarketFactor1). Output size was 1, as I was predicting the next day's exposure.

**Training Trials and Tribulations**

Training was... an experience. I used Mean Squared Error (MSE) as the loss function and the Adam optimizer.

```python
# Assume train_loader is a PyTorch DataLoader providing batches of (sequence, label)
# model = ExposureLSTM(input_size=input_features, hidden_layer_size=64, num_layers=2, output_size=1)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# epochs = 50 # Or more, depending on convergence

# for i in range(epochs):
#     for seq, labels in train_loader:
#         optimizer.zero_grad()
        
#         # Reset hidden state for each new sequence/batch if stateful
#         # model.hidden_cell = (torch.zeros(model.num_layers, seq.size(0), model.hidden_layer_size).to(device),
#         #                      torch.zeros(model.num_layers, seq.size(0), model.hidden_layer_size).to(device))
        
#         y_pred = model(seq)
        
#         single_loss = loss_function(y_pred, labels)
#         single_loss.backward()
#         optimizer.step()
        
#     if i%5 == 1:
#         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
```
*(Note: The commented-out hidden state reset needs careful handling depending on how sequences are batched and if you want to maintain state across batches of the same long sequence. For my independent sequences, re-initializing or letting PyTorch handle zero-init for `h_0, c_0` per call was fine).*

One of the first major headaches was the loss just not decreasing, or even exploding into NaNs. This led me down a rabbit hole of checking data normalization (again!), learning rates, and gradient clipping (`torch.nn.utils.clip_grad_norm_`). A smaller learning rate (e.g., 0.0005 instead of 0.001) and gradient clipping helped stabilize things. I also played around with the number of hidden units and layers. Too few, and it wouldn't learn; too many, and it would overfit super quickly or take forever to train on my modest laptop. I didn't have access to a powerful GPU cluster, so efficiency was a concern.

I recall one late night where the model was learning, but predictions were consistently lagging the actuals by one time step. This is a classic trap with time series if you're not careful with how you structure your input sequences and targets. I had to meticulously check my sequence creation logic – ensuring `X_t` was truly predicting `y_{t+1}` and not just `y_t` shifted.

### Simulating Market Shocks

Once I had a somewhat working EAD forecasting model, the next step was to simulate market shocks. For this, I defined a few scenarios:
1.  A sudden interest rate hike.
2.  A sharp drop in a relevant market index (like the S&P 500, though I used a proxy).
3.  Increased market volatility.

I represented these as changes to the "MarketFactor1" (and potentially other factors if I had them) that fed into the LSTM. For example, a shock might mean setting `MarketFactor1` to an unusually high or low value for a certain period. The LSTM, having been trained on historical data including this factor, would then predict how the EAD profile changes under these stressed conditions. This was more of a qualitative assessment initially – does the EAD go up or down as expected?

### Counterparty Default Simulation

This part was less AI-driven in my current version. I considered building another model for Probability of Default (PD) but decided to keep it simpler for now to focus on the EAD forecasting. I opted for a basic approach where I assigned a baseline PD to each counterparty and then used a multiplier based on the severity of the market shock.

```python
def get_stressed_pd(baseline_pd, shock_severity_factor):
    # shock_severity_factor could be derived from how much MarketFactor1 changed
    stressed_pd = baseline_pd * (1 + shock_severity_factor * 2.0) # Arbitrary multiplier
    return min(stressed_pd, 1.0) # Cap at 100%

# Example
baseline_pds = {'CP_1': 0.01, 'CP_2': 0.02, 'CP_3': 0.005, 'CP_4': 0.015, 'CP_5': 0.03}
shock_factor = 0.5 # Representing a moderate shock

stressed_pds_cp = {}
for cp, pd_val in baseline_pds.items():
    stressed_pds_cp[cp] = get_stressed_pd(pd_val, shock_factor)
```
Then, for each counterparty, under a given shock scenario:
1.  Forecast the EAD using the LSTM with the stressed market factor.
2.  Calculate the stressed PD.
3.  Assume a fixed Loss Given Default (LGD), for example, 45% (a common figure in finance).
4.  Run a Monte Carlo simulation: for each simulation trial, the counterparty defaults if a random number (0-1) is less than its stressed PD.
5.  If a default occurs, the loss for that trial is `Stressed EAD * LGD`.

Aggregating these simulated losses across many trials and counterparties would give an idea of the total stress loss.

### Integrating the Components

Bringing all these pieces together into a cohesive Python tool was challenging. I had separate scripts for data generation, model training, and then a main script to run the stress tests. The main script would:
1.  Load the trained LSTM model.
2.  Allow the user to define a shock scenario (e.g., by specifying values for market factors).
3.  For each counterparty:
    a.  Prepare the input sequence with the shocked market data.
    b.  Get the EAD forecast from the LSTM.
    c.  Calculate the stressed PD.
    d.  Simulate defaults and calculate losses.
4.  Report the aggregated potential losses.

Managing file paths for the model, the scalers (very important to save the `MinMaxScaler` used on the training data and apply the *exact same* transformation to new/test data!), and the output reports required careful organization. I ended up pickling the trained model and the scaler.

### That "Aha!" Moment

One of the most satisfying moments was when the EAD forecasts started to look sensible under shock. I had struggled for a while with the market factor not having enough impact. The LSTM seemed to be relying too heavily on the autoregressive component (past exposure values). I eventually figured out that the scale of my simulated market factor was too small compared to the exposure values, even after normalization. Once I adjusted the generation of `MarketFactor1` to have a more significant (normalized) variance and retrained, the EAD profiles started reacting much more dynamically to the simulated shocks. Seeing the forecasted EAD spike when I applied a "severe" shock to the market factor was a real breakthrough. It wasn't perfect, but it was *reacting*.

### Key Learnings and Future Work

This project was a huge learning experience.
*   **Data is King (Even Simulated Data):** The quality and characteristics of the input data profoundly impact the model. Simulating data that has some plausible dynamics was harder than I thought.
*   **LSTMs Have Nuances:** Getting the input shapes right, managing hidden states (though often PyTorch handles it well for simpler cases), and tuning hyperparameters for LSTMs takes patience. The PyTorch documentation and forums were my best friends. I specifically remember a few PyTorch forum threads on `LSTM` input/output shapes that were invaluable.
*   **Iterative Process:** I didn't get it right on the first try. Or the tenth. It was a lot of tweaking, retraining, and re-evaluating.
*   **Simplification is Necessary:** For a student project, you have to make simplifying assumptions (like the PD model or fixed LGD). Otherwise, the scope becomes unmanageable.

**What I'd Do Differently Next Time:**
*   **Better PD Modeling:** Explore a more data-driven approach for PD, perhaps another ML model if I could find/simulate appropriate data.
*   **More Sophisticated Shock Scenarios:** Incorporate multiple market factors and model their correlations.
*   **Model Interpretability:** Try to understand *why* the LSTM makes certain predictions, perhaps using techniques like SHAP, though that's an advanced topic.
*   **Code Structure:** Early on, my code was a bit of a monolith. I gradually refactored it into more modular functions and classes, but I could have planned that better from the start.

This project has definitely solidified my interest in financial risk modeling and AI. While the "AI Counterparty Stress Tester" is still a work in progress and more of a proof-of-concept, the process of building it has been incredibly rewarding. There's still a lot to improve, like refining the default simulation and adding more complex shock interactions, but it's a solid foundation.