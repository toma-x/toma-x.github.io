---
layout: post
title: AI Options Pricing Anomaly Detector
---

## Taming Tick Data: My Journey Building an Options Pricing Anomaly Detector with LSTMs

This project has been a bit of a marathon, but I’m finally at a point where I can share some of my experiences building an AI to sniff out potential mispricings in options contracts. The idea was to leverage Long Short-Term Memory networks (LSTMs) because of their strength in handling time-series data, and what’s more time-sensitive than tick-level market data? The goal: train a model to identify moments where an option's price seemed out of whack, potentially signaling an arbitrage opportunity. Easier said than done, as I quickly found out.

The core of this endeavor revolved around using **TensorFlow and Keras**. I’d used Keras for a few class projects before, and its relatively straightforward API felt like the right choice, especially since I knew I'd be spending a lot of time wrestling with the data and model architecture, not fighting the framework.

**The Data Beast: Tick-Level Headaches**

First off, getting my hands on usable tick-level options data was a massive hurdle. This stuff isn't usually just lying around for free, especially clean, comprehensive datasets. After a lot of searching, I managed to acquire a dataset covering a few specific tickers over several months. It was… dense. Millions of rows, each a tiny snapshot of price, volume, strike, expiry, and the underlying's price at that exact moment.

The initial preprocessing was a slog. The raw data was noisy, with occasional outliers that made no sense and missing values that needed careful handling. I spent what felt like ages just cleaning it. For missing numerical values, I mostly used forward-fill, assuming the last known price was the most reasonable estimate for a very short gap. For some outliers, I ended up capping them based on a multiple of the standard deviation from a rolling mean – it felt a bit arbitrary, but leaving them in skewed everything badly.

One of the first things I realized was that raw prices weren't ideal. Normalization became key. I opted for a min-max normalization on a per-option basis, scaled between 0 and 1, for features like the option price itself and the underlying stock price. My thinking was that LSTMs prefer input features to be in a similar range.

```python
# Example of how I created sequences
# 'df' is my preprocessed pandas DataFrame
# 'look_back' is the number of previous time steps to use as input variables
# to predict the next time period.
def create_dataset(dataset, look_back=60): # using 60 ticks, so about a minute of data if 1 tick/sec
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :] # All features for the look_back period
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0]) # Predicting the next option price (index 0)
    return np.array(dataX), np.array(dataY)

# This 'scaled_data' would be the result of applying MinMaxScaler
# For a specific option contract's data
# X, y = create_dataset(scaled_data_for_one_option, time_steps)
```
I decided to frame the problem, at least initially, as predicting the next tick's price for an option. If my model could get good at that, then significant deviations between its prediction and the actual next tick could indicate an anomaly.

**Wrestling with the LSTM Architecture in Keras**

My first LSTM model was probably too simple. I started with a single LSTM layer and a Dense output layer.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Initial thoughts for the model structure
# num_features would be the number of input features per time step
# time_steps would be my look_back period

# model = Sequential()
# model.add(LSTM(units=50, input_shape=(time_steps, num_features))) # A modest number of units
# model.add(Dense(units=1)) # Predicting one value - the next price
# model.compile(optimizer='adam', loss='mean_squared_error')
```
The results were... underwhelming. The loss plateaued quickly, and the predictions were sluggish, often just lagging the actual price. I spent a lot of time on forums and reading through Keras documentation. One StackOverflow thread (I wish I’d saved the link!) mentioned the importance of stacking LSTM layers for more complex patterns, especially with `return_sequences=True` on the intermediate LSTM layers. This allows the next LSTM layer to receive the full sequence output from the previous one, not just the final step's output.

So, I moved to a stacked LSTM structure. I also experimented with adding `Dropout` layers to combat overfitting, which started becoming an issue as I increased model complexity and training epochs.
My thinking for the number of units in LSTM layers (e.g., 64, then 32) was a bit of a heuristic – start with a reasonable number, see if it underfits or overfits, and adjust. I didn't have the computational resources for extensive hyperparameter grid searches across huge ranges.

```python
# A later iteration of the model
# num_features = X_train.shape # Inferred from the training data shape

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(time_steps, num_features)))
model.add(Dropout(0.2)) # Adding some dropout
model.add(LSTM(units=32, return_sequences=False)) # Last LSTM layer before Dense
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Still predicting the next price

model.compile(optimizer='adam', loss='mean_squared_error')
# print(model.summary()) # Always good to check this
```
I stuck with the 'adam' optimizer as it’s generally a good default. For the loss function, 'mean_squared_error' (MSE) made sense since I was doing regression (predicting the next price).

**Training: The Waiting Game and False Dawns**

Training these models on tick data, even for a single option contract, took *ages* on my laptop. I’d set `model.fit()` running and come back hours later. I used an `EarlyStopping` callback to prevent wasting time if the validation loss stopped improving.

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks I found useful
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# history = model.fit(X_train, y_train,
#                     epochs=100, # A high number, but early stopping will kick in
#                     batch_size=32, # A common batch size
#                     validation_data=(X_val, y_val),
#                     callbacks=[early_stopping, reduce_lr],
#                     verbose=1)
```
There were many frustrating moments. Sometimes the validation loss would be significantly higher than the training loss, a clear sign of overfitting. I’d then go back, increase dropout, simplify the model a bit, or try to get more varied training data. Other times, the loss would just stagnate, refusing to go down no matter what I tweaked. This often sent me back to re-examine my data preprocessing – was there some weird scaling issue? Did I shuffle my sequences correctly before splitting into train/validation?

One breakthrough came when I started paying more attention to the learning rate. The `ReduceLROnPlateau` callback helped by automatically reducing the learning rate when the validation loss stagnated. This often gave the model the little nudge it needed to find a better minimum.

**Spotting the "Anomalies"**

Once I had a model that seemed to perform reasonably well on validation data (meaning its predicted prices weren't wildly off from the actual next prices), the next step was to define what constituted an "anomaly."

My approach was to calculate the prediction error (predicted price vs. actual price) on the test set. Then, I looked for errors that were significantly larger than the typical error. I used a threshold based on a multiple of the standard deviation of these prediction errors.

```python
# After model.predict(X_test)
# predictions = model.predict(X_test_contract_A)
# actual_prices = y_test_contract_A # These would be the true next prices

# # Inverse transform if prices were scaled, to get actual price differences
# # predictions_descaled = scaler_price.inverse_transform(predictions)
# # actual_prices_descaled = scaler_price.inverse_transform(actual_prices.reshape(-1,1))

# errors = actual_prices_descaled - predictions_descaled
# mean_error = np.mean(errors)
# std_error = np.std(errors)

# anomaly_threshold_upper = mean_error + (3 * std_error) # e.g., 3 standard deviations
# anomaly_threshold_lower = mean_error - (3 * std_error)

# potential_mispricings = []
# for i in range(len(errors)):
#     if errors[i] > anomaly_threshold_upper or errors[i] < anomaly_threshold_lower:
#         # This indicates the actual price was significantly different from prediction
#         # Store the index, the actual price, predicted price, and the error
#         potential_mispricings.append({
#             "index": i, # original index in test set
#             "actual": actual_prices_descaled[i],
#             "predicted": predictions_descaled[i],
#             "error": errors[i]
#         })
#         # print(f"Anomaly found at index {i}: Actual={actual_prices_descaled[i]}, Predicted={predictions_descaled[i]}, Error={errors[i]}")
```

When I first saw a few flagged points where my model's prediction was, say, $0.05 off from the actual subsequent tick price, and the standard deviation of errors was only $0.01, it was genuinely exciting. Could this be it? A real, detectable mispricing?

Of course, the immediate follow-up thought was: is this *actual* arbitrage? Probably not directly. The model is identifying statistical anomalies based on its learned patterns. Real-world arbitrage needs to account for transaction costs, bid-ask spreads, latency, and the risk that the "mispricing" corrects itself before a trade can even be executed, or worse, moves further against you. My model doesn't inherently understand Black-Scholes or any formal option pricing theory; it's purely learning from the historical sequence of prices and whatever other features I fed it.

**Lessons Burned In**

This project was a huge learning curve.
1.  **Data is King, Queen, and the entire Royal Court:** The sheer effort to get, clean, and prepare tick-level data was something I underestimated. Any imperfection here cascades into model performance.
2.  **Patience with Training:** LSTMs, especially on large sequences, are not quick to train on consumer hardware. Many cups of coffee were consumed waiting for epochs to complete.
3.  **Overfitting is a Persistent Foe:** Especially with complex models like LSTMs, it’s so easy for the model to memorize the training data rather than learn generalizable patterns. Regularization techniques (Dropout, L1/L2) and robust validation are critical. I found myself constantly looking at training vs. validation loss curves.
4.  **Interpreting "Anomalies":** Just because the model flags something doesn't mean it's free money. Context is crucial. Is it a data error? A genuinely unusual market event? Or a fleeting inefficiency? I realized my "anomaly detector" was more of a "points of interest highlighter."

I remember one particular week where I was convinced my sequence generation was off by one timestep. My predictions always seemed to lag perfectly. I spent hours stepping through the `create_dataset` function with tiny sample arrays, drawing diagrams of indices. Turns out, the issue was more subtle, related to how I was aligning the features from the *current* tick intended to predict the *next* option price tick. It was a detail in the `dataY.append(dataset[i + look_back, 0])` part and ensuring all features in `dataset[i:(i + look_back), :]` were truly prior to that `y` value. Small indexing errors can lead to huge headaches and subtly wrong models.

**Where To From Here?**

While I wouldn't trust this current iteration to manage my (non-existent) life savings, it’s been an incredible learning experience. The model can definitely highlight unusual price movements based on recent historical context.

Future ideas?
*   **Incorporate Greeks:** If I could get reliable, synchronized tick-level Greeks, that would be a huge feature enhancement.
*   **Attention Mechanisms/Transformers:** Given the buzz, exploring Transformer architectures for time-series forecasting would be interesting, though likely even more data-hungry and computationally intensive.
*   **More Sophisticated Anomaly Definition:** Instead of just deviation from predicted price, maybe train a model specifically on features designed to represent "normal" market behavior and look for deviations in that latent space.
*   **Consider Bid-Ask Spreads:** For any real trading application, predictions need to be actionable considering the spread. My current model predicts a single price, not the bid or ask.

This project definitely hammered home how challenging financial markets are to model. But the process of trying to decode those patterns, even a tiny piece, with tools like TensorFlow and Keras, has been incredibly rewarding. It’s one thing to read about LSTMs, it’s another to see them (slowly) learn from a chaotic stream of numbers.