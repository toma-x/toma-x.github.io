---
layout: post
title: Asian Equity Pair Trading AI
---

## LSTM for Cointegration-Based Pair Trading on HKEX: A Log of Trials and (Some) Triumphs

Following up on my last project with DQNs and Hang Seng futures, I decided to venture into a different domain of algorithmic trading: statistical arbitrage, specifically cointegration-based pair trading. The idea of building a market-neutral strategy was appealing, and I wanted to see if I could apply some more explicitly time-series focused machine learning techniques. This time, my focus was on Hong Kong Stock Exchange (HKEX) equities, using an LSTM model to predict spread behavior, with data sourced from Alpha Vantage and backtesting attempted with Zipline.

### The Allure of Cointegration and Why HKEX

Pair trading, in its classic sense, relies on finding two stocks whose prices have historically moved together. When they temporarily diverge, the strategy bets on their convergence. Cointegration is the statistical property that underpins this long-run equilibrium. If two (non-stationary) time series are cointegrated, a linear combination of them is stationary. This stationary series, the "spread," is what we trade. I'd read a few papers on it, and it seemed more statistically grounded than trying to predict raw price movements.

I picked HKEX equities partly because my previous HSI project gave me some familiarity with the Hong Kong market context, and partly because Alpha Vantage offered historical data for HKEX tickers. I was hoping to find some less-efficiently priced pairs compared to, say, major US stocks.

### Data Wrangling with Alpha Vantage: The First Test of Patience

Getting the data was the first hurdle. Alpha Vantage is great for free access, but it comes with its limitations. I wrote a Python script using their API to download daily adjusted closing prices for a list of HKEX tickers I compiled.

```python
# Part of my data fetching script
# import requests
# import pandas as pd
# from alpha_vantage.timeseries import TimeSeries
# import time

# ALPHA_VANTAGE_API_KEY = "MY_ACTUAL_API_KEY_WOULD_BE_HERE" # stored this in an env var usually
# ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# hkex_tickers = ['0001.HK', '0005.HK', '0700.HK', ... ] # Had a longer list

# all_stock_data = {}
# for ticker in hkex_tickers:
#     try:
#         print(f"Fetching data for {ticker}...")
#         data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
#         # Alpha Vantage returns data with '1. open', '2. high', etc. column names
#         # I'd rename them later to something more Zipline friendly like 'open', 'high', 'low', 'close', 'volume'
#         data.rename(columns={'5. adjusted close': 'adj_close'}, inplace=True)
#         all_stock_data[ticker] = data['adj_close'].sort_index()
#         time.sleep(13) # Basic rate limiting - 5 calls per minute, so >12 seconds
#     except Exception as e:
#         print(f"Could not fetch data for {ticker}: {e}")
#         # Sometimes it was 'Invalid API call', other times just timeouts
```
The main issues were:
1.  **Rate Limiting:** Alpha Vantage's free tier has strict rate limits (5 calls per minute, 500 per day). Downloading data for a decent list of tickers took *ages*. I had to build in `time.sleep()` calls and often run the script overnight. More than once, I hit the daily limit and had to resume the next day.
2.  **Data Availability & Quality:** Not all HKEX tickers I initially wanted were available, or some had very short histories. For those I did get, I had to perform checks for missing values (mostly forward-filled them if gaps were small) and ensure the "adjusted close" was indeed accounting for dividends and splits, which Alpha Vantage generally does. There were a couple of smaller cap stocks where the volume was so low on some days I decided to exclude them due to potential unreliability of price data.

### The Hunt for Cointegrated Pairs: Needles in a Haystack

Once I had a decent dataset (a few years of daily data for about 50-60 HKEX stocks), the next step was finding cointegrated pairs. I opted for the Engle-Granger two-step method because it felt more intuitive to implement initially than the Johansen test.
1.  For every possible pair of stocks, run a linear regression of one stock's price on the other (e.g., `price_A = beta * price_B + intercept`).
2.  Calculate the residuals of this regression: `spread = price_A - beta * price_B - intercept`.
3.  Test these residuals for stationarity using the Augmented Dickey-Fuller (ADF) test. If the residuals are stationary (p-value below a threshold, typically 0.05 or 0.01), the pair is considered cointegrated.

```python
# from statsmodels.tsa.stattools import adfuller
# import statsmodels.api as sm
# import numpy as np

# def find_cointegrated_pairs(data_df, significance_level=0.05):
#     n_series = data_df.shape
#     series_names = data_df.columns
#     cointegrated_pairs = []

#     for i in range(n_series):
#         for j in range(i + 1, n_series):
#             series1_name = series_names[i]
#             series2_name = series_names[j]
            
#             series1 = data_df[series1_name].dropna()
#             series2 = data_df[series2_name].dropna()
            
#             # Align series by index
#             common_idx = series1.index.intersection(series2.index)
#             series1 = series1.loc[common_idx]
#             series2 = series2.loc[common_idx]

#             if len(common_idx) < 252: # Need at least a year of overlapping data
#                 continue

#             # Step 1: OLS regression
#             # Using log prices is often recommended, I tried both, settled on log for stability
#             y = np.log(series1)
#             X = np.log(series2)
#             X_with_const = sm.add_constant(X)
            
#             model = sm.OLS(y, X_with_const).fit()
#             hedge_ratio = model.params.iloc # Beta coefficient
#             # intercept = model.params.iloc # Not directly used for spread calc if we use actual prices for spread

#             # Step 2: Calculate residuals (spread)
#             # spread = y - hedge_ratio * X - intercept # This is the regression residual
#             # For actual trading spread, it's often S_t = log(Y_t) - beta * log(X_t)
#             # I used the latter form for my LSTM input.
#             current_spread = np.log(series1) - hedge_ratio * np.log(series2)

#             # Step 3: ADF test on residuals
#             adf_result = adfuller(current_spread)
#             p_value = adf_result

#             if p_value < significance_level:
#                 cointegrated_pairs.append((series1_name, series2_name, hedge_ratio, p_value))
#                 print(f"Found cointegrated pair: {series1_name} and {series2_name} with p-value: {p_value:.4f} and hedge_ratio: {hedge_ratio:.4f}")
#     return cointegrated_pairs

# # Assuming 'all_prices_df' is a DataFrame with stock prices, columns are tickers
# # pairs = find_cointegrated_pairs(all_prices_df, significance_level=0.05)
```

This part was more frustrating than I anticipated. I ran my script over a rolling window of data (e.g., 2 years) to find pairs. Many initially promising pairs would lose cointegration in subsequent periods. I remember reading a few forum posts where people mentioned that true, stable cointegration is rare and often short-lived. I had to experiment with the lookback period for the OLS regression and the significance level for the ADF test. Setting the p-value too low (e.g., 0.01) yielded very few pairs; too high (e.g., 0.1) and I got pairs whose spreads didn't look very stationary visually. I eventually settled on a p-value of 0.05 and a lookback of about 252 trading days (1 year) for the cointegration test itself, and then used a longer period for actually generating the spread data for the LSTM.

### LSTMs for Spread Dynamics: An Ambitious Step?

Instead of just using a simple z-score of the spread to generate trading signals (i.e., trade when spread > 2 std dev, revert when it crosses 0), I wanted to see if an LSTM could learn more complex patterns in the spread's movement or predict its mean reversion. The hope was that an LSTM could capture subtle dynamics that a fixed z-score threshold might miss, potentially leading to earlier entries or better exits.

My input features for the LSTM were primarily sequences of the normalized spread itself. I normalized the spread for each pair by subtracting its rolling mean and dividing by its rolling standard deviation (essentially a z-score, but the LSTM would see a sequence of these).

The LSTM architecture took some fiddling:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# For a specific pair, after calculating its spread series
# spread_series = ... # This is a pandas Series of the calculated spread values
# normalized_spread = (spread_series - spread_series.rolling(window=60).mean()) / spread_series.rolling(window=60).std()
# normalized_spread.dropna(inplace=True)

# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:(i + seq_length)])
#         y.append(data[i + seq_length]) # Predict the next spread value
#     return np.array(X), np.array(y)

# SEQUENCE_LENGTH = 20 # Use last 20 days of spread to predict next day's
# # X_data, y_data = create_sequences(normalized_spread.values, SEQUENCE_LENGTH)
# # X_data = X_data.reshape((X_data.shape, X_data.shape, 1)) # Reshape for LSTM [samples, timesteps, features]

# def build_lstm_model(seq_length, num_features=1):
#     model = Sequential()
#     model.add(LSTM(64, input_shape=(seq_length, num_features), return_sequences=True, activation='tanh')) # tanh is common for LSTMs
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization()) # Added this after some unstable training runs
#     model.add(LSTM(32, return_sequences=False, activation='tanh'))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(Dense(1)) # Output is the predicted next (normalized) spread value
    
#     model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse') # Started with 0.001, but 0.0005 felt more stable
#     return model

# # Assuming X_train, y_train, X_test, y_test are prepared for a given pair
# # lstm_model_pair_XY = build_lstm_model(SEQUENCE_LENGTH)
# # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# # history = lstm_model_pair_XY.fit(X_train, y_train, epochs=100, batch_size=32,
# #                                validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
```
I spent a good week just getting the data pipeline into the LSTM correct. The `[samples, timesteps, features]` input shape for LSTMs in Keras is a classic stumbling block, and I definitely had my share of `ValueError` exceptions until the reshaping was right. I chose `tanh` activation because I read it's often preferred for LSTMs when data is normalized around zero. Batch Normalization seemed to help with some of the unstable gradients I was seeing in the loss curves, especially when I tried slightly deeper or wider networks. The learning rate for Adam was another point of experimentation; too high and the loss would explode or oscillate wildly. I found 0.0005 to be a reasonable starting point for many pairs. Early stopping was crucial to prevent overfitting, as the amount of *stable* spread data for any given pair wasn't enormous.

One particular struggle was that for some pairs, the LSTM would essentially learn to predict the *current* value as the *next* value, showing very low MSE but offering no real predictive power. This usually happened when the spread was very mean-reverting and oscillated quickly around zero. Plotting predictions against actuals was key to diagnosing this.

### Defining Trading Logic Based on LSTM Predictions

My LSTM predicted the next day's normalized spread value. The trading logic was:
*   If the LSTM predicts the spread will move further away from zero (e.g., current spread is 1.5, LSTM predicts 1.8), and we are not yet in a trade, consider entering short on the spread (short stock A, long stock B, adjusted by hedge ratio).
*   If the LSTM predicts the spread will revert towards zero (e.g., current spread is 2.0, LSTM predicts 1.2), and we are in a short spread position, consider holding or tightening the exit.
*   My actual entry rules were based on the LSTM's prediction being beyond a certain threshold (e.g., predict > 1.5 for shorting spread, predict < -1.5 for longing spread) AND the current spread also being beyond a slightly less extreme threshold (e.g., current > 1.0).
*   Exits were triggered if the LSTM predicted a strong reversion past zero, or if the spread hit a predefined stop-loss level (e.g., original entry threshold + an extra standard deviation, to avoid getting stopped out by noise).

Position sizing was kept simple: allocate a fixed percentage of capital to each leg of the pair. I knew this was suboptimal but wanted to get the core strategy working first.

### The Zipline Gauntlet: Backtesting Woes

Integrating this into Zipline was... an experience. Zipline is powerful but has a learning curve, especially with custom data and models.
My Zipline script needed to:
1.  In `initialize(context)`:
    *   Load the pre-calculated cointegrated pairs and their hedge ratios.
    *   Load the pre-trained LSTM model for each pair. This was tricky; I saved each Keras model to a file (`model_STOCKA_STOCKB.h5`) and loaded them.
    *   Set commission and slippage. I started with Zipline's defaults and then tried to make them more realistic for HKEX.
    ```python
    # from zipline.api import symbol, order_target_percent, record, schedule_function, get_open_orders
    # from zipline.utils.events import date_rules, time_rules
    # from tensorflow.keras.models import load_model
    # import numpy as np
    # import pandas as pd
    
    # def initialize(context):
    #     context.lookback_days_for_cointegration = 252 # For calculating spread stats, not LSTM sequence
    #     context.lstm_sequence_length = 20
    #     context.trading_pairs_details = [] # To store tuples of (sid_A, sid_B, hedge_ratio, lstm_model_object)
    
    #     # Example: This would be populated by my pair finding and model training script
    #     # For HKEX, you need to map tickers to sids. Zipline usually needs custom data ingestion for non-US exchanges.
    #     # For this example, assume '0001.HK_sid' and '0700.HK_sid' are valid Security objects Zipline understands
    #     # In a real scenario, I'd use a custom data bundle or fetch data appropriately.
    #     # For now, let's placeholder with strings and imagine they are sids
    #     # pair1_ticker_A = '0001.HK'; pair1_ticker_B = '0700.HK' 
    #     # pair1_sid_A = symbol(pair1_ticker_A); pair1_sid_B = symbol(pair1_ticker_B)
    #     # pair1_hedge_ratio = 0.65 # Example, from cointegration test
    #     # try:
    #     #     pair1_lstm_model = load_model(f'lstm_model_{pair1_ticker_A}_{pair1_ticker_B}.h5')
    #     # except IOError:
    #     #     print(f"Could not load model for {pair1_ticker_A}-{pair1_ticker_B}")
    #     #     pair1_lstm_model = None # Handle missing model
    #     # if pair1_lstm_model:
    #     #    context.trading_pairs_details.append({'sids': (pair1_sid_A, pair1_sid_B), 
    #     #                                       'hedge_ratio': pair1_hedge_ratio, 
    #     #                                       'model': pair1_lstm_model,
    #     #                                       'tickers': (pair1_ticker_A, pair1_ticker_B)})

    #     # This part is highly dependent on having a Zipline bundle with HKEX data.
    #     # If I were running this for real, I'd have a custom bundle. For this blog post, it's more conceptual.
    #     # Assume I found a pair: StockX and StockY
    #     # context.stock_X = symbol('STOCKX.HK') # This would need to be a valid SID
    #     # context.stock_Y = symbol('STOCKY.HK') # This would need to be a valid SID
    #     # context.hedge_ratio_Y_on_X = ... # from cointegration
    #     # context.lstm_model = load_model('path_to_my_trained_lstm_for_X_Y.h5')
        
    #     context.entry_threshold_pred = 1.5 # LSTM predicts spread > 1.5 std dev
    #     context.entry_threshold_curr = 1.0 # Current spread > 1.0 std dev
    #     context.exit_threshold_pred = 0.5  # LSTM predicts spread < 0.5 std dev (reversion)
    #     context.stop_loss_abs_spread = 3.0 # Absolute spread value for stop loss

    #     # schedule_function(trade_logic, date_rules.every_day(), time_rules.market_open(minutes=30))
    #     pass # Placeholder for full init
    ```

2.  In `handle_data(context, data)` or a scheduled function:
    *   For each pair, fetch historical price data using `data.history()`. This was a pain point – ensuring I fetched enough data for the LSTM sequence AND for calculating the current spread's normalization parameters, without look-ahead bias. I had a few off-by-one errors here that skewed early backtests.
    *   Calculate the current normalized spread.
    *   Prepare the input sequence for the LSTM.
    *   Get the prediction: `predicted_spread_normalized = model.predict(input_sequence)`. Running `model.predict()` for multiple pairs on every bar was slow; Zipline isn't really optimized for this kind of per-bar ML prediction loop with Keras models without careful management.
    *   Implement the entry/exit logic using `order_target_percent`.

One of the biggest headaches was ensuring the data Zipline feeds into `handle_data` was correctly aligned with what my LSTM was trained on, especially regarding adjustments for splits/dividends and the exact timing of data points. I remember one backtest where performance looked stellar, only to realize my `data.history()` call was inadvertently getting future information by a single bar due to how I was indexing into the returned DataFrame. Fixing that brought the results back to a more sobering reality. Also, managing the state for multiple pairs (e.g., current positions, outstanding orders) within `context` required careful organization.

### Sobering Results and What I Learned

After countless hours of coding, training, and debugging, the Zipline backtests were… educational.
For most pairs, the strategy did not consistently outperform a simpler z-score based approach. While the LSTM sometimes identified good entry points, it was also prone to false signals, especially when the underlying cointegration relationship began to decay.
*   **Performance:** A few pairs showed modest positive returns over specific test periods, but Sharpe ratios were generally low (often < 0.5). Many pairs just lost money, especially after factoring in commissions and slippage (I used 0.1% commission per trade and a small fixed slippage in Zipline).
*   **Cointegration is Fickle:** This was the biggest lesson. Pairs that looked beautifully cointegrated in one 2-year window would completely diverge in the next. My LSTM, trained on the "stable" period, was then operating on flawed assumptions. The models themselves weren't designed to detect the breakdown of cointegration.
*   **LSTM Sensitivity:** The LSTMs were quite sensitive to the specific characteristics of each pair's spread. A model architecture and set of hyperparameters that worked okay for one pair might be terrible for another. Retraining frequently would be necessary.
*   **Data Limitations:** Daily data might be too coarse for capturing the fast mean reversion that some pair trading strategies rely on. Intraday data would likely be better but comes with its own set of (even bigger) data handling and processing challenges. Alpha Vantage free tier doesn't offer extensive intraday for HKEX.
*   **Zipline for ML:** While Zipline is great for many things, integrating complex ML models that require significant per-bar computation and state management for many assets simultaneously felt like I was pushing it a bit, or at least, I needed more advanced Zipline skills.

### Where I Might Go From Here (If I Had Infinite Time)

This project definitely highlighted the gap between a cool idea and a profitable, robust trading strategy.
1.  **Dynamic Cointegration Monitoring:** The most critical improvement would be a system to dynamically monitor the stability of cointegration for each active pair. If the p-value from a rolling ADF test degrades, or if the spread's properties change too much, the strategy should stop trading that pair or at least reduce position size.
2.  **LSTM for Regime Detection:** Perhaps an LSTM could be trained not just to predict the spread, but to classify the "state" of the spread (e.g., mean-reverting, trending, broken).
3.  **More Robust Feature Engineering:** For the LSTM, incorporating features like the volatility of the spread, the volatility of the individual stocks, or even broader market regime indicators might help.
4.  **Ensemble Methods:** Maybe combining the LSTM's signal with simpler z-score signals or other indicators could yield a more robust outcome.
5.  **Better Backtesting Rig:** For strategies involving many specific trained models like this, a more batch-oriented backtesting setup, perhaps outside of Zipline's per-bar loop for the initial signal generation, might be more efficient. Test signals, then run through Zipline.

This was a far more involved project than I initially thought. The theoretical appeal of cointegration combined with LSTMs is strong, but the practical implementation details, especially around data quality, model training for noisy financial series, and the non-stationarity of market relationships, are where the real demons lie. The amount of code written just for data processing, pair finding, model training/saving/loading, and then trying to stitch it all into Zipline was substantial. A valuable, if humbling, experience.