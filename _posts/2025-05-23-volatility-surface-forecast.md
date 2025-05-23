---
layout: post
title: Volatility Surface Forecaster
---

## Predicting Volatility Surfaces with LSTMs from Raw Tick Data: A Deep Dive

It’s been a while since my last update, mostly because I’ve been completely buried in what turned out to be a much more ambitious project than I initially anticipated. I’m excited to finally share some details about my work on the "Volatility Surface Forecaster." The core idea was to leverage **TensorFlow LSTM** networks to predict option volatility surfaces directly from **raw tick data**. The benchmark to beat was a standard GARCH model, and I’m pretty thrilled with the outcome: a **7% MAE reduction** on **S&P 500 options data**. This post will walk through the journey, the struggles, and some of the key learnings.

### The Starting Point: Why Volatility Surfaces and Raw Ticks?

I’ve always been fascinated by financial markets, and options pricing in particular. The volatility surface, which shows implied volatility across different strike prices and expirations, is a cornerstone of options trading and risk management. Most models I encountered seemed to use daily or even lower frequency data. My hypothesis was that **raw tick data**, despite its noise and sheer volume, might contain finer-grained information that could lead to more accurate short-term volatility forecasts. LSTMs seemed like a natural fit for this kind of sequential, high-frequency data.

### Grappling with the Data: The Unseen 80%

This is where a significant chunk of the project time went. Sourcing good quality **S&P 500 options data** along with corresponding **raw tick data** for the underlying (SPX or SPY) was the first hurdle. I eventually managed to get access to a historical dataset through a university research portal, but it was far from clean.

The tick data was massive – terabytes of it. My initial attempts to load even a small fraction into Pandas on my local machine were… optimistic. I ended up writing scripts to process the data in chunks. The preprocessing pipeline for the tick data involved:

1.  **Filtering:** Removing obviously erroneous ticks (e.g., negative prices, crazy volumes). This was more art than science, relying on some statistical outlier detection and common sense.
2.  **Synchronization:** Option prices and underlying ticks don't arrive perfectly aligned. I had to develop a consistent method for snapshotting relevant underlying market features (like bid, ask, last traded price, volume) at the moments option quotes were updated, or at fixed intervals.
3.  **Feature Engineering from Ticks:** Simply feeding raw tick prices into an LSTM didn't feel right. I decided to aggregate tick data into 1-minute intervals. For each interval, I calculated features like:
    *   Volatility (realized volatility from ticks within the minute)
    *   Order imbalance (from bid/ask sizes if available, though this was spotty in my dataset)
    *   Tick frequency
    *   Bid-ask spread dynamics

Here’s a very simplified look at how I started to structure the tick data aggregation. This is a very early version and had many iterations:

```python
# Example: Aggregating some features from tick data
# 'df_ticks' is a pandas DataFrame with columns like 'timestamp', 'price', 'volume', 'bid', 'ask'

df_ticks['timestamp'] = pd.to_datetime(df_ticks['timestamp'])
df_ticks = df_ticks.set_index('timestamp')

# Resample to 1-minute intervals
# Note: I had to be careful here with how aggregation was done,
# 'ohlc' for price, 'sum' for volume, etc.
# This is a simplified example focusing on just a couple of features.
agg_rules = {
    'price': 'ohlc',
    'volume': 'sum',
    'bid': 'last', # This was a simplification, better methods exist
    'ask': 'last'
}

# df_minute_features = df_ticks.resample('1min').apply(agg_rules) # Initial thought
# Actually, I had to be more careful. `apply` was too slow.
# I ended up doing something like this for each feature:
vol_proxy = df_ticks['price'].resample('1min').std() # proxy for realized vol
trade_count = df_ticks['price'].resample('1min').count()
# ... and so on for other engineered features.
# Then I'd join these series together.
```

The target variable itself, the volatility surface, also needed careful construction. For each 1-minute interval, I needed the corresponding implied volatilities. I used the Black-Scholes model (making standard assumptions about interest rates and dividends, which I sourced separately) to back out IVs from option prices. The "surface" was then defined as a fixed grid of moneyness (K/S) and time-to-expiration buckets. For example, 10 buckets for moneyness (e.g., 0.8, 0.85, ..., 1.2) and 5 buckets for expiration (e.g., <1 week, 1-2 weeks, 2-4 weeks, 1-2 months, >2 months). The LSTM would then predict the IV for each cell in this grid. This discretization was a simplification; more advanced methods like SVI parameterization exist, but seemed like overkill for a first attempt given my time constraints.

### The LSTM Adventure with TensorFlow

Once I had a somewhat manageable dataset of input sequences (my 1-minute engineered features) and corresponding target volatility surface grids (flattened into vectors), it was time to build the **TensorFlow LSTM**.

My initial architecture was quite simple:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

# Assuming n_timesteps, n_features, n_outputs (flattened surface size) are defined
# n_outputs = num_moneyness_bins * num_expiration_bins

model = Sequential()
model.add(LSTM(units=100, # A starting number, tuned later
               input_shape=(n_timesteps, n_features),
               return_sequences=True)) # True because I planned to stack another LSTM
model.add(Dropout(0.2)) # Trying to combat overfitting early on
model.add(BatchNormalization()) # Found this helped stabilize training

model.add(LSTM(units=75, return_sequences=False)) # False for the last LSTM layer before Dense
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(units=n_outputs)) # Linear activation for regression

model.compile(optimizer='adam', loss='mae') # MAE as per project goal
```
Getting the `input_shape` right for the first LSTM layer took an embarrassingly long time. I remember staring at TensorFlow errors about dimensions not matching for hours. StackOverflow posts about LSTM input shapes became my nightly reading. A common trip-up was forgetting that `(batch_size, timesteps, features)` is the expected input.

One of the major challenges was overfitting. Financial time series are notoriously noisy, and LSTMs are powerful enough to memorize noise if you're not careful. `Dropout` helped, as did `BatchNormalization`. I experimented with L1/L2 regularization too, but dropout seemed more effective for my setup. I also spent a lot of time tuning the number of units and layers. More layers didn't always mean better performance and significantly increased training time. My university’s GPU cluster access was a lifesaver here, though even then, each training run took several hours.

I also considered GRUs as an alternative to LSTMs, as I read they could be faster to train with similar performance. I even coded up a GRU version. For my specific dataset and task, the LSTM seemed to have a slight edge, so I stuck with it, but it was a close call. The TensorFlow documentation was generally good, but sometimes specific examples for multi-output regression with LSTMs were a bit sparse, leading to some trial and error.

A particularly frustrating period involved the LSTM outputting NaN values during training. This sent me down a rabbit hole of checking for NaNs in input data (which I thought I had cleaned meticulously), exploding gradients (tried gradient clipping, which seemed to help a bit), and learning rates. Reducing the Adam optimizer's learning rate and ensuring all input features were scaled properly (e.g., using `StandardScaler` from scikit-learn) eventually resolved this.

### Benchmarking: The GARCH Hurdle

To claim any improvement, I needed a solid benchmark. GARCH(1,1) is a workhorse in volatility modeling. The challenge was that GARCH typically models the volatility of a single time series (e.g., the underlying asset's returns). Predicting an entire *surface* with GARCH isn't straightforward.

My approach was to fit a separate GARCH(1,1) model for the at-the-money (ATM) volatility for several key expiration buckets. I used the `arch` library in Python for this, which is excellent.

```python
from arch import arch_model

# Example for a single ATM option series
# 'atm_vol_series' is a pandas Series of ATM implied volatility
# This had to be done for different relevant option series, or for the underlying returns
# which then would need a model to map to IV.
# I opted to fit GARCH on historical IV of ATM options directly for a more direct comparison.

# This loop is conceptual; I had to manage data for each option series
# for key_option_series in list_of_atm_series:
#   garch_model = arch_model(key_option_series.dropna(), vol='Garch', p=1, q=1)
#   garch_results = garch_model.fit(disp='off') # Turn off verbose output
#   forecast = garch_results.forecast(horizon=1)
#   predicted_atm_vol = forecast.variance.iloc[-1,0]**0.5
# This gave me ATM vol predictions.
```

For the rest of the surface (the smile/skew), I used a simpler approach for the GARCH benchmark: I assumed the shape of the smile relative to the ATM volatility would persist or follow a very simple model based on recent history. This was definitely a simplification, but building a full-blown GARCH-based surface model was a project in itself and beyond the scope of what I could reasonably benchmark against with my available time. The goal was to see if the LSTM could capture the dynamics of the *entire surface* better than a GARCH model focused on ATM volatility combined with simpler smile assumptions.

### The Results: A Promising Reduction in MAE

After weeks of tuning, training, and debugging, comparing the LSTM's out-of-sample predictions to the GARCH benchmark was the moment of truth. I calculated Mean Absolute Error (MAE) for each predicted point on the volatility surface grid and then averaged these MAEs.

The **TensorFlow LSTM** model achieved an MAE that was, on average, **7% lower** than my GARCH-based benchmark on the held-out **S&P 500 options data**. This was a significant moment!

Visually (I generated a lot of plots, though I can't embed them here), the LSTM seemed to capture shifts in the skew and smile more effectively than the GARCH benchmark, especially during periods of changing market sentiment. The GARCH model was decent at predicting the general level of ATM volatility but less responsive to changes in the surface's shape. The LSTM, having been trained on sequences leading up to the full surface, appeared to learn these cross-sectional relationships.

For instance, during a sharp market downturn in my test set, the LSTM-predicted surfaces showed a more pronounced steepening of the skew (higher demand for OTM puts) more quickly than the benchmark could adapt. There were still periods where GARCH was closer, particularly in very stable market regimes, but overall, the LSTM showed a clear advantage.

### Key Challenges and Learnings

*   **Data is King (and a Tyrant):** The sheer effort involved in preprocessing and managing high-frequency data cannot be overstated. This was easily the most time-consuming part. Any future work would involve investing even more in robust data pipelines.
*   **Hyperparameter Tuning is an Art and a Science (and a Lot of Patience):** Finding the right LSTM architecture, learning rate, batch size, etc., was an iterative process involving a lot of experimentation. Tools like KerasTuner could be useful, but I did most of it manually to get a better intuition.
*   **The "Black Box" Fear:** LSTMs can feel like black boxes. I spent considerable time trying to understand *why* certain predictions were made, looking at attention mechanisms (though I didn't formally implement a full attention layer in this version, it's on the list for future work) and feature importance.
*   **Computational Resources:** Training deep learning models on large datasets is computationally intensive. Access to GPUs was critical. Without it, the iteration cycle would have been impractically slow.
*   **Defining the "Surface":** The way I discretized the volatility surface was a practical choice, but it's an approximation. Exploring parametric models for the surface (like SVI mentioned earlier) as the LSTM output could be a more elegant solution.

### Conclusion and Next Steps

This project was a fantastic learning experience. Building a **TensorFlow LSTM** to predict volatility surfaces from **raw tick data** and achieving a **7% MAE reduction** over a GARCH benchmark felt like a real accomplishment. It showed me the potential of deep learning in a domain traditionally dominated by classical econometric models.

There’s still a lot to explore. Future enhancements could include:
*   More sophisticated feature engineering from tick data.
*   Using attention mechanisms within the LSTM.
*   Predicting parameters of a continuous volatility surface model instead of a discrete grid.
*   Testing on different asset classes.

For now, I’m just glad to have gotten this far and to have something tangible to show for countless hours of work. The world of quantitative finance and machine learning is incredibly deep, and this project has only scratched the surface (pun intended!).