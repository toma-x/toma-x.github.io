---
layout: post
title: Algorithmic Trading Backtester
---

Developing a Reliable Algorithmic Trading Backtester

Building my own backtesting engine has been a project I wanted to tackle for a while. The goal was to create a flexible tool to test trading strategies on historical equity data before even thinking about live trading. I wanted control over every aspect, from data handling to execution logic and performance analysis.

I started with Python, naturally, leaning on Pandas for data manipulation and NumPy for numerical operations. The initial challenge was handling tick data. This isn't like working with daily or even minute bars; you get every single trade and quote, resulting in massive datasets. Loading and processing even a few days of data for a single stock could mean dealing with millions of rows.

My first attempts at applying strategy logic involved iterating through the DataFrame row by row. This worked for small samples, but quickly became impossibly slow. I was trying something simple like:

```python
# Initial slow approach (simplified)
# Don't do this on large tick data!
data['signal'] = 0
for i in range(1, len(data)):
    # Example: Simple momentum condition
    if data['price'].iloc[i] > data['price'].iloc[i-1] * 1.001: # price increased by 0.1%
        data['signal'].iloc[i] = 1
    elif data['price'].iloc[i] < data['price'].iloc[i-1] * 0.999: # price decreased by 0.1%
        data['signal'].iloc[i] = -1
    # This was painfully slow for millions of rows.
```

It became clear I needed to embrace vectorization. Pandas operations are much faster when applied to entire columns or arrays rather than individual elements in a loop. I started rewriting my strategy logic to use things like `.shift()` and direct column arithmetic. Implementing strategies like a basic moving average crossover became much more performant this way. For example, calculating a rolling mean:

```python
# Better: Vectorized moving average
data['SMA_50'] = data['price'].rolling(window=50).mean()
# Then compare data['price'] to data['SMA_50'] for signals
```

This was better, but applying more complex conditions or simulating trades based on signals still involved logic that wasn't purely vectorized Pandas. Calculating performance metrics like Sharpe ratio or Sortino ratio also required iterating through simulated trades or daily returns, which could still be sluggish. I needed to aggregate tick data into bars or calculate returns efficiently. Calculating daily returns, for instance, meant resampling the tick data, which Pandas handles reasonably well with methods like `.resample()`.

The biggest hurdle, though, was simulating the actual trades and tracking position state over time on tick data. Even after generating signals using vectorized methods, the process of checking conditions at each tick, managing a hypothetical position (entering, exiting, tracking P/L), and accounting for transaction costs felt inherently sequential. My simulation loop, even when optimized as much as I knew how with Pandas, was still the bottleneck.

I remember spending a frustrating evening where a backtest on just a week of data for one liquid stock was taking over 10 minutes. I knew this wouldn't scale. I started searching for ways to speed up Python loops, especially those dealing with numerical operations. I looked into Cython briefly, but the idea of learning a new syntax and compilation step felt like overkill for this project's scope at the time. Then I stumbled upon Numba.

Numba is a JIT (Just-In-Time) compiler that can translate Python and NumPy code into fast machine code. This sounded perfect. The promise was that you could often speed up numerical functions just by adding a decorator like `@jit`. It wasn't quite that simple in practice, though. Numba works best on functions that primarily use NumPy arrays and basic Python types, without relying heavily on Pandas DataFrames or complex Python objects *inside* the jitted function.

My breakthrough came when I refactored the core simulation logic. Instead of passing the entire Pandas DataFrame into a Numba function, I extracted the necessary NumPy arrays (like timestamps, prices, signals). I then wrote a plain Python function that took these NumPy arrays and performed the step-by-step simulation logic within a loop. *Then*, I applied the `@jit(nopython=True)` decorator to *that* function. The `nopython=True` mode is stricter but provides the best performance boost as it ensures Numba can compile the entire function without falling back to the slower Python interpreter.

Here's a simplified idea of what the core simulation function, optimized with Numba, might look like:

```python
from numba import jit
import numpy as np

@jit(nopython=True) # Use nopython=True for best performance if possible
def run_simulation_numba(timestamps, prices, signals, initial_cash, transaction_cost_rate):
    # Need basic Python types or Numba-supported types inside
    cash = initial_cash
    position = 0.0 # Share quantity
    portfolio_value = initial_cash
    # Store results - pre-allocate arrays for performance in Numba
    trade_log = [] # Or better, use fixed-size numpy arrays if possible
    # ... need to figure out how to efficiently collect variable trade data in Numba ...
    # Okay, maybe collect minimal data and process outside Numba, or use Numba's typed.List

    # Let's simplify and just track portfolio value for now
    n = timestamps.shape
    portfolio_history = np.empty(n, dtype=np.float64)

    for i in range(n):
        current_price = prices[i]
        signal = signals[i]
        # Add more complex state like entry price, stop loss etc.
        # For this simplified example, just basic entry/exit

        # Calculate current portfolio value
        portfolio_value = cash + position * current_price
        portfolio_history[i] = portfolio_value # Storing value at each tick

        # Check signals and execute (simplified logic)
        if signal == 1 and cash > 0: # Buy signal
            # Let's buy with all cash for simplicity here
            # In reality, need to calculate max shares based on cash, price, transaction costs
            buy_shares = cash / (current_price * (1 + transaction_cost_rate))
            position += buy_shares
            cash -= buy_shares * current_price * (1 + transaction_cost_rate)
            # Need to log this trade... Numba lists are tricky, maybe append outside or pre-allocate
            # print(f"BUY at {timestamps[i]}: {current_price}") # Can't use print easily in nopython mode
        elif signal == -1 and position > 0: # Sell signal
            # Sell all shares
            sell_shares = position
            cash += sell_shares * current_price * (1 - transaction_cost_rate)
            position = 0.0
            # print(f"SELL at {timestamps[i]}: {current_price}")
            # Log trade...

        # This loop structure is where Numba shines compared to Python's interpreter loop.

    return portfolio_history # Return results processed outside

# In the main Python code:
# Extract arrays from Pandas DataFrame
# timestamps_arr = data['timestamp'].values # Numba likes numpy arrays
# prices_arr = data['price'].values
# signals_arr = data['signal'].values
# initial_cash = 10000.0
# transaction_cost = 0.001 # 0.1% per side
#
# portfolio_history = run_simulation_numba(timestamps_arr, prices_arr, signals_arr, initial_cash, transaction_cost)
#
# Now convert portfolio_history back to Pandas Series with original index for analysis
# portfolio_series = pd.Series(portfolio_history, index=data.index)
```

This refactoring was key. Moving the tight loop that processes tick by tick into a Numba-jitted function dramatically reduced the simulation time from minutes to seconds for the same dataset. There was a learning curve with Numba, specifically understanding what operations and data types it supports well in `nopython` mode, and how to handle things like collecting variable amounts of trade data (I ended up collecting minimal data in Numba and doing more complex logging outside the jitted function). Stack Overflow and the Numba documentation were essential here.

Beyond performance, I focused on the architecture. I wanted to easily swap out strategies or data sources. This led to designing the backtester with distinct components: a data loader, a strategy module (each strategy is a class with `generate_signals` method), an execution engine (the Numba part), and a performance analysis module. This modularity, though requiring more initial planning, made adding a new strategy just a matter of writing a new class following a simple interface.

Implementing the performance metrics like Sharpe and Sortino ratios involved calculating daily returns from the portfolio value history (which I got from the Numba simulation), then using standard statistical formulas. Pandas' rolling window functions and statistical methods made this part relatively straightforward once I had the return series.

Overall, this project was a deep dive into performance optimization in Python for data-intensive tasks. Dealing with tick data forced me to confront the limitations of pure Python and even Pandas for sequential operations, leading me to discover and effectively use Numba. The process involved a lot of trial and error, realizing initial designs were too slow, profiling code, and refactoring critical sections. It was challenging, but the result is a backtesting engine that is not only functional but fast enough to actually be useful for testing strategies on realistic data volumes.