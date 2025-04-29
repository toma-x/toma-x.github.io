---
layout: post
title: Algorithmic Strategy Backtester
---

My recent project has been the development of a quantitative backtesting platform. This wasn't just an academic exercise; I wanted a tangible tool to explore systematic trading strategies and understand their performance characteristics based on historical data. The goal was to build something from the ground up, allowing me to control every aspect of the simulation and truly grasp what goes into evaluating a trading idea before ever considering its real-world application.

The core technologies I settled on were Python, Pandas, and NumPy. Python, being versatile and having a rich ecosystem, was an obvious choice. Pandas was essential for handling time-series financial data; its DataFrames provide a robust structure for organizing historical prices, indicators, and trade signals. NumPy was needed for the numerical heavy lifting, particularly for potentially vectorizing calculations to improve performance when processing large datasets.

Getting started, the first hurdle was data. Where to get reliable historical data, and how to load it efficiently? I initially experimented with scraping data from free sources, but the inconsistency and incompleteness were major headaches. Eventually, I found a reasonably accessible source providing historical daily price data in CSV format. Loading this into Pandas was straightforward using `pd.read_csv()`. However, ensuring the data was clean – dealing with missing values, verifying dates, and making sure the columns (Open, High, Low, Close, Volume) were correctly formatted – took more time than I expected. I wrote a small helper function just for this, which felt tedious at the time but prevented many future errors.

```python
import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """Loads CSV data, sets index, and performs basic cleaning."""
    try:
        df = pd.read_csv(filepath)
        # Assuming the CSV has 'Date' and standard OHLCV columns
        if 'Date' not in df.columns:
            print(f"Error: 'Date' column not found in {filepath}")
            return None

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index() # Ensure chronological order

        # Simple check for missing values and handle if necessary (e.g., forward fill)
        if df.isnull().sum().sum() > 0:
            print(f"Warning: Missing values found in {filepath}. Attempting forward fill.")
            df = df.fillna(method='ffill') # One approach, might not be best for all data

        # Drop rows with any remaining NaNs after ffill (e.g., at the very beginning)
        df = df.dropna()

        # Basic data type check (ensure numerical columns are numeric)
        num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in num_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce invalid parsing to NaN
        df = df.dropna(subset=num_cols) # Drop rows where key columns couldn't be converted

        print(f"Successfully loaded and cleaned data from {filepath}")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Example usage:
# data_file = 'path/to/my/stock_data.csv'
# price_data = load_and_clean_data(data_file)
# if price_data is not None:
#     print(price_data.head())
```
This function, while basic, was my first attempt at standardizing the data input. I quickly realized that simply filling missing values with the previous day's price (`ffill`) might not be the best approach for all scenarios, especially low-volume assets, but for the initial phase focusing on common stocks, it was "good enough" to move forward.

Next came the strategy implementation. I decided to structure strategies as Python classes. Each strategy would take the price data and parameters upon initialization and have a method to generate trading signals (buy, sell, or hold) for each point in time. My initial idea was to iterate through the data day by day *within* the strategy class to calculate indicators and signals.

```python
# Initial, naive strategy concept
class SimpleMovingAverageStrategy_V1:
    def __init__(self, data, short_window, long_window):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signals = pd.DataFrame(index=data.index)
        self.signals['signal'] = 0 # Default to hold

    def generate_signals(self):
        # Calculate moving averages *within the loop*? Bad idea.
        # This was my initial thought process, which I later realized was inefficient
        # and prone to lookahead bias if not extremely careful.

        # A better way is to use Pandas built-in rolling()
        self.signals['short_mavg'] = self.data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        self.signals['long_mavg'] = self.data['Close'].rolling(window=self.long_window, min_periods=1).mean()

        # Generate signals based on crossover
        # When short crosses above long -> buy (signal = 1)
        self.signals['signal'][self.signals['short_mavg'] > self.signals['long_mavg']] = 1

        # When short crosses below long -> sell (signal = -1)
        self.signals['signal'][self.signals['short_mavg'] < self.signals['long_mavg']] = -1

        # Calculate the actual trade order - differences indicate trades
        # A buy signal is 1, sell is -1. diff() finds points where signal changes
        # This took me a while to figure out. The .diff() method is key here.
        self.signals['positions'] = self.signals['signal'].diff()

        # Need to make sure first signal isn't interpreted as a position change
        # (e.g., if first signal is 1, diff() will be 1, looks like a buy)
        # The first position should typically be 0 unless explicitly entering at start.
        # This handling of the first signal was confusing initially.
        # Setting the first value of 'positions' to 0 if the first 'signal' isn't 0
        # This is a common point of error/confusion in backtesting.
        if self.signals['signal'].iloc[0] != 0:
             # If the initial signal isn't neutral, we might need to adjust the first position entry
             # For simplicity, let's assume we start flat (no position)
             self.signals['positions'].iloc[0] = 0 # Ensure we don't start with an implicit trade
        # Or maybe the logic needs refinement based on specific requirements...
        # This part felt a bit hand-wavy initially, needing iteration to get right.
        # Stack Overflow had many discussions on handling initial positions correctly.

        return self.signals

# Example usage:
# if price_data is not None:
#     sma_strategy = SimpleMovingAverageStrategy_V1(price_data, short_window=50, long_window=200)
#     strategy_signals = sma_strategy.generate_signals()
#     print(strategy_signals.head())
#     print(strategy_signals['positions'].value_counts()) # See how many trades there are
```
The realization that I could use Pandas' built-in `rolling()` and vectorized operations like boolean indexing and `diff()` was a significant breakthrough. My first thought was a slow loop, calculating averages manually day by day. When I found the Pandas methods, it drastically simplified the code and improved performance. The concept of using `diff()` on the signals to identify actual trade entry/exit points (`1` for buy entry, `-1` for sell entry, `-1` for buy exit, `1` for sell exit) was something I pieced together from examples online, particularly on quantitative finance blogs and Stack Overflow. Getting the *first* trade right, especially if the strategy starts with a non-zero signal, was a specific point of confusion that required debugging and referencing examples. I decided to explicitly set the first 'positions' value to 0 to assume starting flat, which simplified the initial backtesting runs, though I knew it wasn't perfect for all scenarios.

The core of the backtester is the simulation loop that processes these signals and tracks positions, equity, and trades. This involved iterating through the `signals` DataFrame generated by the strategy. For each time step, I check the `positions` column to see if a trade is triggered (value is 1 or -1). If so, I calculate the number of shares to buy or sell based on available capital and the current price, update the position, and record the trade details (price, quantity, direction, timestamp).

```python
# Simplified backtest execution loop logic
def run_backtest(price_data, signals, initial_capital=100000.0):
    equity_curve = pd.DataFrame(index=signals.index)
    equity_curve['capital'] = initial_capital
    position = 0 # Number of shares held
    cash = initial_capital
    trades = [] # List to store trade details

    # Loop through each bar (daily data point)
    # This loop is necessary to track state (position, cash) over time
    # While some parts of backtesting can be vectorized, the core simulation of trades
    # often requires iteration due to dependency on previous state.
    # I initially tried to avoid the loop entirely with more vectorization,
    # but realized quickly that trade execution logic (slippage, commissions, capital constraints)
    # makes a sequential process more realistic and manageable for a first version.
    # Iterating through Pandas rows can be slow, but using iterrows() or
    # converting relevant series to NumPy arrays before the loop helps a bit.
    # For this simplified version, iterrows is okay for clarity, though not the most performant
    # for huge datasets. A more optimized version might convert signals and prices
    # to NumPy arrays and use index lookups.

    for i, (date, signal_row) in enumerate(signals.iterrows()):
        current_price = price_data.loc[date, 'Close'] # Assuming trades execute at Close
        trade_signal = signal_row['positions'] # 1 for buy entry, -1 for sell entry/buy exit

        # Calculate current portfolio value
        current_value = cash + position * current_price
        equity_curve.loc[date, 'capital'] = current_value # Record equity at end of day

        # Check for trade signal
        if trade_signal == 1: # Buy signal
            # For simplicity, buy with all available cash
            # A real backtester needs more sophisticated order sizing/management
            if cash > 0:
                # Calculate how many shares we can buy
                # integer=True argument added after finding issues with fractional shares
                shares_to_buy = int(cash / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    # Assuming no commission/slippage for simplicity initially
                    # This was a conscious decision to simplify for the first working version.
                    # Adding these later would involve deducting from cash/equity.
                    cash -= cost
                    position += shares_to_buy
                    trades.append({
                        'date': date,
                        'type': 'buy',
                        'price': current_price,
                        'qty': shares_to_buy,
                        'value': cost
                    })
                    # print(f"{date.date()}: BUY {shares_to_buy} shares at {current_price:.2f}") # Debug print

        elif trade_signal == -1: # Sell signal (either exiting a long or entering a short)
            # For this example, let's assume it's always an exit from a long position
            # Handling short selling would add complexity.
            if position > 0:
                sell_value = position * current_price
                cash += sell_value
                # Again, no commission/slippage initially
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': current_price,
                    'qty': position, # Sell the entire position
                    'value': sell_value
                })
                # print(f"{date.date()}: SELL {position} shares at {current_price:.2f}") # Debug print
                position = 0 # Exit position

    # After the loop, if we still hold a position, liquidate it at the final price
    if position > 0:
         final_price = price_data['Close'].iloc[-1]
         sell_value = position * final_price
         cash += sell_value
         trades.append({
             'date': price_data.index[-1],
             'type': 'sell (final)',
             'price': final_price,
             'qty': position,
             'value': sell_value
         })
         # print(f"{price_data.index[-1].date()}: FINAL SELL {position} shares at {final_price:.2f}")

    # Final equity update
    equity_curve.iloc[-1] = cash # Position is 0 after liquidation

    trades_df = pd.DataFrame(trades)
    return equity_curve, trades_df

# Example usage:
# if price_data is not None and strategy_signals is not None:
#      equity_curve, trades_record = run_backtest(price_data, strategy_signals)
#      print("\nEquity Curve Head:")
#      print(equity_curve.head())
#      print("\nTrades Record Head:")
#      print(trades_record.head())
```

Simulating trades day-by-day in a loop felt necessary to correctly track the changing cash and position. My initial instinct was to somehow vectorize this entirely, but I quickly ran into issues trying to calculate cumulative cash flow and positions without referencing the previous day's state. A sequential loop, while less performant than pure vectorization, was significantly easier to implement correctly and debug for this specific task of simulating state changes. The decision to execute trades at the closing price was a simplification; a more advanced version would consider limit/market orders and intraday prices, but for now, closing price was sufficient for daily data. Handling fractional shares was a minor fix – initially, I just used floating-point numbers for shares, which isn't realistic, so casting to `int` for `shares_to_buy` was a small but necessary adjustment.

Calculating performance metrics was the final piece. This involved analyzing the `equity_curve`. Metrics like total return were simple (`(final_equity / initial_capital) - 1`). Calculating maximum drawdown was more complex. I initially tried a manual loop, keeping track of the peak seen so far, but this was slow and error-prone. A Stack Overflow search led me to the Pandas `expanding().max()` method, which is perfect for finding the peak value up to each point in time.

```python
def analyze_performance(equity_curve, initial_capital):
    """Calculates basic performance metrics."""
    if equity_curve.empty:
        print("Equity curve is empty. Cannot analyze performance.")
        return {}

    total_return = (equity_curve.iloc[-1]['capital'] / initial_capital) - 1.0
    # Annualized return requires knowing the time span, assuming daily data for simplicity
    # This calculation is an approximation and depends on the data frequency/length
    num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    annualized_return = (1 + total_return)**(1/num_years) - 1 if num_years > 0 else total_return

    # Max Drawdown Calculation
    # This is where expanding().max() was a lifesaver compared to manual loops.
    # Peak value up to each point in time
    peak_value = equity_curve['capital'].expanding().max()
    # Current drawdown is the difference from the peak
    drawdown = equity_curve['capital'] - peak_value
    # Drawdown percentage
    drawdown_percent = drawdown / peak_value
    max_drawdown = drawdown_percent.min() # Find the minimum (most negative) drawdown percentage

    # Sharpe Ratio calculation (requires risk-free rate, often assumed 0 for simplicity in backtests)
    # Need daily returns first
    equity_returns = equity_curve['capital'].pct_change().dropna()
    # Daily risk-free rate (assuming 0 for simplicity)
    risk_free_rate_daily = 0.0
    excess_returns = equity_returns - risk_free_rate_daily
    # Annualize Sharpe Ratio - sqrt(annualization factor) * mean(excess_returns) / std(excess_returns)
    # Annualization factor for daily data is sqrt(252) assuming 252 trading days/year
    # Need to be careful about data frequency here. If it's not daily, the factor changes.
    # Assuming daily data for this calculation.
    annualization_factor = np.sqrt(252) # Assuming 252 trading days a year
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * annualization_factor if excess_returns.std() != 0 else np.nan


    performance_metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Maximum Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
        # Could add many more metrics: Sortino Ratio, Calmar Ratio, Win Rate, Avg Trade P/L, etc.
    }

    return performance_metrics

# Example usage:
# if equity_curve is not None and not equity_curve.empty:
#     initial_capital = 100000.0 # Make sure this matches the backtest input
#     metrics = analyze_performance(equity_curve, initial_capital)
#     print("\nPerformance Metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")
```

Calculating the Sharpe Ratio brought up questions about annualization factors and how to handle the risk-free rate. For simplicity, I initially set the risk-free rate to zero, which is common in basic backtests but not entirely realistic. The annualization factor depends on the data frequency (252 for daily data is a common assumption), and ensuring I used the correct factor for my specific data was important. This part required consulting documentation on Sharpe Ratio calculation in backtesting contexts.

Overall, building this backtester from scratch, while challenging, provided invaluable insight into the mechanics of quantitative trading strategy evaluation. I encountered numerous small problems – data formatting inconsistencies, off-by-one errors in signal generation, subtle bugs in trade simulation logic, performance bottlenecks with large datasets, and getting the performance metric calculations exactly right. Each problem required digging into documentation, searching forums (Stack Overflow was a frequent destination), and careful debugging. It highlighted that quantitative development is as much about meticulous implementation and handling real-world data complexities as it is about the trading ideas themselves.

Future work involves adding more realistic order execution models (slippage, commissions), implementing stop-loss and take-profit logic, enabling short selling, and potentially adding optimization capabilities to find the best strategy parameters. Visualizing the equity curve and trades is also a high priority for easier analysis. But for now, having a functional platform capable of testing basic systematic ideas using real historical data feels like a significant step. It's a tool I built, I understand its limitations, and I know exactly how the sausage is made, which is exactly what I set out to achieve.