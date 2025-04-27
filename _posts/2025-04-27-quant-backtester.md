---
layout: post
title: Quantitative Backtesting Platform
---

Alright, finished the quantitative backtesting platform project I've been chipping away at. The goal was to build something that could actually handle a decent amount of historical data without grinding to a halt, unlike some initial attempts with basic Python loops on large files. The idea was to get a feel for building a pipeline from data storage to strategy execution and performance analysis, specifically aiming for speed when processing years of minute-level data.

The core problem was processing large time series datasets quickly. My initial thought was just standard Pandas, but after running some tests with datasets hitting into the gigabytes, it was clear that wasn't going to scale well on my machine for iterative backtests. Memory usage was a significant constraint.

I started looking into alternatives for fast data manipulation. Apache Arrow seemed promising, and Polars uses it under the hood. I saw some benchmarks comparing Polars and Pandas, and the performance gains, especially with lazy evaluation and multi-threading built-in, looked substantial for tabular data operations which is exactly what I needed for historical price data. Decided to commit to using **Polars** for the heavy lifting of data loading, cleaning, and transformation.

For data storage, I needed something more robust than CSVs or flat files. A relational database made sense for organizing different instruments, timeframes, and potentially strategy results. **PostgreSQL** is free, powerful, and widely used, so that felt like a solid choice. The integration point between Polars and PostgreSQL became important – how to get data *into* Polars DataFrames efficiently from the DB and results back *out*.

Getting the initial data loaded was a bit of a hurdle. I had raw historical data in various formats. Writing Python scripts to parse these and insert them into PostgreSQL required careful handling of data types and ensuring no duplicates. I ended up writing a small data ingestion script that would read chunk by chunk, clean timestamps, and use `psycopg2`'s `execute_values` for faster bulk inserts rather than individual `INSERT` statements, which was agonizingly slow at first.

Here's a snippet of how I'd load data from the DB into Polars for a specific symbol and timeframe:

```python
import polars as pl
import psycopg2
from datetime import datetime

# Connection details (simplified)
db_params = {
    "database": "quant_db",
    "user": "myuser",
    "password": "mypassword",
    "host": "localhost",
    "port": "5432",
}

def load_historical_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pl.DataFrame:
    """Loads historical bar data for a given symbol and timeframe from PostgreSQL."""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # This query structure evolved quite a bit. Initially, I just did SELECT *,
        # but explicitly selecting columns and ordering is better practice.
        # Filtering by time is crucial for performance.
        query = """
        SELECT
            timestamp, open, high, low, close, volume
        FROM
            historical_data
        WHERE
            symbol = %s AND timeframe = %s AND timestamp BETWEEN %s AND %s
        ORDER BY
            timestamp;
        """
        cursor.execute(query, (symbol, timeframe, start_date, end_date))

        # Using Polars read_database was simpler than fetching all rows and then creating DataFrame
        # Initially tried fetching all and then pl.DataFrame(), but Polars read_database
        # seems optimized for directly reading from DB connections.
        # Had an issue here where timestamps weren't always timezone-aware initially,
        # which caused problems with time-based filtering and joining later.
        # Had to ensure data stored in DB and loaded by Polars was consistent (UTC).
        data_df = pl.read_database(conn, query, parameters=(symbol, timeframe, start_date, end_date))

        # Ensure timestamp is the correct type and sorted, though ORDER BY should handle sorting
        data_df = data_df.with_columns(pl.col("timestamp").cast(pl.Datetime))
        # data_df = data_df.sort("timestamp") # Redundant if ORDER BY is used and respected

        return data_df

    except (psycopg2.Error, pl.exceptions.PolarsError) as e:
        print(f"Error loading data: {e}")
        return pl.DataFrame() # Return empty DataFrame on error
    finally:
        if conn:
            conn.close()

# Example usage:
# start = datetime(2020, 1, 1)
# end = datetime(2021, 1, 1)
# stock_data = load_historical_data("AAPL", "1Min", start, end)
# print(stock_data.head())
```

The major challenge was implementing trading strategies in a **vectorized** manner. Traditional backtesters often loop through each bar sequentially, applying strategy logic. This is simple to code but incredibly slow on large datasets. A vectorized approach means expressing the strategy logic as operations on entire columns or series of data at once. Polars is built for this.

For instance, calculating a simple Moving Average crossover strategy. Instead of a loop, you calculate the MAs for the entire dataset using Polars' rolling window functions, then compare the resulting columns directly.

```python
def implement_moving_average_crossover(data_df: pl.DataFrame, short_period: int = 20, long_period: int = 50) -> pl.DataFrame:
    """
    Calculates signals for a simple Moving Average Crossover strategy using Polars.
    Assumes input DataFrame has a 'close' column and is sorted by timestamp.
    """
    if data_df.is_empty():
        return data_df.with_columns(pl.lit(0).alias("signal")) # Add signal column even if empty

    # Calculate SMAs using rolling_mean. This is where Polars shines for speed.
    # Was initially confused about how windowing works in Polars vs Pandas,
    # spent some time in the documentation and trying examples.
    # The `min_periods` argument is important to avoid NaNs at the start.
    data_df = data_df.with_columns([
        pl.col("close").rolling_mean(window_size=short_period, min_periods=1).alias(f"sma_{short_period}"),
        pl.col("close").rolling_mean(window_size=long_period, min_periods=1).alias(f"sma_{long_period}")
    ])

    # Generate signals based on crossover.
    # 1 for buy signal (short crosses above long), -1 for sell signal (short crosses below long).
    # Using `when().then().otherwise()` pattern for conditional logic.
    # The `.shift(1)` is crucial to compare current MA values with previous ones
    # to detect the *crossover event*, not just the state.
    # My first attempt didn't use shift and just compared current values, leading to signals on every bar
    # where short > long, which is not what a crossover strategy does.
    # Found the solution involving shift on StackOverflow while searching for "polars detect crossover".
    data_df = data_df.with_columns(
        pl.when(
            (pl.col(f"sma_{short_period}") > pl.col(f"sma_{long_period}")) &
            (pl.col(f"sma_{short_period}").shift(1) <= pl.col(f"sma_{long_period}").shift(1))
        )
        .then(pl.lit(1)) # Buy signal
        .when(
            (pl.col(f"sma_{short_period}") < pl.col(f"sma_{long_period}")) &
            (pl.col(f"sma_{short_period}").shift(1) >= pl.col(f"sma_{long_period}").shift(1))
        )
        .then(pl.lit(-1)) # Sell signal
        .otherwise(pl.lit(0)) # No signal
        .alias("signal")
    )

    return data_df

# Example usage:
# data_with_signals = implement_moving_average_crossover(stock_data)
# print(data_with_signals.filter(pl.col("signal") != 0).head())
```

The `signal` column now indicates when a trade should ideally be entered based on the crossover rule. Backtesting then involves iterating through these signals, managing positions, and calculating P&L. This part still requires some careful sequential logic (handling open positions, calculating trade-by-trade profit), but the heavy data manipulation (calculating indicators) is done efficiently upfront.

A significant breakthrough moment was realizing how to correctly handle the "state" of the backtest. While indicator calculation is vectorized, the actual trade execution (entry, exit, position sizing, calculating P&L) is inherently sequential. You can't fully vectorize "buy if signal=1 and I'm not already in a position". I ended up structuring the backtest engine to iterate through the data *after* indicators and signals were calculated, applying a simple state machine (no position, long position, short position). This hybrid approach leveraged Polars for speed where possible while handling the necessary sequential logic.

Another point of confusion was performance tuning. Initially, loading data into Polars was fast, but subsequent operations sometimes felt slow. I learned about Polars' lazy API (`scan_parquet`, `scan_csv`, `scan_ipc`, etc., although I was primarily loading from DB) and explicitly collecting results only when needed. For database loads via `read_database`, it fetches eagerly, but for complex multi-step transformations *after* loading, using lazy execution on the DataFrame could potentially improve performance by optimizing the query plan. I didn't fully implement a lazy evaluation pipeline from the DB source in this version, sticking mostly to eager Polars operations on loaded DataFrames, but it's something to explore for future work.

Overall, building this provided a solid understanding of handling financial time series data, the power of columnar processing libraries like Polars, and the importance of vectorization for performance. It also highlighted the practical challenges of integrating different tools (Python, Polars, PostgreSQL) and translating theoretical strategy ideas into concrete, efficient code. There are many areas for improvement, like adding more sophisticated order execution models (slippage, commissions) and performance metrics, but the core engine feels robust enough to start testing actual strategy ideas against historical data at a reasonable speed.