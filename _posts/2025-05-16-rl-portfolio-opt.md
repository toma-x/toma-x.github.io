---
layout: post
title: Portfolio Optimization via RL
---

## Portfolio Optimization via Reinforcement Learning: A Deep Dive

This project has been quite a journey. For a while now, I've been fascinated by the potential of Reinforcement Learning (RL) beyond games, and financial markets seemed like a challenging and interesting application domain. The goal was to develop an RL agent capable of performing dynamic asset allocation, hopefully outperforming simpler strategies. I decided to document the process, the struggles, and the eventual (small) victories.

### The Foundation: Data and Tools

First things first: data. Any trading strategy, RL-based or not, is only as good as the data it's trained and tested on. I opted for **Quandl** for historical market data. I'd used it for a previous project and found their API relatively easy to work with, plus they have a good range of datasets. Specifically, I decided to focus on a few major stock tickers from their WIKI EOD (End of Day) stock prices dataset. My thinking was to start with a manageable number of assets before trying to conquer the entire S&P 500.

Getting the data into a usable format was the initial step. I pulled adjusted close prices, volume, and a few other common features for a selection of tickers (AAPL, MSFT, GOOG, AMZN) over a period of about 10 years. I stored this locally in a SQLite database. This made querying and preprocessing much easier than juggling multiple CSV files.

Here’s a rough idea of how I was pulling and structuring some of that data using Python and the `quandl` library, then pushing to SQL:

```python
import quandl
import pandas as pd
import sqlite3

# طبعا، مفتاح API الخاص بي محذوف هنا
# quandl.ApiConfig.api_key = "YOUR_API_KEY"

# tickers = ['WIKI/AAPL', 'WIKI/MSFT', 'WIKI/GOOG', 'WIKI/AMZN']
# data_frames = []
# for ticker in tickers:
#     try:
#         data = quandl.get(ticker, start_date="2010-01-01", end_date="2023-12-31")
#         # I was mostly interested in adjusted close for price action
#         data_frames.append(data[['Adj. Close']].rename(columns={'Adj. Close': ticker.split('/')}))
#     except Exception as e:
#         print(f"Error fetching {ticker}: {e}") # Ran into a few of these with less common tickers initially

# if data_frames:
#     df_prices = pd.concat(data_frames, axis=1).dropna()
#     # df_prices.to_sql('stock_prices', conn, if_exists='replace', index=True)
#     # conn.close()
```
I remember spending some time making sure the data was aligned correctly by date, as different stocks might have missing data points on different days. `dropna()` was a bit too aggressive initially, so I had to be more careful with forward-filling or aligning data from different Quandl tables.

### The Core: Building the Reinforcement Learning Agent

This was the most challenging and, honestly, the most exciting part. The idea was to frame the asset allocation problem as an RL task:
*   **Environment:** The stock market.
*   **Agent:** Our trading algorithm.
*   **State:** Information about the current market conditions and our portfolio (e.g., recent price history, current allocations).
*   **Action:** How to reallocate the portfolio (e.g., shift X% from asset A to asset B).
*   **Reward:** The change in portfolio value, or perhaps something more sophisticated like the Sharpe ratio over a period.

I decided to use Python, given its strong ecosystem for both finance and machine learning. For the RL framework, I looked into a few options. Implementing algorithms like Q-learning or DQN from scratch for continuous action spaces seemed daunting for a first attempt at this scale. I eventually settled on **Stable Baselines3**, which provides pre-built implementations of common RL algorithms. After reading some documentation and a few comparison blog posts, I leaned towards PPO (Proximal Policy Optimization) due to its reputation for being relatively robust and sample-efficient. A guide on Medium comparing PPO and A2C for financial tasks was particularly insightful, though I can't find the exact link right now – it highlighted PPO's stability.

#### Crafting the Environment

The custom Gym environment was where most of the initial heavy lifting happened. My `StockTradingEnv` needed to:
1.  Load historical price data.
2.  Maintain the current portfolio state (cash and holdings).
3.  Process agent actions (buy/sell orders based on target allocations).
4.  Calculate rewards.
5.  Provide the next state observation.

My state representation was a window of the past `N` days' price changes for each asset, plus the current portfolio weights. For actions, the agent would output a vector of desired portfolio weights for the next period. These weights would then be translated into buy/sell orders, considering transaction costs (a small percentage, say 0.1%).

Here's a snippet from my environment's `step` method. It's simplified, but gives the idea:

```python
# Inside my custom Gym environment class
# def step(self, action):
#     # action is a numpy array of target weights
#     # Normalize actions to sum to 1 (or close enough after clipping)
#     # current_price_vector = self.df.iloc[self.current_step]
#     # previous_portfolio_value = self._calculate_portfolio_value(self.current_holdings, current_price_vector)

#     # Simulate trades to reach target_weights, considering transaction_costs
#     # self.current_holdings = self._execute_trades(action, current_price_vector)
    
#     self.current_step += 1
#     if self.current_step >= self.max_steps:
#         self.done = True

#     next_observation = self._get_observation()
#     next_price_vector = self.df.iloc[self.current_step]
#     current_portfolio_value = self._calculate_portfolio_value(self.current_holdings, next_price_vector)
#     reward = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value # Simple percentage return

    # info = {'portfolio_value': current_portfolio_value}
    # return next_observation, reward, self.done, info
```

The reward function was a major point of iteration. I started with simple profit/loss per step. This led to some erratic behavior – the agent might take huge risks for a small chance of a big reward. I then tried incorporating a Sharpe ratio-like component into the reward to penalize volatility, but calculating it meaningfully on a per-step basis was tricky. I eventually settled on a reward based on the log return of the portfolio, with a small penalty for excessive trading (high turnover) to account for transaction costs implicitly. I found a forum discussion (I think on QuantConnect or a similar platform) arguing that log returns are often better for long-term growth objectives in RL for finance.

#### Training Pains

Training was... an experience. My first few runs with PPO were disastrous. The agent would either learn to do nothing (hold all cash) or make wild, random trades. I spent a lot of time tweaking hyperparameters: `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`. The Stable Baselines3 documentation is good, but finding the right combination for a custom financial environment is more art than science.

One breakthrough came when I realized my observation normalization was off. Stock prices have very different scales, and feeding raw prices or simple returns without proper scaling was confusing the agent. I switched to using percentage changes and then standardizing the observation window.

Another hurdle was the sheer time it took to train. Even with a few years of daily data, running hundreds of thousands of timesteps took hours. I made heavy use of `tensorboard_log` in Stable Baselines3 to monitor training progress. Seeing the `ep_rew_mean` (mean episode reward) slowly, agonizingly, start to trend upwards was a huge relief. There was a point where it plateaued for ages, and I was convinced there was a bug in my reward calculation or state representation. I specifically remember a Stack Overflow question, something like "PPO agent not learning in custom Gym environment," which had a checklist of common pitfalls. Going through that list, I found I hadn't properly handled the `done` signal when the agent ran out of money or data.

This is a rough idea of the training script:
```python
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stock_trading_env import StockTradingEnv # My custom environment
# import pandas as pd

# # df_prices = pd.read_sql('SELECT * FROM stock_prices', conn_to_my_db) # Pseudocode for db access

# # env_kwargs = {'df': df_prices, 'initial_balance': 100000, 'window_size': 30}
# # env = make_vec_env(StockTradingEnv, n_envs=4, env_kwargs=env_kwargs) # Using multiple environments for faster training

# model = PPO("MlpPolicy", 
#             env, 
#             verbose=1, 
#             tensorboard_log="./ppo_stock_tensorboard/",
#             learning_rate=0.0003, # This took a lot of tuning
#             n_steps=2048,
#             batch_size=64,
#             gamma=0.99,
#             gae_lambda=0.95
# )

# # model.learn(total_timesteps=500000)
# # model.save("ppo_stock_trader")
```
The choice of `MlpPolicy` (Multi-Layer Perceptron Policy) was standard for this kind of vectorized input. I considered LSTMs for potentially capturing longer-term dependencies, but wanted to get a simpler MLP working first.

### Backtesting the Strategy

Once I had a trained model that seemed somewhat promising (i.e., it wasn't losing all its money immediately in TensorBoard), the next step was rigorous backtesting on data it hadn't seen during training. This is crucial – it's easy to overfit to the training period.

My backtesting script loaded the saved model and ran it through a separate period of historical data. I calculated standard performance metrics:
*   Total Return
*   Annualized Return
*   Annualized Volatility
*   Sharpe Ratio (risk-free rate assumed to be low, like 1-2%)
*   Max Drawdown

The backtesting loop was similar to the environment's `step` logic, but without the learning aspect – just taking actions based on the trained policy.

```python
# # Backtesting snippet
# # model = PPO.load("ppo_stock_trader")
# # test_df = load_my_test_data() # Data not seen by the model during training
# # test_env = StockTradingEnv(df=test_df, initial_balance=100000, window_size=30)

# obs = test_env.reset()
# done = False
# portfolio_values = [test_env.initial_balance]
# asset_allocations_over_time = []

# while not done:
#     action, _states = model.predict(obs, deterministic=True) # Use deterministic actions for backtesting
#     obs, reward, done, info = test_env.step(action)
#     portfolio_values.append(info['portfolio_value'])
    # asset_allocations_over_time.append(test_env.current_holdings.copy()) # Or however I stored weights

# # Then calculate metrics based on portfolio_values
# # import numpy as np
# # returns = np.diff(portfolio_values) / portfolio_values[:-1]
# # sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) # Assuming daily data
```
Comparing the RL agent's performance to a simple benchmark like "buy and hold" the S&P 500 (e.g., SPY ETF) was sobering. Initially, my agent often underperformed, especially after accounting for transaction costs. This led to more iterations on the reward function and feature engineering. Adding features like moving average convergence divergence (MACD) or relative strength index (RSI) to the state representation did seem to help the agent make slightly more informed decisions.

### Visualization with Plotly Dash

Staring at numbers and printouts gets old fast. I wanted a way to visualize the agent's performance and decisions. I decided to use **Plotly Dash** because I'd seen some impressive interactive dashboards built with it, and it integrates well with Python and Pandas.

My dashboard had a few key components:
1.  A line chart showing the portfolio value over time, compared to a benchmark.
2.  A stacked area chart showing the allocation of assets in the portfolio over time.
3.  Perhaps some summary statistics.

Setting up the Dash app took a bit of fiddling with layouts and callbacks. The callback to update the graphs based on the backtesting results was the core of it.

Here's a taste of what a simple layout might look like:
```python
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.express as px

# # Assume backtest_results_df has columns like 'Date', 'PortfolioValue', 'BenchmarkValue', and asset weights
# # app = dash.Dash(__name__)

# # app.layout = html.Div([
# #     html.H1("RL Portfolio Optimization Results"),
# #     dcc.Graph(id='portfolio-performance-graph'),
# #     dcc.Graph(id='asset-allocation-graph')
# #     # Potentially add controls for selecting different backtest runs later
# # ])

# # @app.callback(
# #     Output('portfolio-performance-graph', 'figure'),
# #     [Input('some-dummy-input-for-now', 'value')] # Or trigger on load
# # )
# # def update_performance_graph(dummy_value):
# #     # fig = px.line(backtest_results_df, x='Date', y=['PortfolioValue', 'BenchmarkValue'],
# #     #               title="Portfolio Value vs. Benchmark")
# #     # return fig
# # # Similar callback for asset_allocation_graph using px.area
```
The most challenging part with Dash was making the graphs update dynamically if I wanted to, for example, re-run a backtest with different parameters via the dashboard itself. For this project, I mostly used it to display pre-computed backtest results. Getting the data formatted correctly for Plotly Express functions (like `px.line` or `px.area`) often required some DataFrame manipulation that I initially underestimated. I remember one frustrating evening trying to get the stacked area chart for allocations to display correctly because the DataFrame wasn't in the "long" format that `px.area` expected for the `color` mapping.

### Reflections and Key Learnings

This project was a significant undertaking, much more so than I initially anticipated.
*   **RL is Hard:** While libraries like Stable Baselines3 abstract away a lot of the algorithmic complexity, designing the environment, state/action spaces, and especially the reward function for a real-world problem like finance is non-trivial. What seems like a good reward on paper might lead to unintended agent behaviors.
*   **Data is King (and a Pain):** Ensuring data quality, proper normalization, and meaningful feature engineering are critical. Garbage in, garbage out definitely applies. My early struggles with Quandl data alignment and later with feature scaling for the RL agent underscored this.
*   **Iteration is Key:** None of the components worked perfectly on the first try. The reward function, the agent's hyperparameters, the features in the state representation – all required multiple iterations and a lot of patience. Seeing the TensorBoard graphs finally trend upwards after numerous tweaks was incredibly rewarding.
*   **Overfitting is a Constant Threat:** It's easy for an RL agent to learn a strategy that performs exceptionally well on the training data but fails miserably out-of-sample. Rigorous backtesting on unseen data, and being honest about the results (even when they weren't great), was crucial.
*   **Tooling Matters:** Using established libraries like Pandas, SQLAlchemy (implicitly via Pandas `to_sql`), Stable Baselines3, and Plotly Dash saved a huge amount of time and allowed me to focus on the problem rather than reinventing the wheel. I did spend time in their respective documentation and GitHub issue trackers when things went wrong.

While the agent developed isn't going to make me rich overnight (and I wouldn't trust it with real money without a *lot* more validation and work on robustness!), the learning experience was invaluable. I gained a much deeper appreciation for the complexities of both reinforcement learning and financial markets.

### Future Directions

If I were to continue this project, I'd explore a few areas:
*   **More Sophisticated State Representation:** Incorporating market sentiment, macroeconomic indicators, or news data.
*   **Different RL Algorithms:** Trying out SAC (Soft Actor-Critic) or other algorithms designed for continuous control tasks.
*   **Risk Management:** More explicit risk controls in the agent's objective or actions. For example, setting limits on drawdown or volatility.
*   **More Assets:** Expanding to a wider universe of stocks or even different asset classes like bonds or commodities. This would likely require more advanced data handling and potentially different RL architectures.
*   **Live Trading Integration (Carefully!):** This is a huge leap, but integrating with a broker API for paper trading would be the ultimate test.

Overall, this was a challenging but incredibly fulfilling project that combined my interests in coding, finance, and AI. It definitely pushed my Python skills and introduced me to the practical difficulties and nuances of applying RL to complex, noisy, real-world domains.