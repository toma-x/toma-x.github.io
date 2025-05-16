---
layout: post
title: Portfolio Optimization via RL
---

## Taming the Market? My Journey into Portfolio Optimization with Reinforcement Learning

This semester, I dived headfirst into a project that’s been on my mind for a while: applying Reinforcement Learning (RL) to the problem of dynamic asset allocation. The goal was to develop an agent that could learn to manage a portfolio, hopefully more effectively than simpler strategies, by interacting with market data. It was a challenging but incredibly rewarding experience, and I wanted to document the process, the hurdles, and some of the small victories.

### The Initial Spark and Setting the Stage

The idea of an AI learning to trade or invest isn't new, but I was particularly interested in framing it as an RL problem. Instead of just predicting prices, the agent would learn a *policy* for rebalancing assets based on market conditions. My toolkit for this endeavor was going to be Python, primarily for the RL agent itself, Quandl for historical market data (accessed via their SQL API), and Plotly Dash for visualizing the backtested performance.

### Wrestling with Data: Quandl and SQL

First things first: data. I needed reliable historical price data for a selection of assets. Quandl seemed like a good choice given its extensive financial datasets. I decided to focus on a few tech stocks (AAPL, MSFT, GOOGL) and an aggregate bond ETF (AGG) to have some diversification.

Getting the data into a usable format was the first mini-challenge. I used the `quandl` Python package initially, but for more complex queries and ensuring data alignment (handling missing data points, aligning dates, etc.), I ended up using their SQL API more directly with a local SQLite database where I'd store and preprocess the data.

A typical query to pull adjusted closing prices might look something like this, after I'd ingested the specific Quandl tables into my local DB:

```sql
SELECT date, ticker, adj_close
FROM daily_prices
WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL', 'AGG')
AND date >= '2015-01-01'
ORDER BY date, ticker;
```
This gave me a clean base to work from. The main preprocessing step involved calculating daily returns, which would be a key input for my RL agent. I also had to decide how to handle NaNs if a stock didn't have data for a particular day when others did – initially, I forward-filled, but then I realized this could introduce lookahead bias in some subtle ways if not careful, so I switched to ensuring my universe of stocks had complete overlapping data for the chosen period or using a pre-fetch buffer for the environment.

### Why Reinforcement Learning? And Choosing an Algorithm

I opted for RL because the problem of portfolio allocation feels inherently sequential. Decisions made today affect future possibilities. I wanted an agent that could learn from these delayed consequences.

After some research, I decided to try Proximal Policy Optimization (PPO). I'd read that PPO strikes a good balance between sample efficiency, ease of implementation, and stability, which sounded appealing compared to something like DQN that's more suited to discrete action spaces, or A2C which can sometimes be a bit finicky to tune. My action space – the allocation percentages for each asset – felt more continuous, although I initially discretized it to simplify things.

Defining the environment components was where things got really interesting (and a bit frustrating):

*   **State**: What information does the agent get to make decisions? I settled on a window of past `n` daily returns for each asset, plus the current portfolio weights. For `n=30`, and 4 assets, this meant a state vector of `30*4 + 4` elements.
*   **Action**: This was tricky. Initially, I tried having the agent output target weights directly. So, for 4 assets, an action would be `[w1, w2, w3, w4]` where `sum(wi) = 1`. This continuous action space was tough for PPO to handle well without more sophisticated network architectures or normalization. I spent a good week getting frustrated with actions that didn't sum to one or were outside the `[0,1]` range. I found a few forum posts discussing this, with some people suggesting a softmax activation on the output layer of the policy network. That helped.
*   **Reward**: This was the hardest part. A simple immediate reward like the daily portfolio return often led to myopic behavior. I experimented with the change in portfolio value, but also tried to incorporate a risk measure. Eventually, I settled on the portfolio's log return for each step, with a small penalty for transaction costs every time the agent rebalanced. I debated adding the Sharpe ratio as part of the reward, but calculating it incrementally and making it a good learning signal proved non-trivial. I saw a paper by Jiang et al. (2017) "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" which gave me some ideas on reward structures.

### Building the PPO Agent in Python

I used PyTorch to build the PPO agent. The core of PPO involves training two neural networks: an actor (the policy) and a critic (the value function).

Here's a simplified look at what my Actor network might have looked like. This is just the `forward` pass, the actual PPO update logic is much more involved with calculating advantages, surrogate loss, etc.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, action_dim) # For continuous actions, outputting mean
        # self.fc_std = nn.Linear(128, action_dim) # And standard deviation

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) # Tanh to keep means in a reasonable range, then scale
        # Forcing actions to sum to 1 and be positive (e.g. via softmax) happened after this
        # std = F.softplus(self.fc_std(x))
        # For simplicity, I often started with a fixed or decaying std.
        return mu # In a full PPO, you'd return a distribution
```

A major hurdle was hyperparameter tuning. Learning rates, the PPO clipping parameter `epsilon`, discount factor `gamma`, number of epochs per update... a small change could send the agent's learning off a cliff. I spent countless hours running experiments, often leaving them overnight, only to find the agent had learned to do nothing (i.e., hold all cash if that was an option) or YOLO into a single asset. One particular issue was the agent consistently allocating almost everything to one asset that had performed well in the early part of the training data, failing to adapt when its characteristics changed. This made me realize how important diverse training data and perhaps some regularization or entropy bonus in the loss function were.

I remember a breakthrough moment when I was debugging the advantage calculation. My rewards were sparse, and the agent wasn't learning. I found a StackOverflow thread (I wish I'd saved the link!) that discussed the importance of Generalized Advantage Estimation (GAE) for stabilizing learning in PPO, especially with noisy rewards. Implementing GAE correctly, ensuring the `done` flags were handled properly at the end of episodes, made a noticeable difference. My `done` flags in the environment were basically just the end of the backtest period.

### Backtesting the Strategy

Once the agent was trained (or at least, seemed to be learning *something*), I needed to backtest it on out-of-sample data. This is critical. It's easy to get fantastic results on the data you trained on.

My backtesting loop involved:
1.  Loading the unseen historical data.
2.  Initializing the portfolio (e.g., starting with all cash or an equal weight).
3.  At each time step (daily in my case):
    *   Get the current state.
    *   Pass the state to the trained agent to get an action (target weights).
    *   Calculate transaction costs based on the change in weights.
    *   Update portfolio value based on the returns of the chosen assets and the new weights.
    *   Record metrics like portfolio value, weights, and returns.

I had to be really careful about lookahead bias. For instance, ensuring that any data transformations (like moving averages if I were to use them in the state) only used past data. Or making sure the agent's decision at time `t` was only based on information available up to `t-1`.

Metrics I focused on were cumulative returns, Sharpe ratio, max drawdown, and Sortino ratio. Comparing these to a benchmark like "buy and hold an S&P 500 ETF" or an equal-weight portfolio was essential.

### Visualizing with Plotly Dash

To make sense of the backtest results and present them, I decided to build a simple web dashboard using Plotly Dash. I'm not a web developer, but Dash makes it surprisingly straightforward to build interactive UIs with Python.

I wanted to visualize:
*   Portfolio value over time compared to a benchmark.
*   Asset allocation weights changing over time.
*   Key performance metrics.

A snippet for a callback to update a graph might look like this (this is very simplified, my actual callbacks had more inputs/outputs for different scenarios):

```python
# app is the Dash app object
# df_portfolio_value would be a pandas DataFrame with 'Date' and 'Value' columns
# df_benchmark_value similar for the benchmark

# from dash import dcc, html, Input, Output
# import plotly.express as px

# app.layout = html.Div([
#     dcc.Graph(id='portfolio-performance-graph'),
#     # ... other components like dropdowns to select backtests ...
# ])

# @app.callback(
#     Output('portfolio-performance-graph', 'figure'),
#     [Input('some-trigger- مثلا-a-button-or-dropdown', 'value')]
# )
# def update_performance_graph(trigger_value):
#     # In a real scenario, trigger_value would determine which backtest data to load
#     # For now, let's assume df_portfolio_value and df_benchmark_value are pre-loaded
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df_portfolio_value['Date'], y=df_portfolio_value['Value'],
#                              mode='lines', name='RL Agent'))
#     fig.add_trace(go.Scatter(x=df_benchmark_value['Date'], y=df_benchmark_value['Value'],
#                              mode='lines', name='Benchmark'))
#     fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Portfolio Value')
#     return fig
```
The main challenge with Dash was managing state and making the dashboard responsive, especially if I wanted to re-run parts of the backtest or load different agent models on the fly. For this student project, I mostly pre-ran the backtests and had Dash load the resulting CSVs/pickles, which simplified things considerably. Debugging callbacks could also be a bit tricky sometimes, with cryptic frontend errors if the data wasn't in the exact expected format.

### Results, More Challenges, and Some "Aha!" Moments

So, did I beat Wall Street? Not quite. The performance was... mixed. On some out-of-sample periods, the agent showed promise, achieving better risk-adjusted returns than the benchmark. On others, it underperformed or made some questionable decisions.

One persistent challenge was the non-stationarity of financial markets. An agent trained on data from 2015-2018 might not fare so well in the market conditions of 2020 or 2022 without retraining or an adaptive mechanism. This is a huge area for future work.

A significant "aha!" moment was when I properly implemented transaction costs. Initially, I ignored them, and the agent would rebalance frantically. Adding even a small per-transaction percentage cost forced the agent to learn smoother, less frequent allocation changes, which felt much more realistic and also often improved net performance.

Another difficult part was interpreting *why* the agent made certain decisions. Neural networks are often black boxes. While I could see the allocations changing, understanding the exact market cues it was reacting to was not straightforward. Some work on attention mechanisms or saliency maps could be interesting here.

### Future Directions and Reflections

This project was a fantastic learning experience. It pushed my Python skills, introduced me to the practicalities of RL, and gave me a much deeper appreciation for the complexities of financial markets.

If I were to continue, I'd explore:
*   More sophisticated state representations: Incorporating things like volatility measures (e.g., GARCH models), market sentiment, or macroeconomic indicators.
*   Hierarchical RL: Perhaps one agent decides on market regime, and another decides on allocation within that regime.
*   Different RL algorithms: Maybe something like Soft Actor-Critic (SAC) for continuous control, which is known for its sample efficiency.
*   More robust backtesting: Using walk-forward optimization and being even more rigorous about data snooping.
*   Ensemble methods: Training multiple agents and combining their decisions.

Overall, while the agent isn't ready to manage my (non-existent) millions, the process of building it taught me an immense amount. The sheer number of small details that can go wrong, from data preprocessing to reward shaping to hyperparameter tuning, was eye-opening. But seeing it finally learn *something* coherent after weeks of effort was incredibly satisfying. It’s definitely a field I want to explore further.