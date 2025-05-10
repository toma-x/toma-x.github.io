---
layout: post
title: DRL for Asian Equity Arbitrage
---

## DRL for Asian Equity Arbitrage: A Deep Dive into Pairs Trading with PPO on CSI 300 Tick Data

After my last project on sentiment analysis, I wanted to explore a different facet of quantitative finance, specifically market microstructure and algorithmic execution. I landed on the idea of using Deep Reinforcement Learning (DRL) for statistical arbitrage, focusing on equity pairs trading. The allure of an agent learning optimal trading policies from high-frequency data was too strong to resist. My chosen battleground: the CSI 300 index, using tick data sourced from Wind. The core of the project was to develop a Proximal Policy Optimization (PPO) agent and see if it could learn to exploit short-term mispricings in identified stock pairs.

### Phase 1: Grappling with CSI 300 Tick Data

The first challenge was data. Wind is a common source for Chinese financial data, but accessing and processing its tick data wasn't trivial. The API, while powerful, had its quirks, and the sheer volume of tick-level data for multiple stocks over a significant period was enormous. My initial plan was to download data for a wide range of CSI 300 constituents and then run cointegration tests to find suitable pairs.

I used the `WindPy` library in Python. Getting historical tick data involved specifying contract codes, start/end times, and the fields I needed (last price, volume, bid/ask spreads).

```python
from WindPy import w
import pandas as pd

# w.start() # Authenticate with Wind terminal

# Example for fetching tick data for one stock - this was part of a larger loop
# sec_code = '600000.SH' # Example: Ping An Bank
# start_date = "2023-01-01 09:30:00"
# end_date = "2023-01-01 15:00:00"
# fields = "last,volume,bid,ask" 

# WindData = w.wst(sec_code, fields, start_date, end_date, "")
# if WindData.ErrorCode != 0:
# print(f"Error fetching data for {sec_code}: {WindData.Data}")
# else:
# tick_df = pd.DataFrame(WindData.Data, index=WindData.Fields, columns=WindData.Times).T
# tick_df.index = pd.to_datetime(tick_df.index)
# tick_df[['last', 'volume', 'bid', 'ask']] = tick_df[['last', 'volume', 'bid', 'ask']].apply(pd.to_numeric, errors='coerce')
# Process and save tick_df... e.g., tick_df.to_parquet(f'{sec_code}_ticks.parquet')
```
My laptop’s RAM quickly became a bottleneck when I tried to load and process data for multiple potential pairs simultaneously. I had to switch to processing stocks one by one, saving intermediate results to Parquet files, which are much more efficient for this kind of columnar data than CSVs.

For pair selection, I started with the classic Engle-Granger two-step method for cointegration but found its results quite sensitive to the testing period. I then explored the Johansen test, which can handle more than two variables and is generally considered more robust. I used `statsmodels.tsa.vector_ar.vecm.coint_johansen` for this. Identifying consistently cointegrated pairs from hundreds of stocks, especially with tick data that can be noisy, was an iterative process. I had to filter pairs based on the trace statistic and eigenvalue results from Johansen, and then visually inspect their spread behavior. Some pairs looked great on paper (statistically cointegrated) but had spreads that were too erratic or mean-reverted too infrequently to be practically tradable, especially after considering transaction costs.

Once pairs were shortlisted, the next hurdle was feature engineering from the tick data. Raw ticks are too granular. I decided to aggregate ticks into 1-minute bars (Open, High, Low, Close, Volume, VWAP) as a first step. From these bars, I calculated the price ratio and spread for each pair. For example, for a pair (A, B), the ratio `P_A / P_B` or the spread `log(P_A) - beta * log(P_B)` (where beta is the hedge ratio from cointegration) became central.

### Phase 2: Crafting the Trading Environment

With processed data, I built a custom trading environment compatible with the OpenAI Gym (now Gymnasium) interface. This was crucial for plugging in DRL agents from libraries like `stable-baselines3`.

**State Space:** What does the agent see? This was a major point of deliberation. I settled on:
1.  **Normalized Spread:** The current spread value, normalized using a rolling z-score (e.g., over the last 100 minutes) to give the agent a sense of how far the current spread is from its recent mean. My initial attempts using just the raw spread value made it hard for the agent to generalize across different volatility regimes.
2.  **Recent Spread Changes:** Differences in the normalized spread over the last few time steps (e.g., `spread_t - spread_{t-1}`, `spread_t - spread_{t-5}`). This was to give the agent some momentum information.
3.  **Current Position:** An integer representing the agent's current market exposure: +1 (long pair: long A, short B), -1 (short pair: short A, long B), or 0 (neutral).

**Action Space:** What can the agent do? I chose a discrete action space:
*   Action 0: Go neutral (close any open position).
*   Action 1: Enter a long position on the pair (if neutral or short).
*   Action 2: Enter a short position on the pair (if neutral or long).
I debated using a continuous action space (e.g., how much to trade), but that adds complexity, and for a first pass, discrete entry/exit signals felt more manageable. I also decided that the agent could only flip or close positions, not add to existing ones, to simplify state management.

**Reward Function:** This was, by far, the most challenging and iterative part.
My first attempt was simple: the change in portfolio value from one step to the next. This led to the agent learning to do nothing most of the time, or making very few, random trades.
`reward = portfolio_value_t - portfolio_value_{t-1}`

Then, I incorporated transaction costs. I assumed a fixed percentage cost for simplicity (e.g., 0.05% of the transaction value, which is probably optimistic for CSI 300 tick trading for a retail setup but was a starting point).
`reward = (portfolio_value_t - portfolio_value_{t-1}) - transaction_cost_if_trade_occurred`

This was better, but the agent was still too timid or sometimes churned positions too much. I found a few papers and forum discussions suggesting a shaping reward that penalizes holding positions for too long if they are not profitable, or provides small positive rewards for profitable exits. I experimented with adding a small penalty for each step a position was held, and a larger penalty if a position was held against a strongly mean-reverting spread. My final reward function was a bit of a heuristic mix, trying to balance profitability with encouraging mean-reversion trades and penalizing excessive trading or adverse positions. It looked something like this conceptually:
`reward = pnl_from_trade_exit - cost_of_entry - holding_penalty_if_no_exit`
The `holding_penalty` itself was scaled by how far the spread had moved against the position. This iterative refinement was painful; small changes to the reward structure could drastically alter the agent's learned behavior.

### Phase 3: Implementing PPO

I chose Proximal Policy Optimization (PPO) due to its reported stability and sample efficiency compared to other algorithms like DQN (which struggles with continuous action spaces, though I used discrete here) or A2C. I'd read that PPO strikes a good balance between performance and ease of tuning. I used the `stable-baselines3` library, which has a well-tested PPO implementation.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
# from my_custom_pairs_trading_env import PairsTradingEnv # This would be my custom Gym environment

# env_params = {'df': training_data_for_pair, 'initial_capital': 100000, 'transaction_cost_pct': 0.0005, ...}
# train_env = DummyVecEnv([lambda: PairsTradingEnv(**env_params)])

# Define network architecture if needed, though default MlpPolicy often works
# policy_kwargs = dict(net_arch=[dict(pi=, vf=)]) # Example custom network

# PPO model setup - these hyperparameters were subject to a LOT of tuning
# model = PPO("MlpPolicy", 
#             train_env, 
#             verbose=1, 
#             # policy_kwargs=policy_kwargs,
#             learning_rate=0.0001, # Often started higher, then reduced
#             n_steps=2048,         # How many steps to run for each environment per update
#             batch_size=64,        # Minibatch size
#             n_epochs=10,          # Number of epochs when optimizing the surrogate loss
#             gamma=0.99,           # Discount factor
#             gae_lambda=0.95,      # Factor for trade-off of bias vs variance for GAE
#             clip_range=0.2,       # PPO clipping parameter
#             ent_coef=0.01,        # Entropy coefficient for exploration
#             tensorboard_log="./ppo_pairs_trading_tensorboard/")


# Define an EvalCallback for periodic evaluation and saving the best model
# eval_env_params = {'df': validation_data_for_pair, ...} # Use separate validation data
# eval_env = DummyVecEnv([lambda: PairsTradingEnv(**eval_env_params)])
# eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
#                              log_path='./logs/results', eval_freq=5000, # Evaluate every 5000 steps
#                              deterministic=True, render=False)

# model.learn(total_timesteps=1000000, callback=eval_callback) # Train for 1 million timesteps
# model.save("ppo_pairs_trader_csi300")
```

Hyperparameter tuning for PPO was a journey in itself. The `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `clip_range`, and `ent_coef` all interact in complex ways. Default values from `stable-baselines3` were a starting point, but rarely optimal for financial time series. I spent a lot of time tweaking these. For instance, a too-high `learning_rate` often led to the policy collapsing or performance wildly oscillating. A too-low `ent_coef` (entropy coefficient) sometimes resulted in the agent converging prematurely to a suboptimal strategy without enough exploration. I didn't have the resources for a full-blown hyperparameter optimization suite like Optuna for every pair, so it was a lot of manual adjustments, running training sessions overnight, and checking TensorBoard logs in the morning. I remember one breakthrough when I drastically increased `n_steps` (the number of steps to run for each environment per update); this seemed to stabilize learning for some pairs, possibly by providing more diverse experiences before each policy update.

Convergence was also an issue. Sometimes the agent's performance would plateau, or even degrade, after an initial period of improvement. I used `EvalCallback` to save the best performing model based on evaluation on a separate validation set, and also to monitor for signs of overfitting. Normalizing input features (like the z-scored spread) was absolutely critical; without it, the neural network struggled to learn effectively.

### Phase 4: Backtesting and (Often Sobering) Results

Once a model was trained, I ran it on out-of-sample test data. This meant feeding the test data through the same preprocessing pipeline and letting the trained agent make decisions. I tracked metrics like cumulative return, Sharpe ratio, Sortino ratio, max drawdown, and the number of trades.

The results were mixed, as expected. For some selected pairs and during certain periods, the agent demonstrated an ability to generate positive returns, sometimes significantly better than a simple mean-reversion strategy based on fixed thresholds. However, for other pairs, or even the same pair in a different market regime (e.g., a sudden volatility spike), the agent struggled. The CSI 300 can be heavily influenced by policy changes and macroeconomic news, and it's unlikely my relatively simple state representation captured all necessary context.

One key observation was that agents often overfitted to the specific patterns in the training data, despite using a validation set. A strategy that looked great on the training and validation periods would sometimes fall apart on unseen future data. This is the classic challenge in financial ML. I tried to mitigate this by using fairly long training periods and being conservative with the evaluation metrics, but it's an ongoing battle. Transaction costs were also a killer; strategies that looked good pre-cost often became flat or negative once realistic slippage and fees were factored in.

I spent considerable time analyzing the agent's trades. Sometimes it learned sensible mean-reversion logic. Other times, its actions were baffling, likely latching onto some spurious correlation in the training data. Debugging the "why" behind a DRL agent's decision is notoriously difficult.

### Key Challenges and "Aha!" Moments

*   **Tick Data Volume:** The sheer amount of data from Wind was initially overwhelming. My "aha!" moment was switching to Parquet and processing in chunks, and realizing that for DRL, I didn't need *all* the ticks; well-constructed 1-minute bars were sufficient for the initial state representation.
*   **State Representation for the Agent:** My first few state designs were too naive (e.g., just raw price ratios). The agent couldn't learn anything. The breakthrough came from reading about how human traders look at normalized indicators. Switching to z-scored spreads and adding momentum features made a huge difference.
*   **Reward Function Engineering:** This was the most iterative part. Pure P&L was a disaster. The "aha!" moment was when I started to think about the *incentives* I was giving the agent. Penalizing prolonged unprofitable trades or rewarding quick, profitable mean reversions, even with small shaping rewards, guided the agent much better than raw P&L. I found a StackOverflow thread discussing reward shaping for trading agents that gave me some ideas for penalties.
*   **PPO Hyperparameters for Financial Data:** The defaults in `stable-baselines3` are tuned for Mujoco physics tasks, not noisy financial series. I almost abandoned PPO until I found a GitHub issue thread where someone discussed using larger `n_steps` and smaller learning rates for time-series forecasting with PPO. That adjustment unlocked better performance.
*   **Non-Stationarity:** Financial markets change. A cointegrated relationship might break down. An agent trained on one regime might fail in another. This was less of an "aha!" moment and more of a persistent, humbling realization. My models didn't explicitly account for regime changes, which is a major limitation.

### Preliminary Observations and What's Next

This project reinforced that DRL for trading is far from a solved problem, especially with high-frequency data where the signal-to-noise ratio is tiny. While the PPO agent did learn to trade some pairs profitably in backtests, the strategies were often brittle and highly sensitive to hyperparameters and the specific training period.

The main takeaway wasn't a "money-making machine," but a deep appreciation for the intricacies of environment design, reward shaping, and the challenges of applying DRL to noisy, non-stationary financial data.

Future directions are plentiful:
1.  **More Sophisticated State Features:** Incorporating order book imbalances (if I could get that data reliably), volatility measures, or even news sentiment (tying back to my previous project!) could enrich the agent's observations.
2.  **Handling Non-Stationarity:** Investigating techniques for detecting regime changes and adapting the agent, perhaps using meta-learning or ensembles of agents trained on different periods.
3.  **Advanced DRL Architectures:** Exploring agents with attention mechanisms to help them focus on relevant past information or using recurrent layers (LSTMs/GRUs) in the policy network might improve learning long-term dependencies.
4.  **Robust Hyperparameter Optimization:** Using tools like Optuna or Ray Tune more systematically for hyperparameter searches.
5.  **Execution Details:** Integrating more realistic transaction cost models, including slippage based on trade size and market depth.

It was an incredibly challenging but rewarding project. The process of defining the problem for the DRL agent, wrestling with data, and iteratively refining the model taught me more than any textbook could.