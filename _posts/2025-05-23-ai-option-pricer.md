---
layout: post
title: AI Option Pricing Agent
---

## An Attempt at Dynamic Option Pricing with Reinforcement Learning

This semester, I decided to dive deep into a project that combined my interest in finance with the ever-evolving field of machine learning: building an AI agent for dynamic option pricing. The core idea was to train a Reinforcement Learning (RL) agent using PyTorch to make pricing decisions, leveraging historical options market data processed with Polars for feature engineering. It's been a journey with a steep learning curve, plenty of head-scratching moments, and a few satisfying breakthroughs.

### The Challenge: Pricing in a Moving Market

Option pricing isn't new; Black-Scholes and other models provide theoretical values. However, real markets are dynamic. Prices fluctuate based on supply, demand, and new information, often deviating from these theoretical values. The goal wasn't to reinvent Black-Scholes, but to see if an RL agent could learn to adjust pricing strategies in response to market conditions, perhaps capturing nuances that static models miss. I wanted it to learn a *policy* for pricing, not just a single formula.

### Laying the Groundwork: Data Wrangling with Polars

The first hurdle was data. I managed to get my hands on a sizeable dataset of historical options trades and quotes – tick-level data for a few underlying assets over a couple of years. This was gigabytes of Parquet files, and my previous experiences with Pandas on datasets of this scale were... painful. I’d read about Polars and its performance claims, especially with its Rust backend and lazy evaluation, so this project felt like the right time to try it.

The initial task was to extract meaningful features. This involved:
1.  Calculating implied volatilities (IVs) for each option.
2.  Getting various Greeks (Delta, Gamma, Vega, Theta).
3.  Engineering features that might give the agent context: time to expiration, moneyness (strike price vs. underlying price), recent price movements of the underlying, and some measures of market activity like trading volume and bid-ask spreads for the options themselves.

Polars' expression API was a bit different from what I was used to with Pandas, but once I got the hang of it, the speed was noticeable. For instance, calculating rolling averages or standard deviations across large windows was significantly faster.

Here’s a snippet of how I started creating some lagged features and rolling calculations. I was trying to capture recent volatility and price trends.

```python
import polars as pl

# Assuming 'df_options_data' is a Polars DataFrame loaded from Parquet
# It has columns like 'timestamp', 'underlying_price', 'option_price', 'strike', 'tte', 'volume'

df_featured = df_options_data.sort("timestamp").group_by("option_id").agg(
    [
        pl.col("underlying_price").pct_change().alias("underlying_ret"),
        pl.col("underlying_price").rolling_mean(window_size="1h", by="timestamp").alias("ul_ma_1h"),
        pl.col("underlying_price").rolling_std(window_size="1h", by="timestamp").alias("ul_std_1h"),
        pl.col("option_price").rolling_mean(window_size="30m", by="timestamp").alias("opt_ma_30m"),
        # Calculating time differences for recency
        pl.col("timestamp").diff().cast(pl.Duration(time_unit="ms")).alias("time_since_last_trade"),
        (pl.col("bid_price") - pl.col("ask_price")).abs().alias("spread")
    ]
).drop_nulls()

# Later, I wanted to add features based on the option's own historical prices
# This was a bit tricky because I needed to do it per option_id
# and ensure I wasn't looking ahead. The group_by().apply() was slower than I hoped initially.
# Eventually, using window functions within group_by expressions was the way.

df_final = df_featured.with_columns([
    (pl.col("underlying_ret").shift(1).over("option_id")).alias("underlying_ret_lag1"),
    (pl.col("ul_std_1h").ewm_mean(alpha=0.3).over("option_id")).alias("ul_std_1h_ema")
])

# print(df_final.head())
```

One specific challenge was handling the sheer number of unique option contracts. Each has its own lifecycle. Applying functions group-wise per `option_id` initially led to some slow queries until I leaned more into Polars' powerful window functions and expression optimization. I remember spending a good afternoon on the Polars documentation and a few StackOverflow threads trying to optimize a particularly complex rolling feature calculation across groups. The key was to avoid `apply()` where possible and stick to the declarative expressions.

### The Agent: A Deep Q-Network in PyTorch

For the RL agent, I chose PyTorch. I'm more comfortable with its Pythonic style compared to TensorFlow, and the ecosystem for RL research in PyTorch feels quite vibrant. I decided to start with a Deep Q-Network (DQN), as it's a foundational algorithm in deep RL and seemed like a reasonable starting point for a value-based approach to pricing. The idea is that the agent learns a Q-value for each possible pricing action in a given state.

The neural network itself wasn't overly complex: a few fully connected layers.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128) # Added an extra layer here after initial tests showed slow learning
        self.layer3 = nn.Linear(128, n_actions)
        # I considered batch norm but decided against it initially to keep it simpler
        # Might revisit if training becomes unstable with more complex states

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# n_observations would be the number of features from Polars
# n_actions would be the discrete price adjustments the agent can make
```

**State, Action, and Reward – The Tricky Bits:**

*   **State:** The input to the DQN. This was a vector of the features I engineered with Polars: normalized current underlying price, time to expiration, implied volatility, historical volatility of the underlying, recent option trading volume, current bid-ask spread of the option, and a few lagged features. Normalizing these was crucial; initially, I forgot to scale some features, and the network had a hard time learning.
*   **Action:** This was one of the harder design choices. Continuous price output is complex for basic DQN. So, I discretized the action space. The agent could choose to:
    1.  Quote a price slightly below the current market bid (aggressive buy).
    2.  Quote at the current market bid.
    3.  Quote slightly above the current market ask (aggressive sell).
    4.  Quote at the current market ask.
    5.  Quote at the mid-price.
    6.  Quote at a "theoretical" price (e.g., Black-Scholes, which was also a feature in the state) +/- a small spread.
    I ended up with about 10 discrete actions representing deviations from some baseline price.
*   **Reward:** This was iterated upon *many* times. My first attempt was simple: +1 for a profitable trade, -1 for a losing trade. This was too sparse and didn't guide the agent well. I then moved to using the actual P&L of a simulated trade. If the agent proposed a price and the trade hypothetically executed, what was the immediate P&L? This was better, but I also wanted to penalize it for not trading or for quoting prices that were way off.
    My eventual reward function was something like:
    `reward = (simulated_pnl_of_action - transaction_cost_if_traded) - (inaction_penalty_if_no_trade_possible) - (spread_penalty_if_quoted_spread_too_wide)`
    The `inaction_penalty` was small, just to encourage participation. The `spread_penalty` was to discourage it from just quoting super wide and never trading.

**The Training Loop and Environment:**

I didn't use a standard Gym environment because the setup was so specific. I built a custom environment that would feed market states (rows from my Polars DataFrame) to the agent. When the agent took an action (proposed a price), the environment would simulate if that trade would have executed based on the *next* tick's market prices and calculate the reward.

The training loop involved an experience replay buffer (`ReplayMemory`) and a target network, standard components for DQN to stabilize learning.

```python
# Simplified snippet from my training loop
# Assume policy_net, target_net, optimizer, memory are initialized

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions)) # Transition is a named tuple

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1})
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad(): # Important: no_grad for target network
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    # Expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss() # Huber loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping - this helped a lot with exploding gradients initially
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) 
    optimizer.step()
```
One of the first major issues was the agent collapsing to always outputting the same action, or the Q-values exploding. Gradient clipping ( `torch.nn.utils.clip_grad_value_`) was a lifesaver here. I also spent a lot of time tuning the learning rate for the Adam optimizer and the `GAMMA` (discount factor). A `GAMMA` too low made it myopic; too high, and it struggled to converge with distant, noisy rewards.

I also found that updating the target network too frequently made training unstable. A slower update (`TAU` parameter for soft updates, or updating every N steps for hard updates) worked much better. I stumbled upon a PyTorch forum discussion about DQN stability which reinforced this.

### Debugging Nightmares and Small Victories

RL is notoriously tricky to debug. There were weeks where the agent just didn't seem to learn anything sensible.
*   **Feature Scaling:** As mentioned, unscaled features were a big problem. The network effectively ignored smaller-magnitude features. Standardizing them (subtract mean, divide by std dev) made a huge difference.
*   **Reward Shaping:** If the reward function accidentally incentivized weird behavior, the agent would find it. At one point, it learned to quote extremely wide spreads to avoid any risk, getting a tiny negative penalty but never learning to actually price competitively. I had to re-think the penalties and rewards to encourage more active, reasonable participation.
*   **Hyperparameter Tuning:** This was more art than science. Epsilon for the epsilon-greedy exploration strategy, learning rate, batch size, replay buffer size... I did a lot of short runs, logged everything to TensorBoard, and tried to build intuition. It felt like walking a tightrope.
*   **Polars to PyTorch Data Flow:** Ensuring the data pipeline from Polars DataFrames into PyTorch Tensors was efficient and correct, especially with batching for the neural network, took some careful coding. No major bugs here, but it required attention to detail with data types and tensor shapes.

A breakthrough moment was when I started seeing the agent's Q-values for "good" actions (like pricing near the future executed price) consistently rise above those for "bad" actions in the TensorBoard logs. Another was when, after a long training run, I backtested it on a held-out dataset, and it actually showed a small, hypothetical profit over just quoting the mid or a static model. It wasn't going to make me rich, but it *learned something*.

### Evaluation and What's Next

Evaluating the agent was done by running it on a test set of historical data it hadn't seen during training. I tracked its hypothetical P&L, how often its quotes would have been "hit" or "lifted" (i.e., a trade occurred at its price), and compared its pricing decisions against the theoretical Black-Scholes price and the actual market prices.

The results were modest. The agent did learn to adjust its quotes based on market volatility and time to expiry in a way that seemed somewhat intelligent. For very liquid options, it essentially learned to quote very close to the market mid, which makes sense. For less liquid ones, it seemed to be more conservative. It didn't dramatically outperform a simple baseline strategy consistently, but there were pockets where it showed promise, especially in reacting to short-term imbalances.

If I were to continue, I'd explore:
1.  **More Sophisticated RL Algorithms:** PPO (Proximal Policy Optimization) or A2C (Advantage Actor-Critic) could potentially offer more stability and handle more complex action spaces.
2.  **Continuous Action Space:** Instead of discrete price adjustments, directly outputting a price (or a bid-ask spread). This would likely require actor-critic methods.
3.  **Better State Representation:** Incorporating order book data, if available, or more sophisticated volatility measures (e.g., GARCH models as input features).
4.  **Longer Training & More Data:** RL agents are data-hungry. More diverse market conditions and longer training periods would be beneficial.
5.  **Ensemble Methods:** Perhaps combining the RL agent's suggestions with traditional model outputs.

This project was an incredible learning experience. The combination of data engineering with Polars and the complexities of RL with PyTorch for a finance application was challenging but very rewarding. It really highlighted the difference between understanding the theory of these tools and models, and actually making them work on a messy, real-world-ish problem.