---
layout: post
title: Portfolio Hedging Reinforcement Learner
---

## Taming Market Volatility: My Journey Building a Portfolio Hedging RL Agent

This semester has been a deep dive into the practical applications of machine learning, and I wanted to tackle something that combined my interest in finance with the more advanced AI concepts we’ve been studying. The idea of portfolio hedging, specifically using options, seemed like a challenging yet rewarding problem to explore. Traditional methods like delta hedging are well-established, but I was curious if a Reinforcement Learning agent could discover more nuanced strategies, especially in volatile market conditions. That curiosity led to the "Portfolio Hedging Reinforcement Learner" project.

**The Core Challenge: Optimal Hedging with Options**

The goal was to develop an RL agent that could learn to optimally hedge a portfolio (let's say, a single stock position for simplicity to start with) using options. This means making decisions on when to buy or sell options, and in what quantities, to minimize risk or stabilize the portfolio's value over time. The financial markets are notoriously noisy and dynamic, making this a non-trivial task. My hypothesis was that an RL agent, by interacting with a simulated market environment based on historical data, could learn patterns and strategies that are not immediately obvious.

**Laying the Groundwork: Data Wrangling with Polars**

Before any fancy RL algorithms could be brought to bear, there was the inescapable reality of data. I needed good quality historical data for both the underlying asset (e.g., a specific stock) and its corresponding options contracts. This involved:
1.  Sourcing daily stock price data (Open, High, Low, Close, Volume).
2.  Sourcing options data (strike prices, bid/ask, volume, open interest, expiration dates, implied volatility for different moneyness levels).

This is where I decided to use **Polars**. I'd been reading about its performance benefits over Pandas for larger datasets, especially its lazy evaluation and Rust backend. Given that financial time series can get quite large, and I anticipated a lot of transformations and feature engineering, Polars seemed like a good investment in terms of processing speed.

```python
# Example of how I started loading and cleaning options data with Polars
import polars as pl

# Let's say I had options_data_raw.csv and stock_data_raw.csv
# This is a simplified example; the actual data sourcing was more fragmented
try:
    df_options_raw = pl.read_csv("options_data_raw.csv")
    df_stock_raw = pl.read_csv("stock_data_raw.csv")

    # Convert date columns - this was a frequent source of early errors!
    # Had to check formats from different sources and unify them.
    df_options = df_options_raw.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d"),
        pl.col("expiration_date").str.to_date("%Y-%m-%d")
    )
    df_stock = df_stock_raw.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )

    # Merging stock data with options data based on the observation date
    # This was crucial for creating features later on.
    df_combined = df_options.join(df_stock, on="date", suffix="_stock")

    # Feature: Days to Expiration (DTE)
    # This one took a bit to get right with Polars' expression API
    # as I was more used to Pandas' apply.
    df_combined = df_combined.with_columns(
        (pl.col("expiration_date") - pl.col("date")).dt.total_days().alias("dte")
    )

    # Filtering out very short DTE options or those with no volume,
    # as these can behave erratically or have poor liquidity.
    df_analysis_ready = df_combined.filter(
        (pl.col("dte") > 5) & (pl.col("volume") > 0) & (pl.col("open_interest") > 0)
    )

    print(f"Processed data shape: {df_analysis_ready.shape}")

except FileNotFoundError:
    print("Error: Ensure options_data_raw.csv and stock_data_raw.csv are present.")
except Exception as e:
    print(f"An error occurred during data processing: {e}")

```
One of the first hurdles was data cleaning. Options data can be messy – missing values, inconsistent formatting for dates across different (hypothetical) providers I might have pieced together, and ensuring proper alignment of option prices with the underlying asset's price at that exact time. Polars' expression API was powerful but had a steeper learning curve initially compared to Pandas. I remember spending a good afternoon figuring out the equivalent of a common Pandas `apply` function for calculating 'days to expiration' efficiently, eventually landing on Polars' more idiomatic date arithmetic. Forums and the Polars documentation were my best friends during this phase.

**Choosing the RL Framework: Python and Ray RLlib**

With the data pipeline starting to take shape, the next big decision was the RL framework.
*   **Python** was a no-brainer. It's the lingua franca for most ML work, and my own familiarity with it made it the practical choice.
*   For the RL library, I considered a few options. Building everything from scratch was tempting for the learning experience but unrealistic given the project's scope and my timeframe. I looked into Stable Baselines3 and TF-Agents, but ultimately settled on **Ray RLlib**.
    *   RLlib's main appeal was its comprehensive set of implemented algorithms and its focus on scalability (even though I'd mostly be running things on my laptop).
    *   The documentation seemed quite thorough, and I found a few academic papers and blog posts where it was used for financial applications, which gave me some confidence.
    *   I specifically recalled seeing some positive mentions of RLlib on a few reinforcement learning subreddits regarding its flexibility for custom environments.

**Crafting the Environment: The Heart of the RL Problem**

This was, without a doubt, the most challenging and iterative part of the project. An RL environment needs a well-defined state space, action space, and reward function.

*   **State Space:** What information does the agent need to make a decision? I started with a relatively simple state:
    1.  Current portfolio value.
    2.  Current holdings of the underlying asset.
    3.  Current holdings of the option (e.g., number of contracts, strike, DTE).
    4.  Current price of the underlying asset.
    5.  Time to expiration (DTE) of the currently held/considered option.
    6.  Maybe some measure of recent volatility.

    I quickly realized this might be too simplistic. For instance, just the "current option" wasn't enough if the agent was to choose *which* option to trade. I iterated a bit, thinking about how to represent available options. For a start, I decided to limit the agent to trading a single, specific type of option (e.g., an at-the-money call with roughly 30 DTE) to simplify the action space, and then the state would just reflect properties of that *type* of option. Later, I considered expanding this, but decided to keep it manageable for a first pass.

*   **Action Space:** What can the agent do? I decided on a discrete action space initially:
    1.  Hold (do nothing).
    2.  Buy X contracts of the chosen option type.
    3.  Sell X contracts of the chosen option type (if holding).
    4.  Buy Y shares of the underlying.
    5.  Sell Y shares of the underlying (if holding).

    Defining 'X' and 'Y' was tricky. Fixed amounts? Percentages of portfolio? I started with fixed small amounts to avoid catastrophic single actions. This definitely felt like a compromise, as a more sophisticated agent might want more granular control.

*   **Reward Function:** This is where I spent a *lot* of time. How do you tell the agent it's doing a "good job" hedging?
    *   My first attempt was simply the change in portfolio value (PnL) at each step. This led to the agent just trying to maximize profit, not necessarily hedge. It would take on risky positions if it thought there was a chance of a big payoff.
    *   Then I tried to penalize volatility. I used the standard deviation of the portfolio's daily returns over a short window as a negative reward component. This started to look more promising.
    *   I also incorporated a penalty for transaction costs. Every trade (buy/sell option or stock) incurred a small negative reward. This was important to make the agent avoid excessive, jittery trading. I found a post on StackOverflow (though I can't find the exact link now) discussing how to implement transaction costs in custom Gym environments, which gave me some ideas.

    A simplified version of my reward logic looked something like this (conceptually, the actual implementation within the RLlib environment class was more involved):

    `reward = daily_pnl - (lambda_vol * portfolio_return_volatility) - (lambda_tc * transaction_costs_this_step)`

    Tuning `lambda_vol` and `lambda_tc` was an empirical process. Too high a penalty for volatility, and the agent would do nothing. Too low, and it wouldn't hedge effectively.

**The Agent: Proximal Policy Optimization (PPO)**

RLlib offers many algorithms. I chose PPO (Proximal Policy Optimization) for a few reasons:
*   It's known for being relatively stable and sample-efficient compared to older algorithms like vanilla Policy Gradients.
*   It works well with both continuous and discrete action spaces (though I was using discrete).
*   Many resources and papers point to PPO as a good general-purpose algorithm and a solid starting point. I remember some RL course materials I reviewed also highlighted PPO as a go-to algorithm.

I mostly started with the default PPO configurations in RLlib and then tweaked things like the learning rate and the size of the neural network (a simple MLP). I didn't have the computational resources for extensive hyperparameter sweeps, so it was more about educated guesses and observing training progress.

**Implementation: Weaving It All Together (and the Inevitable Bugs)**

My custom environment was a Python class inheriting from `gym.Env`. The `step` method was where the core logic resided: take an action, update the portfolio, calculate the reward, and return the new state. The `reset` method would initialize a new episode, typically by setting the portfolio to an initial state and picking a new starting point in my historical data.

```python
# A very simplified snippet of what the environment's step function might look like
# (This is conceptual and heavily abridged)
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OptionsHedgingEnv(gym.Env):
    def __init__(self, historical_data, initial_cash, trading_lot_size_option=1, trading_lot_size_stock=10):
        super().__init__()
        self.historical_data = historical_data # This would be my Polars DataFrame
        self.current_step = 0
        self.initial_cash = initial_cash
        self.trading_lot_size_option = trading_lot_size_option
        self.trading_lot_size_stock = trading_lot_size_stock

        # Define action space: 0:Hold, 1:BuyOpt, 2:SellOpt, 3:BuyStock, 4:SellStock
        self.action_space = spaces.Discrete(5)

        # Define observation space (example: cash, stock_pos, option_pos, stock_price, option_price, dte)
        # The actual space was larger and normalized.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._reset_portfolio() # Initialize portfolio variables

    def _reset_portfolio(self):
        self.cash = self.initial_cash
        self.stock_position = 0
        self.option_position = 0 # Number of contracts
        self.option_strike = 0 # Store current option details if any
        self.option_dte = 0
        self.portfolio_value_history = [self.initial_cash]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, len(self.historical_data) - 200) # Start at a random point in data for variability, leave room for episode
        self._reset_portfolio()
        # Initialize state from historical_data at self.current_step
        obs = self._get_observation()
        info = {} # RLlib expects an info dict
        return obs, info

    def _get_observation(self):
        # Construct the observation array from current market and portfolio state
        # This involves querying self.historical_data at self.current_step
        # and self.portfolio (cash, positions, etc.)
        # This part needs careful normalization for the neural network.
        # For brevity, returning a placeholder.
        # Example:
        # stock_price = self.historical_data["stock_close"][self.current_step]
        # option_price = self.historical_data["option_close_approx"][self.current_step] # Needs careful selection
        # dte = self.historical_data["dte"][self.current_step]
        # obs = np.array([self.cash, self.stock_position, self.option_position, stock_price, option_price, dte])
        return np.random.rand(6).astype(np.float32) # Placeholder

    def step(self, action):
        # Store previous portfolio value for PnL calculation
        prev_portfolio_value = self.cash + (self.stock_position * self._get_current_stock_price()) \
                               + (self.option_position * self._get_current_option_price()) # Simplified

        transaction_cost_this_step = 0
        # Execute action (update self.cash, self.stock_position, self.option_position)
        # This is where I'd implement the logic for buying/selling based on 'action'
        # and decrement cash by transaction_cost_this_step
        # For example:
        if action == 1: # Buy Option
            # cost = self._get_current_option_price() * self.trading_lot_size_option
            # if self.cash >= cost:
            #    self.option_position += self.trading_lot_size_option
            #    self.cash -= cost
            #    transaction_cost_this_step += some_fixed_or_percentage_cost
            pass # Actual logic is more complex

        self.current_step += 1
        terminated = self.current_step >= (len(self.historical_data) -1) or self.option_dte <= 1 # Or if an option expires

        # Calculate current portfolio value
        current_portfolio_value = self.cash + (self.stock_position * self._get_current_stock_price()) \
                                  + (self.option_position * self._get_current_option_price()) # Simplified

        daily_pnl = current_portfolio_value - prev_portfolio_value
        self.portfolio_value_history.append(current_portfolio_value)

        # Calculate reward
        # A more robust reward would look at volatility of PnL etc.
        reward = daily_pnl - transaction_cost_this_step
        # if len(self.portfolio_value_history) > 20:
        #     pnl_series = pl.Series(self.portfolio_value_history[-20:])
        #     reward -= 0.1 * pnl_series.std() # Penalize volatility

        obs = self._get_observation()
        info = {}
        return obs, reward, terminated, False, info # False for truncated

    def _get_current_stock_price(self):
        # Fetch from self.historical_data at self.current_step
        return self.historical_data["stock_close"][self.current_step] # Example column name

    def _get_current_option_price(self):
        # Fetch from self.historical_data at self.current_step
        # This needs to be for the specific option the agent might be holding/trading
        return self.historical_data["option_close_approx"][self.current_step] # Example

```

Debugging the environment was a pain. There was one instance where my reward signal was consistently near zero, and the agent wasn't learning anything. It took me ages to figure out I had a bug in how I was calculating portfolio value changes – an off-by-one error in indexing my historical price series meant the PnL was almost always flat. Stepping through the `step` function line-by-line with a debugger became a common ritual.

Training was also a test of patience. Even with RLlib, training an agent for hundreds of thousands of timesteps on my laptop meant letting it run overnight and hoping for the best. I started with shorter episodes and smaller slices of data to iterate faster on the environment design before committing to longer training runs.

**Backtesting: The Moment of Truth**

After training an agent, the crucial step was backtesting it on unseen historical data. This meant feeding the trained policy new market scenarios and seeing how it decided to hedge, then evaluating the performance.
Metrics I focused on:
*   Overall P&L of the hedged portfolio.
*   Volatility (standard deviation) of the portfolio's daily returns – lower is better for a hedging strategy.
*   Comparison to a benchmark, like a simple delta-hedging strategy or even a buy-and-hold strategy for the underlying.

My initial results were... underwhelming. The agent often made strange decisions or failed to hedge effectively. This usually sent me back to the drawing board, primarily to rethink the reward function or the state representation. For example, I realized that not having a good proxy for implied volatility in the state was a major blind spot, so I added that.

One "aha!" moment came when I adjusted the reward function to more heavily penalize sharp drawdowns in portfolio value, rather than just overall volatility. This made the agent more conservative and more focused on preventing large losses, which is a key aspect of hedging.

**Key Challenges and "Gotchas"**

*   **Market Non-stationarity:** Financial markets change. An agent trained on data from 2015-2018 might not perform well in the market conditions of 2020. This is a fundamental challenge for any ML model in finance, and RL is no exception. My agent is essentially curve-fitted to the historical data it saw during training.
*   **Transaction Costs:** Accurately modeling these is hard. Bid-ask spreads, commissions, market impact for larger trades... I included a simple fixed cost per trade, but it's an approximation.
*   **Defining "Optimal":** "Optimal hedging" can mean different things. Minimum variance? Maximum Sharpe ratio for the hedged portfolio? Path-dependent goals? The reward function is your way of defining this, and it's more art than science sometimes.
*   **Curse of Dimensionality:** If I wanted the agent to choose from *any* available option (any strike, any DTE), the action space would explode, making it much harder for the agent to learn. This is why I simplified it.

**Reflections and What's Next**

This project was a massive learning experience. Wrestling with data, designing the RL environment, debugging the interactions between the agent and the market simulation, and then trying to interpret the results – it was all incredibly challenging but also very satisfying.

If I were to continue, I'd explore:
*   More sophisticated state features: perhaps incorporating market sentiment indicators or macroeconomic data.
*   A more nuanced action space, possibly using parameterised actions to allow the agent to choose option characteristics.
*   Different RL algorithms, maybe something from the model-based RL family if I could build a decent market model.
*   More rigorous backtesting, including sensitivity analysis to different market regimes and transaction cost assumptions.

While the agent isn't ready to manage real money (far from it!), the process of building it has given me a much deeper appreciation for both the complexities of financial markets and the potential (and limitations) of reinforcement learning. It definitely solidified my understanding of RL concepts far more than just reading about them would have. And, getting Polars to effectively chew through the data felt like a small victory in itself!