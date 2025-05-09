---
layout: post
title: Futures Momentum with RL
---

## Project Log: DQN for Hang Seng Futures Momentum

After a fairly intense couple of months, I've finally reached a point where I can document my latest project: applying Deep Q-Networks (DQN) to trade Hang Seng Index (HSI) futures based on momentum signals. This one was a real dive into the deep end of reinforcement learning and high-frequency data, and definitely pushed my limits.

### The Initial Idea and Why HSI Futures

The core idea was to see if an RL agent could learn a profitable strategy in a fast-moving market like HSI futures. I'd been reading a bit about how DQN was originally used for games, but its application in finance, particularly for direct policy learning without needing to predict prices explicitly, seemed fascinating. Momentum strategies are common, but I wondered if an agent could discover more nuanced patterns than a simple rule-based approach, especially with high-frequency data.

I chose HSI futures for a few reasons. Firstly, its volatility and liquidity make it an interesting candidate for algorithmic trading. Secondly, I had access to a (somewhat messy) dataset of 1-minute HSI futures data through a university research portal, which was a major deciding factor, as getting good quality high-frequency data is notoriously difficult and expensive.

### Setting Up the Environment: OpenAI Gym

The first major step was to build a trading environment. OpenAI Gym seemed like the standard, and I wanted to get familiar with its API. This was where the first set of challenges really began.

My `TradingEnv` needed to:
1.  **Represent the market state:** I decided to use a rolling window of past price changes (returns) and a couple of simple moving average differences as features. Initially, I just threw in raw prices, but that didn't work well because the network struggled with non-stationarity. Normalizing the returns was key.
2.  **Define actions:** Long, Short, or Flat (hold no position). So, three discrete actions.
3.  **Calculate rewards:** This was tricky. I started with simple profit/loss per step, but this can be very noisy. I eventually settled on a shaped reward that also penalized frequent trading due to commission and slippage, which I had to estimate.

Here's a snippet of my environment's `step` function. It's not perfect, and I wrestled with how to properly handle position changes and costs for a while.

```python
# Inside my TradingEnv class, derived from gym.Env
def step(self, action):
    # action: 0 for flat, 1 for long, 2 for short
    current_price = self.data.iloc[self.current_step]['close']
    next_price = self.data.iloc[self.current_step + 1]['close']

    reward = 0
    done = self.current_step >= len(self.data) - 2 # -2 to ensure we have a next_price for reward

    # Calculate P&L based on current position and price movement
    if self.current_position == 1: # Long
        reward = (next_price - current_price) / current_price
    elif self.current_position == -1: # Short
        reward = (current_price - next_price) / current_price

    # Apply transaction cost if action changes position
    if (action == 1 and self.current_position != 1) or \
       (action == 2 and self.current_position != -1) or \
       (action == 0 and self.current_position != 0):
        reward -= self.transaction_cost_rate # A flat rate per trade

    # Update position
    if action == 0:
        self.current_position = 0
    elif action == 1:
        self.current_position = 1
    elif action == 2:
        self.current_position = -1

    self.current_step += 1
    next_state = self._get_state() # Method to extract features for the next state

    # A small penalty for just holding, to encourage action if profitable
    if self.current_position == 0 and reward == 0: # only if no PnL from trade
         reward -= 0.00001


    # Update portfolio value (simplified for this snippet)
    # self.portfolio_value *= (1 + reward) # This was more complex in reality

    return next_state, reward, done, {}

def _get_state(self):
    # Extract features like past N returns, MAs, etc.
    # Normalization was super important here
    # Example:
    # price_window = self.data.iloc[self.current_step - self.window_size : self.current_step]['close']
    # returns = price_window.pct_change().dropna().values
    # if len(returns) < self.window_size -1: # Handle edge cases at the beginning
    #     returns = np.zeros(self.window_size -1)
    # return (returns - self.mean_returns_for_norm) / self.std_returns_for_norm # Example of normalization
    # This part took a lot of trial and error with feature scaling.
    
    # For the actual project, I used a window of log returns and some custom indicators
    # This is a simplified representation of getting the state
    if self.current_step < self.look_back_window:
        # Return a zero state or some padding if not enough data yet
        return np.zeros(self.observation_space.shape)

    start_idx = self.current_step - self.look_back_window
    end_idx = self.current_step

    # Extract relevant data points (e.g., 'price_diff', 'sma_diff')
    # These would be columns I pre-calculated in my dataset
    # For HFT, these features would be things like order book imbalance, micro-price movements, etc.
    # My 1-min data wasn't true HFT, so features were simpler.
    state_features = self.processed_data.iloc[start_idx:end_idx][['norm_ret', 'sma5_sma20_diff', 'volatility_proxy']].values.flatten()
    
    # Some basic normalization was applied during data preprocessing
    # For instance, `norm_ret` was (price_t - price_t-1) / price_t-1
    # `sma5_sma20_diff` was (sma5 - sma20) / price_t-1
    # `volatility_proxy` was std dev of returns over a short window

    return state_features
```
One specific issue I remember spending a whole weekend on was the `done` condition. If it triggered too early or incorrectly, the agent wouldn't learn long-term dependencies. Also, ensuring the state representation was normalized correctly took ages; without it, my network's gradients were all over the place. I recall looking at some Keras examples and a few blog posts about financial time series preprocessing which emphasized scaling features to a similar range (e.g., [-1, 1] or [0, 1]).

### The DQN Agent Implementation

I decided to implement the DQN from scratch using TensorFlow/Keras. While there are libraries like Stable Baselines3, I wanted to understand the nuts and bolts. This meant building:

1.  **The Q-Network:** A simple Multi-Layer Perceptron (MLP). I experimented with different architectures. Initially, I tried a very deep network, but it overfit quickly. A network with two hidden layers (e.g., 64 units then 32 units, with ReLU activation) seemed to offer a reasonable balance.
    ```python
    # Simplified Q-Network model
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense
    # from tensorflow.keras.optimizers import Adam

    # def build_model(input_shape, action_space_size):
    #     model = Sequential()
    #     model.add(Dense(64, input_shape=input_shape, activation='relu'))
    #     model.add(Dense(32, activation='relu'))
    #     model.add(Dense(action_space_size, activation='linear')) # Q-values are linear
    #     model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    #     return model
    ```
    I remember one frustrating period where my loss wasn't decreasing. It turned out my learning rate was way too high for the initial random weights, causing Q-values to explode. A StackOverflow thread mentioned trying much smaller learning rates (like 1e-4 or 1e-5) for financial data, which helped stabilize things.

2.  **Experience Replay:** Storing `(state, action, reward, next_state, done)` tuples in a `collections.deque`. This was relatively straightforward, but getting the sampling logic right for batch updates was crucial.

3.  **Target Network:** A separate network that's a delayed copy of the main Q-network, used to stabilize training. Updating its weights (the "soft update" or "hard update" choice) was another parameter to tune. I started with a hard update every N steps, as it was simpler to implement.

The training loop itself was complex. Epsilon-greedy exploration (decaying epsilon over time) was implemented to balance exploring new actions versus exploiting known good ones.

```python
# Snippet from my DQN agent's training logic
# self.model is the main Q-network
# self.target_model is the target Q-network
# self.replay_memory is a deque

# def replay(self, batch_size):
#     if len(self.replay_memory) < batch_size:
#         return # Not enough experiences to replay

#     minibatch = random.sample(self.replay_memory, batch_size)

#     states = np.array([experience.state for experience in minibatch])
#     next_states = np.array([experience.next_state for experience in minibatch])

#     # Predict Q-values for current states and next states
#     current_q_values = self.model.predict(states, verbose=0) # verbose=0 to keep logs clean
#     next_q_values_target = self.target_model.predict(next_states, verbose=0)

#     for i, experience in enumerate(minibatch):
#         state, action, reward, next_state, done = experience.state, experience.action, experience.reward, experience.next_state, experience.done
        
#         if not done:
#             # Bellman equation for Q-learning
#             target_q = reward + self.gamma * np.amax(next_q_values_target[i])
#         else:
#             target_q = reward

#         # Update the Q-value for the action taken
#         current_q_values[i][action] = target_q

#     # Train the main model
#     self.model.fit(states, current_q_values, epochs=1, verbose=0) # Fit on the updated Q-values

#     # Update exploration rate
#     if self.epsilon > self.epsilon_min:
#         self.epsilon *= self.epsilon_decay
```
One big hurdle was debugging the Q-value updates. For a while, the agent would just learn to take one action (e.g., always go long or always stay flat). Plotting the average Q-values per action during training helped me see if they were diverging or converging meaningfully. I found a helpful tip on a forum (I think it was Reddit's r/reinforcementlearning) about checking if the rewards were scaled appropriately – if they were too small, learning would be slow; too large, and it could become unstable.

### Data, Training, and Agony

The HSI futures data I had was 1-minute OHLCV. It was quite noisy and had some gaps, which required careful preprocessing (forward fill, or sometimes just dropping small segments). Feature engineering was iterative. I started with just lagged returns, then added some simple moving average crosses, and a proxy for volatility (like ATR or standard deviation of recent returns). I tried to keep the state space manageable.

Training took a *long* time on my personal laptop. Each episode was a run through a segment of the historical data (e.g., a few months). I'd let it run overnight and check the logs in the morning. Hyperparameter tuning was a significant time sink. Parameters like `learning_rate`, `gamma` (discount factor), `epsilon_decay`, `batch_size`, and `target_network_update_frequency` all interacted in complex ways. I mostly did a manual grid search, which was painful but I didn't have the setup for more sophisticated hyperparameter optimization tools at the time.

There was one week where the agent consistently lost money. The cumulative reward curve just went down and down. I was close to giving up, thinking DQN just wasn't suited for this. Then, I found a bug in my reward calculation where I was inadvertently penalizing profitable trades under certain position-closing conditions. It was a single line of code, off by one index, that cost me days of debugging. The relief when I fixed it and saw the reward curve finally start to trend upwards was immense.

### Simulation and Results (The Sobering Part)

After training (which never felt truly "done"), I ran the agent on a hold-out test set – a period of data it had never seen.
The results were… modest. It didn't make millions (unsurprisingly). It did manage to achieve a positive return over the test period, slightly outperforming a simple buy-and-hold strategy for that specific period, but its Sharpe ratio wasn't spectacular. The transaction costs, even small estimated ones, really ate into profits.

What was interesting was looking at the agent's behavior. It learned to be quite conservative, often staying flat during choppy periods and trying to catch short-term momentum bursts. It definitely wasn't a high-frequency scalper given the 1-minute data and my feature set.

I realized that with 1-minute data, you're missing a lot of the micro-price action that true high-frequency strategies might exploit. Also, my state representation was likely too simplistic to capture all the nuances of HSI futures momentum.

### Key Learnings and Future Directions

This project was a huge learning experience, more so in the process than in the final trading performance.
1.  **RL is Hard:** Debugging RL agents is non-trivial. The feedback loop is much longer and more indirect than in supervised learning.
2.  **Data is King (and Queen, and the Entire Royal Court):** The quality, frequency, and relevance of data are paramount. My 1-minute data had limitations. Access to tick data or even order book data would open up many more possibilities (and complexities).
3.  **Environment Design is Critical:** The state representation and reward function fundamentally shape what the agent can learn. Small changes here can have massive impacts. I spent at least 60% of my time on the environment and data preprocessing.
4.  **Hyperparameters are a Beast:** Finding the right set is an art and a science, and very time-consuming without robust optimization tools and more compute power.
5.  **Simplicity First:** I initially tried to make the state too complex. Simplifying it actually helped the agent learn better, at least to a certain point.

If I were to continue this, I'd explore:
*   **More sophisticated state features:** Perhaps from more advanced technical indicators, or even market sentiment if I could get that data.
*   **Different RL algorithms:** Maybe A2C or PPO, which are known for being a bit more stable than DQN in some contexts.
*   **Better handling of transaction costs and slippage:** My current model is a bit basic.
*   **Proper backtesting infrastructure:** Something more robust than my current scripts, perhaps using a library like `zipline` or `backtrader` (though integrating RL agents into those can be another project in itself).

Overall, while the agent isn't going to make me rich, the experience of building it from the ground up was incredibly valuable. It solidified a lot of theoretical concepts about RL and gave me a much greater appreciation for the practical challenges of applying it to financial markets.