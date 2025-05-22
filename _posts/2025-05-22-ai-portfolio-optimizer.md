---
layout: post
title: AI-Driven Portfolio Risk Optimizer
---

## AI-Driven Portfolio Risk Optimizer: A Deep Reinforcement Learning Journey

This past semester, I embarked on a project that felt like a natural intersection of my coursework in machine learning and a growing interest in financial markets. The goal was to develop an "AI-Driven Portfolio Risk Optimizer." Essentially, I wanted to see if I could build a Python tool using deep reinforcement learning – specifically PyTorch – to actively manage portfolio risk and, hopefully, improve its risk-adjusted returns. The target metric was the Sharpe ratio, and I was aiming to see a tangible improvement on simulated FX and equity market data.

**The Starting Point: Why Reinforcement Learning?**

The idea of an agent learning optimal trading or allocation strategies through trial and error in a simulated environment was really compelling. Traditional portfolio optimization methods, like Markowitz mean-variance optimization, often rely on historical estimates of returns and covariances, which can be unstable and not always predictive of future market conditions. Reinforcement Learning (RL) felt like it could offer a more adaptive approach. My initial thought was that an RL agent could potentially learn complex, non-linear relationships in market data that simpler models might miss.

**Choosing the Toolkit: Python and PyTorch**

Python was an obvious choice given its extensive libraries for data science and machine learning. When it came to the deep learning framework, I opted for PyTorch. I'd had some exposure to it in a previous course on neural networks and found its more "Pythonic" style and dynamic computation graphs more intuitive for debugging compared to TensorFlow, which I'd briefly tried on another small project. That prior familiarity, however limited, made the ramp-up a bit less daunting, though this project quickly pushed me beyond what I knew.

**The Data Hurdle: Simulated FX/Equity Markets**

One of the first major challenges was sourcing or creating appropriate market data. I needed a sufficiently long and complex dataset covering both foreign exchange (FX) and equities to train the RL agent. My first instinct was to try and generate this myself using stochastic processes like Geometric Brownian Motion for individual assets and then try to model correlations. I spent a good week trying to get realistic cross-asset correlations, particularly between, say, EUR/USD and an S&P 500 proxy, but it turned out to be much harder than I anticipated to get something that didn’t feel completely artificial.

Eventually, after a lot of searching, I stumbled upon a publicly available simulated dataset from a university research group. It wasn't perfect – it covered a limited set of major FX pairs (EURUSD, GBPUSD, USDJPY) and a few broad equity index trackers – but it was multi-asset and had enough history and complexity for my purposes. It still required a fair bit of preprocessing: normalization of price series (I ended up using percentage returns as inputs to the agent), handling any missing values, and splitting it into training, validation, and a final hold-out test set.

**The Brains of the Operation: Deep Reinforcement Learning with PPO**

This was the core of the project. I knew I wanted to use a model-free RL algorithm. My initial research led me to consider Deep Q-Networks (DQN), but those are primarily for discrete action spaces, and I envisioned my agent outputting continuous portfolio weights. This pointed me towards algorithms like Deep Deterministic Policy Gradient (DDPG). I spent a couple of weeks trying to implement DDPG, but I found it incredibly sensitive to hyperparameter settings. My agent either wouldn't learn at all or would converge to really strange, suboptimal strategies. I recall one particularly frustrating late-night session where the DDPG agent decided the best strategy was to allocate 100% to one asset and 0% to everything else, regardless of market conditions.

After venting my frustrations on a forum and doing more reading, I came across several recommendations for Proximal Policy Optimization (PPO), particularly for its relative stability and ease of tuning compared to DDPG. The OpenAI paper on PPO and several blog posts discussing its implementation in PyTorch were my guides here. It seemed to strike a good balance between sample efficiency and stability.

**Designing the Agent and Environment**

The RL framework requires an `environment` and an `agent`.

*   **The Environment:** I had to build a custom OpenAI Gym-like environment that wrapped my simulated market data. The `step` function would take an action from the agent (the desired portfolio weights), calculate the portfolio return for that step, update the portfolio value, and return the next state, reward, and a `done` flag. The `state` I fed to the agent included a window of past price returns for all assets, plus the current portfolio weights. Initially, I just used raw price returns, but I found that adding a couple of simple technical indicators like a 10-period Relative Strength Index (RSI) and a 20-period simple moving average for each asset to the state representation seemed to help the agent learn more robust features. This idea came from reading a few QuantConnect forum discussions where traders shared features they found useful for their own algorithmic strategies.

*   **The Reward Function:** This was critical and took a lot of iteration. My ultimate goal was to optimize the Sharpe ratio, but the Sharpe ratio is typically calculated over a period, not per time step. My first attempt at a reward function was just the portfolio's percentage return at each step. This led to the agent taking on excessive risk to maximize short-term gains. Then I tried penalizing high volatility in the reward, which helped a bit.
    Eventually, I settled on a hybrid approach: a small reward based on the log return of the portfolio at each step, combined with a much larger terminal reward at the end of each episode (e.g., one simulated year of trading) based on the Sharpe ratio calculated over that entire episode. I also added a small penalty for transaction costs every time the agent rebalanced the portfolio. This seemed to encourage more stable, long-term behavior.

*   **The Agent (The Neural Network):** For the PPO agent, I needed an actor network (to decide the actions/weights) and a critic network (to estimate the value of a state). Given the time-series nature of market data, I decided to use LSTMs in both.

    Here’s a simplified version of what my Actor network looked like after a few iterations. Getting the LSTM input and hidden state dimensions correct was a source of many early bugs. I remember specifically struggling with the `batch_first=True` argument and ensuring the hidden states `h_n` and `c_n` were correctly passed between sequential calls when an episode was running. The PyTorch documentation on LSTMs became my constant companion.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ActorNetwork(nn.Module):
        def __init__(self, input_feature_dim, lstm_hidden_dim, fc_hidden_dim, num_assets):
            super(ActorNetwork, self).__init__()
            self.num_assets = num_assets
            self.lstm = nn.LSTM(input_feature_dim, lstm_hidden_dim, batch_first=True)
            # input_feature_dim would be num_assets * (features_per_asset like price_return, rsi, ma)
            self.fc1 = nn.Linear(lstm_hidden_dim, fc_hidden_dim)
            self.fc_policy_head = nn.Linear(fc_hidden_dim, num_assets) # Outputs raw scores for softmax

            # I remember trying different initializations but settled on PyTorch defaults for most layers
            # after finding no clear benefit from more complex ones for my specific setup.

        def forward(self, state, hidden_cell=None):
            # state shape: (batch_size, seq_len, input_feature_dim)
            # For single step inference during episode rollout, seq_len is 1
            # During training with collected trajectories, seq_len can be longer
            
            # This unsqueezing was a common point of error if the input wasn't shaped correctly.
            # Especially when passing single states vs batches.
            if state.ndim == 2: # If state is (batch_size, features) or (features) for single step.
                 state = state.unsqueeze(1) # Add sequence length dimension

            lstm_out, (h_n, c_n) = self.lstm(state, hidden_cell)
            
            # We want the output of the last LSTM cell
            last_lstm_out = lstm_out[:, -1, :] 
            
            x = F.relu(self.fc1(last_lstm_out))
            action_scores = self.fc_policy_head(x)
            
            # Apply softmax to get portfolio weights
            # Adding a small epsilon for numerical stability with softmax, learned this the hard way
            action_probabilities = F.softmax(action_scores, dim=-1) + 1e-8 
            
            return action_probabilities, (h_n, c_n)

    # The Critic was structured similarly but output a single value V(s)
    ```
    My first version of this network was purely fully-connected layers. It learned, but very slowly and seemed to get stuck in local optima. Introducing LSTMs significantly improved its ability to pick up on temporal patterns, although it also increased training time and complexity. The `action_probabilities` directly represent the portfolio weights for each asset. I had to ensure these summed to 1, which the softmax layer conveniently handles.

**Training Trials and Tribulations**

Training the PPO agent was a lengthy process of trial and error. Key PPO hyperparameters like the learning rate for the actor and critic, the clipping parameter `epsilon`, the number of epochs per data collection phase, and the `gamma` and `lambda` for Generalized Advantage Estimation (GAE) all interacted in complex ways. I spent a lot of time looking at sample PPO implementations online, like the ones from OpenAI Baselines or Stable Baselines3, to get a sense of reasonable starting ranges for these.

I didn't have access to a powerful GPU cluster, so training was mostly done on my somewhat aging laptop CPU (with a modest NVIDIA GPU that PyTorch could use, thankfully). This meant each training run for a decent number of timesteps could take hours, sometimes overnight. I remember one particularly painful moment when I discovered a bug in my reward calculation logic after a 12-hour training run, invalidating the results. Debugging RL can be tricky because the feedback loop is so long. It’s not like a supervised learning problem where you can immediately see if your loss is going down. Here, the agent might learn nothing for thousands of steps before suddenly showing improvement, or vice-versa.

A breakthrough moment came when I started systematically logging not just the rewards but also the entropy of the policy and the value loss. Plotting these helped me understand if the agent was exploring enough or if the value function was being learned correctly. I found a fantastic StackOverflow answer that discussed how to interpret these diagnostic plots for PPO, which was immensely helpful in guiding my hyperparameter tuning.

**Results: A Modest Improvement**

After numerous iterations, hyperparameter tuning (mostly manual, with some ad-hoc grid searching when I was really stuck), and many hours of training, the agent started to exhibit sensible behavior. It learned to diversify, and its allocations seemed to react to changing market characteristics in the simulation.

To evaluate its performance, I used the hold-out test set that the agent had never seen during training. I compared the agent's performance against a couple of baselines: a simple buy-and-hold strategy of an equally weighted portfolio and a periodically rebalanced (monthly) equally weighted portfolio.

The AI-driven portfolio, managed by the PPO agent, achieved a Sharpe ratio that was approximately 15% higher than the equally weighted rebalanced portfolio on the test data. While not a groundbreaking result that would make me rich overnight, it was a clear, measurable improvement. The agent seemed to be better at navigating periods of higher volatility by reducing overall risk exposure or shifting allocations more dynamically.

**Reflections and What’s Next**

This project was one of the most challenging yet rewarding experiences I've had. The process of wrestling with RL concepts, translating them into working PyTorch code, debugging the subtle issues, and patiently waiting for models to train taught me a lot about persistence.

If I were to continue this, there are several things I'd explore:
*   **More Sophisticated State Representation:** Incorporating more advanced features, perhaps from alternative data sources if available, or using attention mechanisms in the neural network to better weigh the importance of different features or time steps.
*   **Different RL Algorithms:** Now that I have a better grasp, I might revisit DDPG or try SAC (Soft Actor-Critic), which is known for its sample efficiency and stability.
*   **Real-World Data & Execution:** The ultimate test would be to adapt this to historical (and eventually live) market data, though that introduces a host of new challenges like API integration, much higher data quality requirements, and the reality of slippage and transaction costs.
*   **Parameter Uncertainty:** The current model outputs weights, but doesn't explicitly account for the uncertainty in its own estimates or the market parameters. Exploring Bayesian RL approaches could be interesting.

Overall, I’m pleased with how this turned out. Building something from the ground up, especially in a complex domain like RL for finance, and seeing it actually *learn* and achieve a quantifiable improvement, was incredibly satisfying. It definitely solidified my interest in AI and its practical applications.