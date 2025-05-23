---
layout: post
title: RL Optimal Trade Execution
---

## RL for Optimal Trade Execution: A Deep Dive into my PPO Agent Project

This project has been a significant undertaking for me over the past few months, and I'm finally at a stage where I can share some of the development process and learnings. The goal was to build a Reinforcement Learning agent capable of executing trades in a way that minimizes slippage against common benchmarks like Time-Weighted Average Price (TWAP) and Volume-Weighted Average Price (VWAP). I opted to implement a Proximal Policy Optimization (PPO) agent in Python, leveraging LOBSTER limit order book (LOB) data for training and PyTorch Lightning for a more streamlined training pipeline.

### The Core Problem: Navigating the Microstructure for Better Execution

The idea of optimal trade execution isn't new, but applying deep RL to it, especially with high-fidelity LOB data, felt like a compelling challenge. When a large order is placed, breaking it down into smaller "child" orders and timing their execution can significantly impact the average price obtained. Poor execution leads to slippage – the difference between the expected price and the actual execution price. My aim was to train an agent to make these sequencing and sizing decisions intelligently.

Initially, I considered more traditional algorithmic trading strategies, but the dynamic nature of the LOB and the potential for an agent to learn complex patterns made RL an attractive, albeit ambitious, path. VWAP and TWAP are standard benchmarks, so minimizing slippage against them became the concrete objective.

### Data Wrangling: The LOBSTER Experience

For any data-driven project, the dataset is foundational. I used LOBSTER (Limit Order Book System - The Efficient Reconstructor), which provides incredibly detailed message-by-message data for NASDAQ-traded stocks. I chose a couple of liquid stocks over a few months to get a decent amount of data.

The raw LOBSTER data comes as message files and orderbook files. The first hurdle was parsing these efficiently to reconstruct LOB snapshots at specific time intervals or upon certain event triggers. Each message (submission, cancellation, deletion, execution) modifies the book. My initial Python scripts for this were painfully slow. I spent a good week optimizing this, moving from iterating row-by-row with pandas to more vectorized operations where possible, and carefully managing the state of the order book. I had to reconstruct a snapshot of the top N levels of the bid and ask sides of the book. I settled on using the top 10 levels, as deeper levels felt like they'd add too much noise for the agent, at least initially.

A significant challenge was time synchronization and deciding how to represent the LOB state for the agent. The raw data is timestamped to nanoseconds. I decided to discretize time into fixed intervals for agent decision-making, but also considered an event-driven approach where the agent acts after a certain volume has traded in the market. The fixed interval seemed simpler to start with.

### Designing the Trading Environment

This was probably the most iterative part of the project. An RL agent learns by interacting with an environment, so defining the state space, action space, and reward function correctly is critical.

**State Representation:**
The state needed to give the agent enough information about the current market conditions. I included:
*   The current bid-ask spread.
*   Prices and volumes for the top 10 levels of the LOB on both bid and ask sides.
*   The remaining quantity of the parent order to be executed.
*   The time remaining in the execution window (e.g., if I have to execute an order over 30 minutes).
*   Recent trade history (e.g., volume and price of the last few market trades).

Normalization of these inputs was also a big deal. Prices are non-stationary, so I ended up using price differences relative to the current mid-price, and normalizing volumes by some rolling average or the total volume to be executed. Getting this wrong meant the agent often received wildly different input scales, which isn't great for neural network stability. I recall one instance where my volume features were not properly scaled, leading to the agent either executing everything at once or nothing at all, because the network outputs were saturated.

**Action Space:**
The agent's task is to decide what portion of the remaining order to place at what price. I simplified this to start:
*   Action: A discrete value representing a percentage of the remaining order to execute (e.g., 0%, 10%, 20%, ..., 100%).
*   Pricing: Initially, I had the agent submit market orders to simplify things. Later, I experimented with allowing it to choose to place a limit order at the current best bid/ask, or one tick more aggressively. This significantly increased the complexity. For the PPO agent discussed here, I stuck to submitting aggressive limit orders (buy at ask, sell at bid) or market orders, essentially choosing *when* and *how much*.

I considered a continuous action space for the order quantity, but PPO works more directly with discrete actions, and given my initial struggles, simpler seemed better. I could always revisit this.

**Reward Function:**
The reward function is what guides the agent. I wanted to minimize slippage against VWAP calculated over the execution horizon.
My reward at each step `t` where an execution occurred was:
`reward_t = (VWAP_benchmark_price_t - execution_price_t) * executed_quantity_t`
A positive reward means the agent did better than the benchmark for that chunk. The cumulative reward over the episode would then reflect the total slippage.

One tricky part was the VWAP benchmark. Since the agent's actions influence market prices (at least in simulation), the "true" VWAP is something that would have occurred in its absence. I used the LOB data to calculate VWAP over the period as if my agent hadn't traded, and then compared my agent's execution. This isn't perfect but was a practical starting point. I also added a penalty for not executing the full order by the deadline. This balancing act between aggressive execution to reduce inventory risk and patient execution to get better prices is the core of the problem.

### PPO Agent: The Brains of the Operation

I chose PPO because it's known for its stability and good performance on a variety of tasks. It's an on-policy actor-critic algorithm that uses a clipped surrogate objective function to avoid destructively large policy updates.

The core components were:
*   **Actor Network:** An MLP that takes the state as input and outputs a probability distribution over the discrete actions.
*   **Critic Network:** An MLP that takes the state as input and outputs an estimate of the value function (i.e., expected future reward).

I didn't go for anything overly complex for the network architectures initially: a few hidden layers with ReLU activations.

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Probabilities for discrete actions
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Outputs a single value (state value)
        )

    def forward(self, state):
        return self.network(state)

# Later in the PPO update logic:
# ...
# advantages = rewards_to_go - values
# advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages
# ...
# ratio = torch.exp(new_log_probs - old_log_probs)
# clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
# actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
# critic_loss = F.mse_loss(values, rewards_to_go)
# entropy_bonus = -dist.entropy().mean()
# total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_bonus
```

Hyperparameter tuning for PPO was a journey. `clip_epsilon`, learning rates for actor and critic, `gamma` for discount factor, `gae_lambda` for Generalized Advantage Estimation, number of epochs per data collection phase, minibatch size... it was a lot. I spent quite some time reading through the original PPO paper and a few implementation guides online, like the one from OpenAI Spinning Up, to get a better feel for reasonable ranges. Calculating GAE correctly, especially ensuring tensors were detached appropriately to prevent incorrect gradient flow, was a source of bugs that took a while to iron out. I remember a specific issue where my advantages were always near zero, which I eventually traced back to a mistake in how I was handling the `done` flags in the GAE computation.

### PyTorch Lightning to the Rescue (Mostly)

Managing the training loop, optimizers, logging, and device placement (CPU/GPU) manually in PyTorch can get messy. I'd used PyTorch Lightning for a previous course project and found it really helpful for abstracting away boilerplate.

I structured my PPO agent within a `LightningModule`:

```python
import pytorch_lightning as pl

class PPOAgent(pl.LightningModule):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, clip_epsilon, gae_lambda, entropy_coef, ...):
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.env = TradingEnvironment(...) # My custom environment
        self.buffer = RolloutBuffer(...) # To store trajectories

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_epsilon = clip_epsilon
        # ... other PPO params

    def collect_trajectories(self):
        # Interact with self.env and store (s, a, r, s', log_prob, done) in self.buffer
        # This part is fairly standard RL data collection
        pass

    def training_step(self, batch, batch_idx): # This batch is actually indices for our buffer
        # Sample from self.buffer based on PPO's on-policy nature
        # For PPO, we usually collect a batch of trajectories, then update multiple times
        # This is simplified here; PyTorch Lightning usually expects a dataloader.
        # I had to adapt its paradigm a bit for on-policy RL.
        
        # Retrieve data from buffer
        states, actions, old_log_probs, advantages, rewards_to_go = self.buffer.get_tensors()

        # Calculate actor loss
        dist = Categorical(self.actor(states))
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs.detach()) # old_log_probs from buffer
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.hparams.clip_epsilon, 1 + self.hparams.clip_epsilon) * advantages.detach()
        actor_loss = -torch.min(surr1, surr2).mean()

        # Calculate critic loss
        values = self.critic(states).squeeze()
        critic_loss = nn.functional.mse_loss(values, rewards_to_go.detach())

        # Calculate entropy bonus
        entropy_bonus = dist.entropy().mean()
        
        total_loss = actor_loss + self.hparams.value_loss_coef * critic_loss - self.hparams.entropy_coef * entropy_bonus
        
        self.log_dict({
            "train/actor_loss": actor_loss,
            "train/critic_loss": critic_loss,
            "train/total_loss": total_loss,
            "train/entropy": entropy_bonus
        }, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        return [optimizer_actor, optimizer_critic] # PTL handles multiple optimizers

    # I had to customize the training loop slightly because PPO is on-policy
    # and typically collects a batch of data, then performs multiple optimization epochs on it.
    # The standard PTL fit loop is more geared towards dataset iteration.
    # I ended up using `automatic_optimization = False` and manually calling optimizer steps
    # within a custom outer loop that managed trajectory collection.
    # This was a bit of a learning curve with PTL.
```

One challenge with PyTorch Lightning for PPO was that PPO is on-policy and often involves collecting a large batch of rollouts, then performing several epochs of gradient updates on that batch. PTL's default training loop is more geared towards iterating through a `DataLoader` that shuffles a static dataset. I eventually found a way by setting `automatic_optimization = False` and managing the `optimizer.step()` calls myself, after collecting trajectories in each "outer" training step. This gave me the control I needed for the PPO update structure. There were some discussions on the PyTorch Lightning GitHub issues page that pointed me in this direction.

### The Training Grind

Training RL agents is notoriously sample-inefficient. I ran training for many iterations, which took a considerable amount of time even with a reasonably powerful GPU. I used TensorBoard (which integrates nicely with PTL) to monitor key metrics: cumulative reward, actor loss, critic loss, and entropy.

The entropy term was particularly important. Initially, without enough entropy bonus, the policy would sometimes prematurely converge to a suboptimal deterministic action. Watching the entropy gradually decrease as the agent became more confident was a good sign.

There were periods where the agent just wouldn't learn. The reward would stay flat or even decrease. This usually meant going back to the drawing board:
*   Is the state representation informative enough?
*   Is the reward function properly incentivizing the desired behavior? For instance, I had an issue where the penalty for not executing the full order was too small, so the agent learned to be overly patient and often failed to complete the parent order.
*   Are the hyperparameters for PPO way off? I once had the learning rate too high, and the policy oscillated wildly.
*   Bugs in the GAE calculation or the advantage normalization. This was a recurring theme.

Debugging was often a process of logging everything, staring at TensorBoard graphs, and sometimes stepping through the code line by line during the update step to see where values might be exploding or vanishing.

### Preliminary Results & Reflections

After many cycles of tuning and debugging, the agent started to show promising behavior. I evaluated it by running it in the simulated environment over unseen periods of LOB data and comparing its execution price against the TWAP/VWAP benchmarks for those periods.

The agent did learn to achieve positive slippage (i.e., better prices than the benchmark) in many scenarios, especially in moderately volatile conditions where timing actually matters. It tended to break down larger orders and place them more strategically than a naive "slice and dice" TWAP execution strategy.

However, it wasn't a silver bullet. In very calm markets, it was hard to beat a simple TWAP. In extremely chaotic, fast-moving markets, the agent sometimes struggled to adapt quickly enough, possibly due to the discrete time steps or the limitations of its state representation.

One specific "aha!" moment came when I improved the state representation to include not just the current LOB snapshot but also a short history of recent LOB changes (e.g., delta in best bid/ask, volume traded at bid/ask). This seemed to give the agent a better sense of market momentum and improved its performance noticeably.

### Challenges, Learnings, and Next Steps

This project was a massive learning experience.
*   **Data is Hard:** Preprocessing and cleaning financial data, especially high-frequency LOB data, is non-trivial and time-consuming.
*   **RL is Fiddly:** Reward shaping, hyperparameter tuning, and debugging RL algorithms require a lot of patience and experimentation. What works for one environment might not work for another. I must have read dozens of StackOverflow posts and GitHub issues related to PPO implementations and PyTorch subtleties.
*   **Environment is Key:** The fidelity of the simulation environment significantly impacts the agent's learned behavior and its transferability to real-world scenarios (though this project was purely simulation-based). My market impact model was very simplistic (assuming my trades execute at current LOB prices without impacting them beyond consuming liquidity). A more realistic model would be a major next step.
*   **PyTorch Lightning:** While great for structure, adapting it to on-policy RL algorithms like PPO required some deeper understanding of its internals. But once set up, it made experimentation much faster.

**What I'd do differently or explore next:**
*   **More Sophisticated State Representation:** Incorporate more features, perhaps using LSTMs or Transformers to capture temporal dependencies in the LOB dynamics.
*   **Continuous Action Space:** Explore algorithms like Soft Actor-Critic (SAC) or DDPG that can handle continuous actions for order size and price.
*   **Better Market Impact Model:** This is crucial for more realistic simulations.
*   **Multi-Agent Systems:** Consider how multiple agents might interact, or how to model the behavior of other market participants.
*   **Transfer Learning:** Train on a variety of assets and market conditions to see if the agent can generalize.

This has been one of the most challenging yet rewarding projects I've worked on. There's a long way to go to create something robust enough for real-world application, but as a deep dive into applying RL to a complex financial problem, it's been an invaluable experience.