---
layout: post
title: Building an Order Book Simulator with a Q-Learning Agent for Optimal Placement
---

Hey everyone,

I wanted to share a project I've been working on recently that combines a few different areas I'm interested in: market dynamics, simulation, and reinforcement learning. The goal was to build a simulation of a limit order book and then train an agent to learn the best strategy for placing an order to maximize its chances of getting filled. It turned out to be quite a journey, involving both C++ and Python, and a dive into Q-learning.

## The Problem: Where to Place Your Order?

So, imagine you want to buy or sell a stock. You usually interact with a **limit order book**. This book contains all the "resting" orders from other people: bids (offers to buy at a certain price) and asks (offers to sell at a certain price).

If you want to buy *immediately*, you can place a "market order" which just takes the best available asking price. But if you want to try for a better price, you place a "limit order" – say, you bid to buy at a price slightly lower than the current best ask.

The challenge is deciding *where* to place that limit order.
*   Place it too aggressively (e.g., bidding very close to the best ask): Higher chance of getting filled quickly, but maybe at a slightly worse price than you could have gotten.
*   Place it too passively (e.g., bidding much lower): You might get a great price *if* the market moves your way, but there's a high chance your order just sits there unfilled as the market moves away.

My project aimed to simulate this environment and see if a reinforcement learning agent could learn a good strategy to balance this trade-off, specifically optimizing for the **fill rate** (the probability of the order getting executed).

## Approach: Agent-Based Simulation + Reinforcement Learning

To model the complex interactions in an order book (many traders placing, canceling, and modifying orders), I decided an **agent-based simulation (ABS)** approach made sense. Instead of trying to model the market with high-level equations, you simulate the individual actions of market participants and see what overall dynamics emerge.

For the "intelligent" agent trying to place its order optimally, **Reinforcement Learning (RL)** seemed like a natural fit. The agent needs to learn a policy (where to place the order) by interacting with an environment (the simulated order book) and receiving feedback (whether its order got filled).

## Building the Simulation Core in C++

Simulating an order book involves handling potentially thousands of orders and matching them quickly. Performance is key here, especially if you want to run many simulations to train an RL agent. That's why I decided to implement the core simulation logic in **C++**.

**Key C++ Components:**

1.  **Order Representation:** A simple `struct` or `class` to hold order details: ID, price, quantity, side (buy/sell), timestamp.
    ```cpp
    // Simplified Order structure
    struct Order {
        long long id;
        double price;
        int quantity;
        bool is_buy_side;
        // Add timestamp, agent ID, etc.
    };
    ```
2.  **Order Book:** The core data structure. I represented the bid and ask sides separately. For efficiency, you need sorted structures. I used `std::map` initially, mapping price levels to lists of orders, but for better performance, especially with frequent insertions/deletions at the best price, using something like linked lists at each price level or custom balanced trees could be considered. Price-time priority is the standard matching rule (best price first, then oldest order at that price).
    ```cpp
    #include <map>
    #include <vector>
    #include <deque> // Using deque for easier front removal

    class OrderBook {
    private:
        // Map price -> deque of orders at that price
        // Bids sorted descending, Asks sorted ascending
        std::map<double, std::deque<Order*>, std::greater<double>> bids;
        std::map<double, std::deque<Order*>> asks;
        long long next_order_id = 0;

        void matchOrders(); // Internal matching logic

    public:
        double getBestBid() const;
        double getBestAsk() const;
        void addOrder(Order* order);
        void cancelOrder(long long order_id);
        // ... other methods like get_depth, etc.
    };
    ```
3.  **Matching Engine:** Logic within the `OrderBook` class that checks if new orders cross the spread (e.g., a new buy order price >= best ask price) and executes trades, removing filled quantities from the book. Implementing price-time priority correctly was a bit tricky.
4.  **Market Agents:** Simple "background" agents that randomly place, cancel, or market-take orders to create some dynamics in the book. These were basic, just generating random actions based on some parameters.

**Challenges in C++:** Performance tuning was definitely a concern. Using standard maps involves some overhead. Ensuring the matching logic was correct and handled all edge cases (partial fills, etc.) took careful testing.

## Python for Control and RL

While C++ was great for the core simulation speed, Python is much friendlier for implementing the RL logic, managing experiments, and analyzing results. I needed a way for these two parts to talk to each other.

I used **pybind11** to create Python bindings for my C++ `OrderBook` class and simulation functions. This allowed me to:
*   Instantiate and control the C++ order book simulation from Python.
*   Step the simulation forward from Python.
*   Get state information (like best bid/ask, depth) from the C++ book into Python for the RL agent.
*   Send actions (place my agent's order) from Python back into the C++ simulation.

```python
# Conceptual Python code using the pybind11 wrapper
# Assume 'SimCore' is the Python module created by pybind11

# Initialize the C++ order book simulator
order_book_sim = SimCore.OrderBook()

# Add some initial random orders maybe
# ...

# --- RL Agent Interaction Loop ---
# Get state from the C++ simulation
current_spread = order_book_sim.getBestAsk() - order_book_sim.getBestBid()
depth_at_bid = order_book_sim.getDepthAtPrice(order_book_sim.getBestBid())
# ... other state features

# RL Agent decides action (e.g., place order at best bid)
action = rl_agent.choose_action(state) # Action could be offset from best bid/ask

# Send action to C++ simulation
my_order_id = order_book_sim.addOrder(SimCore.Order(price=..., quantity=..., is_buy_side=True))

# Run C++ simulation for a step (or until agent's order is filled/timeout)
order_book_sim.step_simulation(num_steps=1)

# Check if our order got filled (need a way to track this)
was_filled = order_book_sim.wasOrderFilled(my_order_id)

# Update RL agent based on reward (fill status)
reward = 1 if was_filled else 0
rl_agent.update(state, action, reward, next_state)
# --- End Loop ---
```
Using `pybind11` worked really well, giving the best of both worlds: C++ speed for the heavy lifting and Python flexibility for the control and learning logic.

## The Q-Learning Agent

I decided to start with **Q-learning**, a classic RL algorithm. It's relatively straightforward and works well when you have discrete states and actions, which seemed like a reasonable starting point.

**RL Setup:**

1.  **State:** What does the agent need to know to make a good decision? I kept it simple initially:
    *   Current spread (best ask - best bid).
    *   Quantity available at best bid and best ask.
    *   Maybe the agent's current order's position relative to the best price (if it's already placed).
    *   *Challenge:* Defining a good, concise state is hard! Too little info, the agent can't learn. Too much, and the state space explodes (curse of dimensionality). I had to discretize continuous values (like spread) into bins.

2.  **Action:** Where can the agent place its (buy) order? I defined a few discrete options:
    *   Place at the current best bid.
    *   Place at best bid - 1 tick.
    *   Place at best bid - 2 ticks.
    *   (Maybe) Place aggressively at the best ask (effectively a market order).
    *   *Challenge:* Defining the right set of actions. Too few might miss good strategies, too many slows down learning.

3.  **Reward:** How do we tell the agent it did well?
    *   Simple: +1 if the order gets fully filled within a certain time limit.
    *   0 otherwise (or maybe a small negative reward for not getting filled).
    *   *Challenge:* Designing rewards that encourage the desired behavior without unintended consequences. Maybe penalize modifying/canceling orders? I stuck to the simple +1 for fill.

**Q-Learning Implementation:**

The agent maintains a Q-table: `Q[state][action]`, storing the expected future reward for taking `action` in `state`. It learns by exploring the environment and updating this table using the Bellman equation:

`Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))`

Where:
*   `alpha` is the learning rate.
*   `gamma` is the discount factor for future rewards.
*   `s'` is the next state after taking action `a`.

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def choose_action(self, state_index):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_space_size) # Explore
        else:
            # Ensure we handle cases where all Q-values for a state are the same
            action_values = self.q_table[state_index]
            best_action_indices = np.where(action_values == np.max(action_values))
            return random.choice(best_action_indices) # Exploit (break ties randomly)


    def update(self, state_index, action_index, reward, next_state_index):
        old_value = self.q_table[state_index, action_index]
        next_max = np.max(self.q_table[next_state_index]) # Q-value of best action in next state

        # Q-learning update rule
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state_index, action_index] = new_value

    # Need functions to map raw state (spread, depth) to a discrete state_index
    # def get_state_index(self, raw_state):
    #    # ... logic to discretize and combine state features into one index ...
    #    pass
```

**Challenges with Q-learning:**
*   **State Discretization:** Finding the right way to bin continuous values like spread was tricky.
*   **Exploration vs. Exploitation:** Tuning `epsilon` (the probability of choosing a random action) was important. Too much exploration, learning is slow. Too little, the agent gets stuck in suboptimal strategies. I used a decaying epsilon over time.
*   **Convergence:** Q-learning can be slow to converge, especially with larger state spaces. Running thousands or millions of simulation steps was necessary.

## Training and Results

I trained the agent by running many simulated "episodes". In each episode, the agent would get a state from the C++ simulator, choose an action (where to place its buy order), place the order via the Python bindings, let the simulation run for a fixed time or until the order was filled/canceled, observe the reward (filled or not), and update its Q-table.

To evaluate performance, I compared my Q-learning agent's fill rate against a simple baseline strategy (e.g., always placing the order at the current best bid). After significant training time, the **Q-learning agent achieved about a 15% higher fill rate** in the simulation compared to the baseline.

This was encouraging! It showed the agent learned *something* about the simulated market dynamics – perhaps learning to place orders slightly more passively when the spread was wide or the queue was deep, or more aggressively in other situations.

**Important Caveat:** This is a **simulated** improvement in a **simplified** environment. Real markets are vastly more complex. My background agents were very basic, and I didn't model things like market impact (large orders moving the price) or sophisticated HFT strategies that exist in reality.

## What I Learned

*   **Agent-Based Simulation:** Got hands-on experience building a simulation from scratch, appreciating the complexities of modeling even simplified market behavior.
*   **C++/Python Integration:** `pybind11` is awesome! It makes combining the strengths of both languages pretty seamless. This is a powerful pattern for performance-critical research code.
*   **Reinforcement Learning:** Implementing Q-learning seems easy conceptually, but making it work effectively involves careful state/action/reward design and lots of tuning. The discretization challenge is real.
*   **Order Books:** Learned the basic mechanics of limit order books and the core trade-offs involved in order placement.

## Future Ideas

This project could be extended in many ways:
*   More realistic background agents (e.g., momentum traders, mean-reversion traders).
*   More sophisticated RL: Using Deep Q-Networks (DQNs) to handle continuous state spaces directly, or policy gradient methods.
*   Adding market impact to the simulation.
*   Modeling latency.
*   Optimizing for other metrics besides fill rate (e.g., minimizing execution cost).

## Conclusion

This was a challenging but really rewarding project. Combining simulation with RL to tackle a problem like optimal order placement felt like a practical application of these techniques. While the C++ simulation core provided the speed, Python and `pybind11` offered the flexibility needed for the RL agent. The 15% simulated fill rate improvement suggests the approach has potential, even if it's just a first step. It definitely gave me a much deeper appreciation for both the complexities of market microstructure and the practical hurdles in applying reinforcement learning.

Let me know if you've worked on similar simulations or have thoughts on this approach!
```