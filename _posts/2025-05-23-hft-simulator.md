---
layout: post
title: HFT Market Microstructure Simulator
---

### Diving Deep: Building an HFT Market Microstructure Simulator

For a while now, I've been fascinated by market microstructure and the outsized impact High-Frequency Trading (HFT) can have on it. Reading papers on price discovery and liquidity in the presence of HFTs is one thing [5, 6, 7, 8, 10], but I really wanted to get my hands dirty and see these dynamics play out. That led to my latest project: building a simulator for limit order book (LOB) dynamics and observing how different HFT agent behaviors affect things like price formation. This turned out to be a much deeper rabbit hole than I initially anticipated, involving a fair bit of Python, a dive into C++, and eventually, wrangling JAX for performance.

#### The Starting Point: Prototyping the LOB in Python

My first instinct for the Limit Order Book was to use Python. It's my go-to for quick prototyping, and I figured a dictionary of lists or perhaps a `collections.defaultdict(list)` mapping price levels to lists of orders would be straightforward. Something like:

```python
class OrderBookPy:
    def __init__(self):
        self.bids = {} # price -> [order1, order2, ...]
        self.asks = {} # price -> [order1, order2, ...]
        self.orders = {} # order_id -> order_details for quick cancellation

    def add_order(self, order):
        # Naive add, sort later or manage sorted lists
        self.orders[order.id] = order
        side_book = self.bids if order.side == 'buy' else self.asks
        if order.price not in side_book:
            side_book[order.price] = []
        side_book[order.price].append(order)
        # Ensure price levels are sorted for BBO
        # This became a bottleneck quickly

    def cancel_order(self, order_id):
        if order_id in self.orders:
            order_to_cancel = self.orders[order_id]
            side_book = self.bids if order_to_cancel.side == 'buy' else self.asks
            if order_to_cancel.price in side_book:
                try:
                    # This was also slow, list removal is O(N)
                    side_book[order_to_cancel.price].remove(order_to_cancel)
                    if not side_book[order_to_cancel.price]:
                        del side_book[order_to_cancel.price]
                    del self.orders[order_id]
                    return True
                except ValueError:
                    # Order not found in list, inconsistency
                    pass
        return False

    # ... matching logic would go here
```

It quickly became apparent that this approach was too slow for simulating anything resembling HFT. Maintaining sorted price levels and updating lists of orders, especially for cancellations or modifications deep within a price level, was computationally expensive. Even using `sortedcontainers.SortedDict` as some online examples suggested [12], while an improvement, wasn't cutting it when I started thinking about thousands of events per second. I profiled the code using `cProfile` and `snakeviz`, and the bottlenecks were clear: sorting price levels and searching/removing orders from lists.

#### The Inevitable Leap to C++ for the LOB Core

The internet is full of advice on fast LOB implementations, and a common theme is C++ for performance-critical parts. [2, 11, 20, 27] I’d read a few blog posts and forum discussions, like one on GitHub Gist detailing LOB data structures [11] and a Reddit thread in r/quant discussing professional setups [20]. The consensus seemed to be `std::map` for price levels (keyed by price, giving sorted iteration) and `std::list` or `std::deque` for orders at each price level (allowing O(1) insertions/deletions at the ends and efficient iteration for time priority).

So, I decided to bite the bullet and implement the LOB itself in C++. This was a learning curve. My C++ was a bit rusty, and dealing with pointers, memory management (even with smart pointers like `std::unique_ptr` and `std::shared_ptr`), and CMake build systems took some ramp-up time. [4]

My C++ LOB structure looked something like this:

```cpp
#include <map>
#include <list>
#include <memory> // For std::shared_ptr

struct Order {
    uint64_t id;
    // char side; // 'B' or 'S'
    bool is_buy;
    double price;
    uint32_t qty;
    // Add timestamp for time priority if not implicit by list order
    // Other agent_id, etc.
};

class OrderBookCpp {
public:
    using OrderList = std::list<std::shared_ptr<Order>>;
    // Using std::greater for bids to have highest price first
    std::map<double, OrderList, std::greater<double>> bids_;
    std::map<double, OrderList> asks_; // Default std::less for asks

    // For O(1) cancellation
    std::unordered_map<uint64_t, OrderList::iterator> order_id_to_iter_map_;
    std::unordered_map<uint64_t, double> order_id_to_price_map_; // To know which book (price level) the iter belongs to
    std::unordered_map<uint64_t, bool> order_id_to_side_map_;


    void add_order(const std::shared_ptr<Order>& order) {
        if (order->is_buy) {
            bids_[order->price].push_back(order);
            OrderList::iterator it = std::prev(bids_[order->price].end());
            order_id_to_iter_map_[order->id] = it;
            order_id_to_price_map_[order->id] = order->price;
            order_id_to_side_map_[order->id] = true;
        } else {
            asks_[order->price].push_back(order);
            OrderList::iterator it = std::prev(asks_[order->price].end());
            order_id_to_iter_map_[order->id] = it;
            order_id_to_price_map_[order->id] = order->price;
            order_id_to_side_map_[order->id] = false;
        }
    }

    bool cancel_order(uint64_t order_id) {
        auto map_iter = order_id_to_iter_map_.find(order_id);
        if (map_iter == order_id_to_iter_map_.end()) {
            return false; // Order not found
        }

        OrderList::iterator order_iter = map_iter->second;
        double price = order_id_to_price_map_[order_id];
        bool is_buy = order_id_to_side_map_[order_id];

        if (is_buy) {
            auto price_level_iter = bids_.find(price);
            if (price_level_iter != bids_.end()) {
                price_level_iter->second.erase(order_iter);
                if (price_level_iter->second.empty()) {
                    bids_.erase(price_level_iter);
                }
            }
        } else {
            // Similar logic for asks
            auto price_level_iter = asks_.find(price);
            if (price_level_iter != asks_.end()) {
                price_level_iter->second.erase(order_iter);
                if (price_level_iter->second.empty()) {
                    asks_.erase(price_level_iter);
                }
            }
        }
        order_id_to_iter_map_.erase(order_id);
        order_id_to_price_map_.erase(order_id);
        order_id_to_side_map_.erase(order_id);
        return true;
    }
    // Matching engine logic to come...
};
```
One tricky part was efficiently handling order cancellations. Storing iterators to the `std::list` elements in an `std::unordered_map` keyed by order ID allowed for O(1) average time complexity for finding the order, which was a significant improvement. [19] I spent a good amount of time debugging issues related to iterator invalidation and ensuring the `unordered_map`s were correctly updated.

To use this C++ LOB from Python, I turned to `pybind11`. [33, 37] It’s surprisingly straightforward for basic cases, but there's always a bit of a learning curve with managing object lifetimes and data conversions between the two languages. [34] I had a few issues with passing `std::shared_ptr` and ensuring Python didn't prematurely garbage collect something C++ was still using, or vice-versa. The `pybind11` documentation and a few StackOverflow posts were invaluable here.

```cpp
// In a bindings.cpp file
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic type conversions of STL containers
#include "OrderBookCpp.h" // My LOB header

namespace py = pybind11;

PYBIND11_MODULE(hft_lob_cpp, m) {
    py::class_<Order, std::shared_ptr<Order>>(m, "Order")
        .def(py::init<uint64_t, bool, double, uint32_t>())
        .def_readwrite("id", &Order::id)
        .def_readwrite("is_buy", &Order::is_buy)
        .def_readwrite("price", &Order::price)
        .def_readwrite("qty", &Order::qty);

    py::class_<OrderBookCpp>(m, "OrderBookCpp")
        .def(py::init<>())
        .def("add_order", &OrderBookCpp::add_order)
        .def("cancel_order", &OrderBookCpp::cancel_order)
        // ... other methods like get_bbo, match_orders etc.
        .def("get_bids", [](const OrderBookCpp& book) {
            // Need to convert C++ map to Python dict for easy inspection
            // This can be slow if called frequently for large books
            py::dict bids_py;
            for(const auto& pair : book.bids_) {
                py::list orders_at_price;
                for(const auto& order_ptr : pair.second) {
                    orders_at_price.append(order_ptr); // pybind11 handles shared_ptr
                }
                bids_py[py::float_(pair.first)] = orders_at_price;
            }
            return bids_py;
        });
        // ... similar for asks
}
```
Profiling this hybrid Python/C++ application was interesting. Standard Python profilers like `cProfile` would just show time spent in the C++ extension. [9, 22] Tools like Intel VTune Profiler or `gprof` (for the C++ side) became necessary for a fuller picture. Nvidia's NVTX could also be useful if GPU work was involved, which it wasn't at this stage. [14]

#### Simulating HFT Agents and the JAX Awakening

With a reasonably fast LOB, I moved on to the HFT agents. Initially, these were simple Python classes making decisions based on the BBO (Best Bid and Offer) and their own (rudimentary) inventory management.

```python
class SimpleMarketMakerPy:
    def __init__(self, agent_id, order_book_ref, offset=0.01, spread_target=0.05):
        self.agent_id = agent_id
        self.lob = order_book_ref # Reference to the C++ LOB via pybind11
        self.offset = offset
        self.spread_target = spread_target
        self.current_bid_id = None
        self.current_ask_id = None
        # basic inventory tracking
        self.inventory = 0
        self.max_inventory = 100


    def decide_and_act(self, current_time_step):
        # Simplified logic
        # First, cancel existing orders
        if self.current_bid_id:
            self.lob.cancel_order(self.current_bid_id)
            self.current_bid_id = None
        if self.current_ask_id:
            self.lob.cancel_order(self.current_ask_id)
            self.current_ask_id = None

        # Get current BBO from the C++ LOB
        # This part requires exposing BBO retrieval from C++ LOB to Python
        # best_bid_price, best_ask_price = self.lob.get_bbo() # Assumed method
        
        # For testing, let's assume we can get it.
        # In reality, the get_bids/get_asks I wrote above is too slow for this.
        # I'd need a dedicated get_bbo() in C++ exposed.
        # Let's pretend:
        # mid_price = (best_bid_price + best_ask_price) / 2 if best_bid_price and best_ask_price else 100.0

        # Dummy mid_price for now if get_bbo isn't fully implemented/exposed efficiently
        # This is a common step: mock parts that aren't ready
        bids = self.lob.get_bids() # This is inefficient for just BBO
        asks = self.lob.get_asks() # Inefficient
        
        best_bid_price = max(bids.keys()) if bids else None
        best_ask_price = min(asks.keys()) if asks else None
        
        mid_price = 0
        if best_bid_price is not None and best_ask_price is not None:
            mid_price = (best_bid_price + best_ask_price) / 2.0
        elif best_bid_price is not None:
            mid_price = best_bid_price + self.offset # guess
        elif best_ask_price is not None:
            mid_price = best_ask_price - self.offset # guess
        else:
            mid_price = 100.0 # default if book is empty

        # Adjust quotes based on inventory
        # A very naive inventory adjustment
        adj_factor = (self.inventory / self.max_inventory) * self.offset * 2

        my_bid_price = round(mid_price - self.spread_target / 2 - adj_factor, 2)
        my_ask_price = round(mid_price + self.spread_target / 2 - adj_factor, 2)
        
        # Ensure bid < ask and positive prices
        if my_bid_price >= my_ask_price:
            my_bid_price = round(my_ask_price - 0.01, 2)
        if my_bid_price <= 0: my_bid_price = 0.01


        # Create new orders (simplified: fixed quantity)
        # Need unique order IDs
        new_bid_order_id = current_time_step * 1000 + self.agent_id # simplistic ID
        new_bid = Order(new_bid_order_id, True, my_bid_price, 10)
        self.lob.add_order(new_bid)
        self.current_bid_id = new_bid.id
        
        new_ask_order_id = new_bid_order_id + 1 # simplistic ID
        new_ask = Order(new_ask_order_id, False, my_ask_price, 10)
        self.lob.add_order(new_ask)
        self.current_ask_id = new_ask.id

        # Update inventory (this would happen on fills, not here)
        # This is just a placeholder to show where inventory logic would affect quoting
```

While the C++ LOB was fast, simulating thousands of these Python agents, each performing its decision logic (even if simple), and interacting with the LOB through `pybind11` in a loop, started to become slow again. The overhead of Python loops and function calls via `pybind11` for each agent at each time step added up.

This is where JAX came into the picture. [3, 13, 18, 21] I'd been reading about JAX for its `jax.jit` (Just-In-Time compilation) and `jax.vmap` (automatic vectorization) capabilities, especially in contexts like reinforcement learning where you often simulate many agents or environments in parallel. [16] My thought was: if the agent decision logic could be expressed as a pure function (or close enough), maybe JAX could speed it up significantly.

This was probably the most challenging part of the project conceptually. JAX works best with pure functions and its own array types. [25, 30] My agent logic was inherently stateful (inventory, current orders) and interacted with an external C++ object (the LOB). I couldn't just `jax.jit` the whole `decide_and_act` method.

My compromise was to use JAX for the "core decision computation" part of the agents, if that part could be isolated and fed the necessary market state as JAX arrays. For example, if multiple agents used a similar pricing model, `jax.vmap` could compute their desired quotes in a batch. The "interaction" part (placing/cancelling orders in the C++ LOB) would still happen in a Python loop, but the computation of *what* orders to place could be much faster.

Let's say I wanted to calculate target bid/ask prices for many agents based on a common mid-price and individual risk parameters:

```python
import jax
import jax.numpy as jnp

# Assume this function now takes all necessary state as arguments
# and returns actions/parameters for Python to execute.
# This would be a refactor of parts of SimpleMarketMakerPy.decide_and_act
@jax.jit
def calculate_quotes_jax(mid_prices, agent_spread_targets, agent_offsets, agent_inventories, max_inventories):
    # mid_prices, agent_spread_targets, etc. are JAX arrays
    # This is a vectorized calculation for multiple agents
    
    adj_factors = (agent_inventories / max_inventories) * agent_offsets * 2.0
    
    target_bids = mid_prices - agent_spread_targets / 2.0 - adj_factors
    target_asks = mid_prices + agent_spread_targets / 2.0 - adj_factors
    
    # Ensure bid < ask; JAX uses jnp.where for conditional logic
    target_bids = jnp.where(target_bids >= target_asks, target_asks - 0.01, target_bids)
    target_bids = jnp.maximum(0.01, target_bids) # Ensure positive prices
    
    # Rounding in JAX (can be tricky, might need to handle carefully for financial data)
    # For simplicity, let's assume direct use or a JAX-compatible rounding
    return target_bids, target_asks

# In the main simulation loop:
# 1. Extract relevant state from LOB (e.g., BBO -> mid_price)
# 2. Prepare JAX arrays for agent parameters
# mid_price_for_all_agents = jnp.array([current_mid_price] * num_agents)
# spread_targets_arr = jnp.array([agent.spread_target for agent in agents])
# ... and so on for other parameters

# 3. Call the JAX function
# target_bids_all_agents, target_asks_all_agents = calculate_quotes_jax(
# mid_price_for_all_agents, spread_targets_arr, offsets_arr, inventories_arr, max_inv_arr
# )

# 4. Python loop to interact with C++ LOB using these calculated quotes
# for i, agent in enumerate(agents):
# agent.update_orders_in_lob(target_bids_all_agents[i], target_asks_all_agents[i])
```

The initial hurdle with JAX was its "pure functions" paradigm. [25] If a function has side effects (like printing, or modifying a global variable), `jax.jit` might behave unexpectedly – often, the side effect only happens once during the initial compilation (tracing) phase. [31] Debugging JAX code can also be tricky. [15, 28] Standard Python print statements inside a `jit`ted function don't always work as expected during execution. `jax.debug.print` and `jax.debug.breakpoint` are the way to go, but it took some getting used to. I remember being particularly confused by `ConcretizationTypeError` when I tried to use a dynamic value (a JAX tracer) where a static value was expected, often in control flow or when defining array shapes. [30] Reading the JAX documentation on "Thinking in JAX" and common gotchas was essential. [25]

The breakthrough came when I managed to vectorize the core pricing logic for a group of market makers. Seeing the computation time for that segment drop dramatically after `jax.jit` and `jax.vmap` was a huge motivator. It wasn't a magic bullet for the whole simulation due to the C++ LOB interaction overhead, but it made simulating more agents with more complex (but JAX-ifiable) internal models feasible.

#### Analyzing HFT Impact on Price Discovery

With the simulator somewhat operational, I could finally start looking at price discovery. The main metrics I focused on were:
1.  **Bid-Ask Spread:** How tight are the spreads with different HFT strategies present? [7]
2.  **Price Volatility:** Do certain HFT activities increase short-term volatility?
3.  **Information Incorporation:** How quickly does the market price react to new "fundamental" price shocks I could introduce into the simulation? (Inspired by metrics like Hasbrouck's information shares, though my setup was much simpler). [6, 10]

I collected data by running simulations with different agent populations (e.g., only "human" noise traders, noise traders + market makers, noise traders + aggressive HFTs). The output was typically a time series of the BBO, trades, and order book snapshots. I used Python (Pandas, Matplotlib) for the analysis and visualization.

One of the interesting, though perhaps not entirely surprising, observations was how quickly "stale" limit orders from slower agents were picked off by faster, aggressive agents when a price shock occurred. This often led to a temporary widening of the spread before market makers could re-quote. It also highlighted the importance of cancellation speed for the market-making agents.

#### Specific Challenges and "Aha!" Moments

*   **Floating Point Precision in C++ LOB:** One particularly nasty bug took me a whole weekend to track down. My C++ `std::map` for price levels was sometimes behaving erratically – orders seemed to disappear or not get matched correctly. It turned out to be a subtle floating-point precision issue with the `double` keys for price. Two prices that should have been identical were sometimes a tiny epsilon apart due to calculations, leading them to be treated as separate price levels. I found a StackOverflow answer discussing comparing doubles as map keys, which suggested using a custom comparator with an epsilon or, more robustly, representing prices as scaled integers. I opted for scaling prices to integers (e.g., price * 10000 for 4 decimal places) to avoid the comparison headaches. This was a huge relief when it finally worked.

*   **Pybind11 and `std::vector` of `std::shared_ptr`:** Getting `pybind11` to correctly handle returning, say, a `std::vector<std::shared_ptr<Order>>` from C++ to Python (representing orders at a price level) took some trial and error. The `pybind11/stl.h` and `pybind11/smart_holder.h` headers are supposed to make this seamless, and they mostly do, but ensuring the ownership semantics were right and that Python could iterate over these objects correctly involved careful reading of the `pybind11` docs and looking at their test cases.

*   **JAX, Tracers, and `static_argnums`:** My first attempts at using `jax.jit` on functions that involved some configuration parameters (like an agent's risk aversion) that I didn't want to be traced as JAX arrays were frustrating. The code would recompile too often or throw errors. Understanding `static_argnums` in `jax.jit` (or `static_argnames`) was key. It allows you to specify which arguments to a function should be treated as compile-time constants, preventing unnecessary recompilations if those values change but the *logic* remains the same. This really cleaned up my JAX integration.

#### Future Directions and Reflections

This project has been an incredible learning experience. There are so many ways it could be extended:
*   More sophisticated HFT agent strategies (e.g., statistical arbitrage, latency arbitrage if I modelled multiple exchanges).
*   Introducing a more realistic matching engine with different order types (market, FOK, IOC). [19]
*   Exploring the impact of different market regulations within the simulation, like circuit breakers or minimum resting times for orders. [32, 39]
*   Using more advanced methods for analyzing price discovery, like those mentioned in ECB or FCA working papers. [5, 6]

If I were to start over, I might consider a more unified framework from the beginning, perhaps something like the Mesa library for agent-based modeling but with a pluggable high-performance core. Or even explore if the entire simulation, including the LOB, could be done in JAX if the discrete event nature of LOB updates could be handled efficiently (which is a big 'if' for traditional LOBs, though some research explores JAX for ABMs and economic simulations [3, 13, 18, 21]).

The biggest takeaway for me has been the practical understanding of the trade-offs involved in building high-performance simulation systems. Python is fantastic for overall orchestration and analysis, but for core loops or computationally intensive agent logic, you often need to reach for C++ or specialized libraries like JAX. And bridging these worlds with tools like `pybind11` is a crucial skill. It's definitely deepened my appreciation for the complexities underlying modern financial markets.