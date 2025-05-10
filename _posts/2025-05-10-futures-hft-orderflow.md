---
layout: post
title: Futures HFT Strategy with Order Flow
---

## Diving Deep: Building a Futures HFT Strategy with Order Flow in C++

This semester, I decided to really push my C++ skills and dive into an area that's fascinated me for a while: high-frequency trading. Specifically, I wanted to explore strategies based on order flow data. After a fair bit of background reading, I settled on a project: analyzing **order book imbalance** signals for **Hang Seng Index futures** and implementing a simulated low-latency trading system in **C++**. This turned out to be a much bigger rabbit hole than I initially anticipated.

### The Initial Spark: Understanding Order Book Imbalance

The core idea seemed straightforward enough: if there's significantly more volume on the bid side of the order book compared to the ask side (or vice-versa) close to the current price, it might indicate short-term price pressure. The Hang Seng Index futures (HSI) market felt like a good candidate – it's liquid and volatile enough for such signals to potentially be meaningful.

My first task was to define "imbalance." I read a few papers and a bunch of forum posts on QuantConnect and Elite Trader. Many definitions exist, but I decided to start with a relatively simple one: the ratio of the cumulative volume of the first *N* bid levels to the cumulative volume of the first *N* ask levels. I parameterized *N* so I could experiment with it later.

The theoretical part was one thing, but I knew the real challenge would be processing market data updates and maintaining an accurate order book representation in a performant way. This screamed C++.

### Setting the Stage: Data and Simulation Environment

Actually getting live HSI futures order book data for a personal project without paying hefty fees is, well, difficult. For development and testing, I decided to build a simulator that could replay historical tick data or even generate synthetic data based on certain patterns. I managed to find some sample historical tick data for HSI futures from a university archive, which included Level 2 order book updates. It wasn't perfect, but it was enough to get started.

My simulator needed to feed my C++ strategy with discrete events: new orders, cancellations, trades. Each event would carry a timestamp, price, volume, and side.

### The Core: C++ for Low Latency

This was where the bulk of the work (and headaches) lay. My main goals were:
1.  Efficiently maintain the state of the order book.
2.  Calculate order book imbalance quickly upon updates.
3.  Process events with minimal latency.

For the order book, I initially thought of using `std::map<double, int>` (price to quantity) for both bids and asks. Bids would be sorted descending, asks ascending. `std::map` gives you O(log N) insertions, deletions, and lookups.

```cpp
// Initial thoughts for order book levels
// Bids are ordered from highest price to lowest
std::map<double, int, std::greater<double>> bids_;
// Asks are ordered from lowest price to highest
std::map<double, int> asks_;

// Function to update a level or insert/remove if quantity is zero
void update_book_level(double price, int quantity, bool is_bid) {
    if (is_bid) {
        if (quantity == 0) {
            bids_.erase(price);
        } else {
            bids_[price] = quantity;
        }
    } else {
        // Similar logic for asks_
        if (quantity == 0) {
            asks_.erase(price);
        } else {
            asks_[price] = quantity;
        }
    }
    // After updating, I'd need to recalculate imbalance
    // This became a performance concern quickly
}
```

The problem with `std::map` became apparent when I started thinking about frequent updates and the need to iterate over the top *N* levels to calculate imbalance. While logarithmic complexity is good, the constant factors for map operations and cache performance aren't always ideal for HFT. Every update meant potentially re-iterating.

I spent a good week just on this. I looked into `boost::flat_map` and even considered custom skip lists or B-tree like structures after stumbling upon some discussions on StackOverflow about HFT order book implementations. Some people recommended just using `std::vector` and keeping it sorted, especially if the number of price levels isn't enormous. For updates far from the BBO (Best Bid Offer), this would be slow (linear time to find and insert/erase). But for updates *near* the BBO, if I could quickly find the spot, it might be faster due to better cache locality.

Given my time constraints and current C++ knowledge, I stuck with `std::map` for the first pass but made a mental note to revisit this if profiling showed it as a major bottleneck. The primary goal was to get a working system first.

Calculating imbalance then involved iterating through the maps:

```cpp
// Simplified imbalance calculation
// depth_levels_ is the N I mentioned earlier
double calculate_imbalance() {
    double bid_volume_sum = 0;
    int levels_counted = 0;
    for (const auto& [price, volume] : bids_) {
        bid_volume_sum += volume;
        levels_counted++;
        if (levels_counted >= depth_levels_) break;
    }

    double ask_volume_sum = 0;
    levels_counted = 0;
    for (const auto& [price, volume] : asks_) {
        ask_volume_sum += volume; // Accumulate ask volume
        levels_counted++;
        if (levels_counted >= depth_levels_) break;
    }

    if (ask_volume_sum + bid_volume_sum == 0) { // Avoid division by zero
        return 0.5; // Neutral imbalance
    }
    return bid_volume_sum / (bid_volume_sum + ask_volume_sum);
}
```
This calculation had to be triggered every time the book changed in a way that could affect the top N levels.

### Event Processing and Decision Logic

I set up a simple event loop. My market data simulator would push `MarketDataEvent` objects into a queue, and my strategy's main thread would pop them off and process them.

```cpp
struct MarketDataEvent {
    enum class UpdateType { ADD, MODIFY, DELETE, TRADE };
    UpdateType type;
    long long timestamp_ns; // Nanosecond timestamp
    double price;
    int quantity;
    bool is_bid; // True for bid, false for ask
    // For trades, price and quantity refer to the trade event itself
};

// In my main processing loop
// std::deque<MarketDataEvent> event_queue_;
// std::mutex queue_mutex_; // To protect the queue if feeder is on another thread

void process_event(const MarketDataEvent& event) {
    // Update order book (bids_ or asks_)
    // This is where I'd call update_book_level or similar
    // ...

    if (event.type != MarketDataEvent::UpdateType::TRADE) {
        // A book update might change imbalance
        current_imbalance_ = calculate_imbalance();
        
        // Decision logic
        if (current_imbalance_ > entry_threshold_long_ && !is_long_) {
            // Place simulated buy order
            std::cout << "Time: " << event.timestamp_ns
                      << " Imbalance: " << current_imbalance_
                      << " Decision: Enter Long at " << asks_.begin()->first << std::endl;
            // In a real system, this would involve an OrderManager
            // For simulation, I just recorded the hypothetical trade
            sim_enter_position(asks_.begin()->first, true, event.timestamp_ns);
            is_long_ = true;
        } else if (current_imbalance_ < entry_threshold_short_ && !is_short_) {
            // Place simulated sell order
             std::cout << "Time: " << event.timestamp_ns
                      << " Imbalance: " << current_imbalance_
                      << " Decision: Enter Short at " << bids_.begin()->first << std::endl;
            sim_enter_position(bids_.begin()->first, false, event.timestamp_ns);
            is_short_ = true;
        }
        // Add logic for exiting positions based on imbalance shift or stop-loss/take-profit
    }
}
```

One major struggle here was handling timestamps and ensuring causality. My simulated data had nanosecond precision, and I wanted my logic to reflect that. Debugging timing-related issues was tricky; `std::cout` debugging can only get you so far before it starts affecting performance itself. I spent a lot of time just stepping through the event stream manually with a small dataset to ensure the book was being constructed correctly.

### Breakthroughs and Bottlenecks

A significant "aha!" moment came when I realized how critical it was to only recalculate imbalance when *relevant* book levels changed. Initially, I was recalculating it on almost every single update. This was wasteful. I refined the logic to check if the update was within the `depth_levels_` I cared about.

Performance-wise, even with `std::map`, for the small `depth_levels_` (e.g., 3 to 5 levels) I was testing, the `calculate_imbalance()` function wasn't the absolute slowest part for moderate update rates. The data parsing from my simulated feed (which was just reading text lines from a file initially) and the overhead of the event queue itself were also contributing factors. I later optimized the data ingestion by memory-mapping the input file and parsing records more directly, which gave a noticeable speedup.

I also considered using a lock-free queue for events, reading up on `boost::lockfree::queue`, but decided against it for this iteration. The complexity of getting lock-free programming right is high, and since my data feeder and processor weren't yet on truly separate, high-contention threads (more of a pipelined approach for now), `std::mutex` around `std::deque` was "good enough" to start. This was a pragmatic choice based on limited time. If I were to scale this to handle truly massive data rates, this would be one of the first things to refactor.

Another tricky bit was defining the entry/exit logic. Just because imbalance crosses a threshold doesn't mean you instantly get a fill at the BBO. For simulation, I assumed fills at the current opposite BBO if the signal was strong. I also added simple stop-loss and take-profit logic based on fixed price offsets, though a more advanced system would use dynamic stops or trailing stops.

### Simulated Trading and Initial Observations

My "simulated trading" was quite basic: when the strategy decided to "trade," it would record the entry price, direction, and timestamp. An exit would be triggered by an opposite imbalance signal, a stop-loss, or a take-profit. I then wrote a separate Python script to parse these trade logs and calculate basic P&L metrics.

The initial results were… mixed. There were periods where the imbalance signal seemed to predict short-term moves quite well, and other periods where it just led to a series of small losses (death by a thousand cuts). This highlighted the importance of parameter tuning (the imbalance threshold, the number of book levels to consider) and potentially incorporating other factors like trade flow or market volatility.

One specific issue I encountered was "flickering" signals. The imbalance could rapidly oscillate around my threshold, causing quick entries and exits – essentially churning commissions in a real scenario. I had to implement some debouncing logic, like waiting for the imbalance to persist for a certain number of updates or a minimum time duration before acting.

```cpp
// Sketch of debouncing idea within process_event
// ... previous logic ...
if (potential_signal_ == SignalType::NONE && new_imbalance_state != current_imbalance_state_) {
    potential_signal_ = new_imbalance_state;
    potential_signal_timestamp_ = event.timestamp_ns;
    potential_signal_confirmations_ = 1;
} else if (potential_signal_ != SignalType::NONE && new_imbalance_state == potential_signal_) {
    potential_signal_confirmations_++;
    if (potential_signal_confirmations_ >= required_confirmations_ &&
        (event.timestamp_ns - potential_signal_timestamp_) >= min_signal_duration_ns_) {
        // Confirmed signal, proceed to trade logic
        // ... trade ...
        current_imbalance_state_ = new_imbalance_state; // Update confirmed state
        potential_signal_ = SignalType::NONE; // Reset potential signal
    }
} else if (new_imbalance_state != potential_signal_) {
    // Signal changed before confirmation, reset
    potential_signal_ = SignalType::NONE;
}
// ...
```
This `SignalType` would be an enum for `STRONG_BID_IMBALANCE`, `STRONG_ASK_IMBALANCE`, `NEUTRAL`. This added more state to manage but was necessary.

### Reflections and Next Steps

This project was an incredible learning experience. Building even a simulated HFT system from scratch in C++ is a significant undertaking. I gained a much deeper appreciation for the nuances of low-latency programming, data structure choices, and the practical challenges of implementing trading ideas.

If I were to continue this, my next steps would be:
1.  **Rigorous Backtesting**: Use more comprehensive historical data and a more robust backtesting framework. My current Python script for P&L is very basic. I'd look into something like QuantLib or build out my C++ backtester further.
2.  **Order Book Optimization**: Profile the `std::map` based order book more thoroughly under heavy load and likely experiment with sorted `std::vector` or other specialized structures. I found a few GitHub repositories with HFT order book implementations that I'd want to study more.
3.  **More Sophisticated Signals**: Combine order book imbalance with other microstructural features, like trade flow intensity or VWAP pressure.
4.  **Realistic Execution Model**: Currently, I assume fills at BBO. A real simulation needs to consider queue position, fill probability, and market impact.
5.  **Concurrency**: Properly explore multi-threading for different components (data feed handling, signal computation, order execution simulation) and the joys of lock-free data structures.

Even though it's just a simulation, seeing my C++ code react to market events and make "decisions" was incredibly rewarding. It's a complex field, but breaking it down piece by piece made it manageable, and the hands-on coding was where the real learning happened. There were many moments of frustration, especially when debugging subtle bugs related to book state or event order, but finally seeing a clean run of data produce plausible (even if not always profitable) trades was worth it.