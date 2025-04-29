---
layout: post
title: Low Latency Order Book
---

Building a Low Latency Order Book Simulator

My recent personal project has been tackling the complexities of building a low-latency order book simulator. The motivation stemmed from a desire to truly understand the mechanics behind high-frequency trading systems, not just theoretically, but by grappling with the engineering challenges firsthand. I wanted to build something that could process market data updates extremely quickly and maintain a consistent view of the order book state. This isn't about *executing* trades in a live market (far from it!), but about creating the core component that would sit at the heart of such a system: the order book itself.

Given the performance requirements implied by "low latency," C++20 was the obvious choice. Its control over memory, execution, and powerful features for concurrency and asynchronous operations are pretty much unmatched for this kind of work. For handling potential asynchronous operations, like reading simulated data from a file descriptor or socket, Boost Asio seemed like the standard library extension to go with, providing robust tools for event-driven programming.

The central piece of this project is, of course, the order book data structure itself. My initial thought was quite naive: maybe just two lists (one for bids, one for asks) and sort them whenever an order comes in? That idea lasted about five seconds before I realized sorting lists on every update for potentially millions of orders wouldn't be "low latency." The requirement is to quickly find the best bid and ask prices and efficiently add, remove, or modify orders within the book.

A more promising approach involved using ordered maps. `std::map` in C++ keeps elements sorted, which is perfect for price levels. I decided to use two `std::map<double, ...>`: one for bids (sorted descending by price) and one for asks (sorted ascending by price).

```cpp
// Initial Order Book structure concept
struct Order {
    long long order_id;
    double price;
    int quantity;
    // Add more fields like timestamp, etc.
};

// Using std::map for price levels
std::map<double, std::vector<Order>> bid_levels; // Price -> List of orders at this price
std::map<double, std::vector<Order>> ask_levels;```

This structure allowed quick access to price levels (`bid_levels.begin()`, `ask_levels.begin()` give the best prices). However, updating or deleting a *specific* order within the `std::vector<Order>` at a given price level was still slow. Finding an order by its `order_id` meant iterating through the vector. If there were many orders at the same price, this linear scan would hurt performance.

So, I refined the data structure. Instead of a vector of orders at each price level, I used another map: `std::map<double, std::map<long long, Order>>`. The outer map is price -> inner map. The inner map is order ID -> Order details.

```cpp
// Improved Order Book structure
struct Order {
    long long order_id;
    double price;
    int quantity;
    // direction (buy/sell), timestamp, etc.
    char side; // 'B' for bid, 'S' for sell
};

class OrderBook {
public:
    // Need methods to add, modify, cancel orders
    // and to get best bid/ask
    void add_order(const Order& order);
    void modify_order(long long order_id, int new_quantity);
    void cancel_order(long long order_id);

    double get_best_bid_price() const;
    double get_best_ask_price() const;

private:
    // std::map keys are ordered. For bids, we need descending order.
    // std::greater<double> makes the map sort keys in descending order.
    std::map<double, std::map<long long, Order>, std::greater<double>> bids_; // price -> order_id -> Order
    std::map<double, std::map<long long, Order>, std::less<double>> asks_; // price -> order_id -> Order

    // Need a way to quickly find an order by ID regardless of price level
    // This map stores a pointer/iterator to the order in bids_ or asks_
    // std::unordered_map is O(1) average lookup
    // Storing iterators is tricky if map structure changes - maybe store raw pointers or references?
    // Storing raw pointers is risky if the map reallocates/moves elements (std::map iterators/pointers are stable)
    // std::unordered_map<long long, Order*> order_id_lookup_; // order_id -> pointer to Order (in bids_ or asks_)
    // Update: Using C++17 node handles or careful iterator management is better than raw pointers.
    // For simplicity in this version, let's stick to finding via price first, or maybe a separate lookup map holding copies or indices?
    // Okay, let's try a map from ID to a structure indicating side and price, then look up in the main book.
     struct OrderLocation {
         char side; // 'B' or 'S'
         double price;
     };
     std::unordered_map<long long, OrderLocation> order_lookup_; // order_id -> location info
     // This lookup map helps find which price level and side to look in.
     // Still requires two map lookups (one in unordered_map, one in std::map) for modify/cancel.
     // Is this efficient enough? Need to test. Maybe storing direct iterators is better, but more complex.
     // Let's go with this approach for now, it feels safer than managing raw pointers into map nodes.
};
```

Using `std::map` for price levels provides the necessary sorting, and the inner `std::map` allows O(log N) (where N is number of price levels) + O(log M) (where M is number of orders at that level) access for modifications/cancellations. The `std::unordered_map` lookup is average O(1), reducing the first step to O(1). So, modify/cancel is roughly O(1) + O(log M). Getting the best bid/ask is O(log N) or O(1) if we cache it. This felt like a reasonable trade-off between complexity and performance for this stage. I used `std::greater` and `std::less` in the map definitions to ensure bids were descending and asks ascending, a small detail that took a moment to get right the first time.

Processing the simulated market data involved reading messages representing Limit Order Book (LOB) events: add order, modify order, cancel order. Each message needed to be parsed quickly and then dispatched to the appropriate method in the `OrderBook` class.

```cpp
// Simplified message processing loop concept
void process_market_data(OrderBook& book, const std::vector<MarketDataMessage>& messages) {
    for (const auto& msg : messages) {
        // Need high-resolution timer here to measure latency
        auto start_time = std::chrono::high_resolution_clock::now();

        if (msg.type == MessageType::ADD) {
            book.add_order(msg.order_details);
        } else if (msg.type == MessageType::MODIFY) {
            book.modify_order(msg.order_id, msg.new_quantity);
        } else if (msg.type == MessageType::CANCEL) {
            book.cancel_order(msg.order_id);
        }
        // ... handle other message types ...

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

        // Log or store the latency measurement
        // std::cout << "Processed message in " << duration.count() << " ns" << std::endl; // Too slow for logging every message!
        // Store durations in a vector and process statistics later.
        latency_measurements_.push_back(duration.count());
    }
}
```

Measuring latency accurately was a challenge. Using `std::chrono::high_resolution_clock` is the standard way in C++. The key is to minimize any work *around* the measured code path. Initially, I had `std::cout` inside the loop to print latency for every message. This completely skewed the results; writing to standard output is slow! I quickly realized I needed to collect measurements in memory (e.g., a `std::vector<long long>`) and calculate statistics (average, percentiles like P99) only after processing a large batch of messages.

Concurrency came into play when thinking about how this simulator would interact with external data sources or other components. While the core order book *processing* loop benefits from being single-threaded to avoid complex locking and context switches within the critical path, asynchronous operations like reading data from a network socket or logging results could be handled elsewhere. Boost Asio was used for this.

My experience with Boost Asio involved a bit of a learning curve. The asynchronous model, based on handlers (callbacks or coroutines), requires thinking differently about program flow. Setting up an `io_context` and posting work to it felt a bit abstract at first.

```cpp
// Basic Boost Asio concept for offloading work
boost::asio::io_context ioc;

// Function to process data (runs in a thread managed by Asio)
void process_data_chunk(std::vector<MarketDataMessage> chunk, OrderBook& book) {
    // ... process the chunk using book methods ...
    // std::cout << "Processed a chunk." << std::endl; // Again, avoid expensive ops in hot path!
}

// Main loop / Data ingestion side
// ... read data ...
// When a chunk of data is ready:
// boost::asio::post(ioc, std::bind(process_data_chunk, std::move(data_chunk), std::ref(my_order_book)));

// Need threads to run the io_context event loop
// std::vector<std::thread> workers;
// for(int i = 0; i < num_threads; ++i) {
//     workers.emplace_back([&]() { ioc.run(); });
// }
// ... workers.join() later ...
```
Initially, I struggled with understanding how Asio's handlers ensure thread safety if they interact with shared resources like the `OrderBook`. The simplest approach (and the one I settled on for the core book processing) was to guarantee that all handler calls that modify the `OrderBook` are *serialized* onto a single thread associated with the `io_context`. Asio's `strand` can help with this, but even simpler is just dedicating one thread to running the `io_context` that handles book updates, while other threads might feed data *into* a queue processed by that single thread. This avoids needing complex locks within the order book modification methods themselves, simplifying implementation and often improving performance by avoiding lock overhead and contention.

Optimization was an iterative process. After getting a basic version working, I used profiling tools (like `perf` on Linux or trying to integrate something like Google Perftools, though that added complexity) to identify bottlenecks. Unsurprisingly, map lookups and allocations were areas of concern.
Specific optimizations I explored or implemented:
1.  **Minimizing Allocations:** `std::map` involves dynamic memory allocation for each node. For very high throughput, these can add up. I briefly looked into custom allocators or object pools for `Order` objects and map nodes, but decided this was adding significant complexity beyond the core simulation logic for this project iteration. It's definitely a real-world HFT technique, though.
2.  **Cache Efficiency:** `std::map` nodes can be scattered in memory, potentially leading to cache misses. Data structures with better memory locality, like sorted arrays or custom tree implementations, could be faster. Again, I decided to stick with `std::map` for its correctness and reasonable performance, acknowledging this as a potential area for further optimization in a more advanced version.
3.  **Atomic Operations/Lock-Free?** I considered using atomic operations or lock-free data structures for parts of the system (like the queue feeding data to the processing thread) to avoid mutexes entirely. This is complex and error-prone in C++. While I read up on `std::atomic` and concurrent data structures (like `moodycamel::concurrent_queue` which looked interesting), I opted for simpler mutex-protected queues or Asio's strand-based serialization where possible to ensure correctness first. The "low latency" goal primarily focused on the core processing time *once* the message was received, not necessarily making the ingestion pipeline entirely lock-free.

The sub-microsecond latency target was challenging. Initially, processing a simple 'add' message might take a few microseconds due to map insertions. By carefully refining the data structures (moving to the nested map + lookup map approach), minimizing unnecessary operations, and ensuring the core processing loop was tight and avoided allocations, I was able to push typical processing times down significantly for individual messages. The specific latency achieved heavily depends on the structure of the simulated data (how many orders at each price, frequency of adds vs. cancels, etc.), but for realistic LOB update streams, getting consistent sub-microsecond latency required profiling and focusing on those O(log N) or O(1) costs. The P99 latency (99th percentile) was a much harder target than the average, revealing the impact of less frequent, more expensive operations or system noise.

Throughout this project, online resources were invaluable. Stack Overflow helped clarify subtle points about C++ containers, memory models, and getting Boost Asio handlers just right (especially understanding `io_context::run` and how to properly `post` or `dispatch` work). Reading documentation for Boost Asio multiple times was necessary; it's a powerful library, but has a steep learning curve. Articles on low-latency programming techniques often discussed the very data structure and concurrency challenges I was facing, confirming that the problems I was hitting were common in this domain.

Overall, building this simulator from scratch was a deep dive into performance-oriented C++ programming and the specific challenges of building a core financial component. It wasn't just about writing code; it was about understanding the trade-offs between different data structures under high throughput, the complexities of managing state with concurrency, and the absolute necessity of measurement to validate performance assumptions. While this version is a simulation and lacks many features of a real-world system (like complex order types, proper exchange protocols, persistent storage), it laid bare the fundamental engineering hurdles. Seeing the latency numbers drop as optimizations were applied was incredibly rewarding and solidified my understanding of what "low latency" really entails in practice. There's plenty more that could be added – persistent order IDs across updates, handling market orders, adding more detailed statistics, potentially exploring lock-free structures again with more confidence – but having a functional, performant core is a significant step.