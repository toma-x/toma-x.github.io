---
layout: post
title: Low-Latency Order Book Simulator
---

Low-Latency Order Book Simulator

Coming off a semester focused on machine learning, I wanted to tackle a project in a completely different domain: low-latency systems, specifically within finance. The goal I set was to build a functional order book simulator in C++ that could process market events – limit orders and trades – extremely quickly, aiming for processing times under a microsecond per event. C++ felt like the obvious choice for this given the performance requirements.

The core challenge was managing the collection of outstanding limit orders (the "book") in a way that allowed for very fast insertion, deletion, and retrieval, which are necessary for adding new orders, canceling existing ones, and executing trades. My initial thought was to use something standard like `std::map` in the C++ STL to store price levels, with each price level holding a collection of orders. A `std::map<double, std::vector<Order>>` seemed intuitive at first, storing bids and asks keyed by price. However, almost immediately, concerns about using `double` for price keys due to potential floating-point inaccuracies and the performance implications of `std::vector` insertions and deletions, especially in the middle, became clear.

I spent a couple of days researching common approaches for order book implementations. Financial systems often use integer-based price representations (scaling pennies or other minimum price increments) to avoid floating-point issues. For the order lists at each price level, `std::list` seemed more appropriate than `std::vector` because order cancellations (deletions) and additions happen frequently, and `std::list` provides constant time insertion/deletion given an iterator, unlike `std::vector`. So, the basic structure evolved to something closer to `std::map<int, std::list<Order*>>` (using pointers to avoid copying large `Order` objects, though managing memory became another concern). I debated implementing a more specialized tree structure or using arrays if the price range was constrained, but decided that `std::map` offered a reasonable balance of performance for lookups and development time compared to writing a custom tree from scratch. This decision probably saved me a week or two of complex coding but meant I had to accept `std::map`'s logarithmic complexity for price level lookups instead of potential constant time with an array-based approach if feasible.

Implementing the core logic for handling incoming events was the next major step. A 'Limit Order' event required finding the correct price level in the map and inserting the new order while maintaining time priority *within* that level. A 'Trade' event (representing a market order or an aggressive limit order hitting resting liquidity) was much more complex – it involved iterating through the opposing side of the book (asks for a buy trade, bids for a sell trade), matching orders, updating quantities, and removing fully filled orders. Getting the iteration and removal logic right while ensuring no orders were skipped or double-counted was tricky. I remember spending a solid evening and the following morning just debugging the trade execution loop, often encountering issues where iterating and erasing from `std::list` simultaneously would invalidate iterators if not handled carefully according to the standard. I had to frequently consult the C++ documentation and Stack Overflow posts on safe `std::list` manipulation during iteration and deletion.

```cpp
// Snippet showing simplified trade execution logic (partially bug-ridden initial draft)
void OrderBook::process_trade(TradeEvent& event) {
    // Assuming event contains side, quantity, maybe aggressor order ID
    // And we've already determined which side of the book to cross
    auto& book_side = (event.side == Side::BUY) ? asks_ : bids_; // asks_ is map<int, list<Order*>>

    int remaining_qty = event.quantity;

    // Need to iterate through price levels starting from the best price
    // Getting the "best price" iterator depends on the side (begin for asks, rbegin for bids map)
    auto it = (event.side == Side::BUY) ? book_side.begin() : book_side.rbegin();

    while (remaining_qty > 0 && it != book_side.end() && it != book_side.rend()) { // Need rend() check for reverse iterator
        auto& orders_at_level = it->second; // list<Order*>

        // This nested loop for orders at a level was where I had issues
        auto order_it = orders_at_level.begin();
        while (remaining_qty > 0 && order_it != orders_at_level.end()) {
            Order* resting_order = *order_it;
            int fill_qty = std::min(remaining_qty, resting_order->get_remaining_quantity());

            // Execute fill (update quantities, generate fill events - omitted for snippet clarity)
            resting_order->reduce_quantity(fill_qty);
            remaining_qty -= fill_qty;

            if (resting_order->get_remaining_quantity() == 0) {
                // Order fully filled, need to remove it from the list
                // This erase invalidates order_it! Need to get the next iterator BEFORE erasing.
                order_it = orders_at_level.erase(order_it); // This was the common mistake
                // Corrected: order_it = orders_at_level.erase(order_it); // This returns the next iterator
            } else {
                ++order_it; // Only increment if not erased
            }
        }

        // Move to the next price level
        // This also needed careful handling for reverse iterators
        if (orders_at_level.empty()) {
             // If level is now empty, should consider removing the map entry? Or just let it be?
             // Removing saves memory/lookup time but adds complexity. Chose to keep empty lists for simplicity initially.
        }

        if (event.side == Side::BUY) {
           ++it; // For forward iterator (asks)
        } else {
           ++it; // For reverse iterator (bids), confusingly also increments
        }
    }
}
```
The simulated real-time feed parsing was relatively simpler. I designed a text-based format for market events (e.g., `L,BUY,100,50.25,OrderID123` for a limit order, `T,BUY,200,OrderID456` for a trade) and wrote a simple loop to read lines, parse the comma-separated values, and construct the appropriate internal event object to pass to the order book processing logic. This parsing part took about a day to get robust enough to handle different event types and basic data validation.

Benchmarking was critical to know if the "sub-microsecond" goal was being met. I used `std::chrono::high_resolution_clock` to time the processing of individual events within the main loop. I ran simulations with millions of generated events, recording the time taken for the `process_event` function calls. Initially, the average times were in the low microseconds, but not consistently below one. Optimizations involved minimizing dynamic memory allocations during event processing, passing objects by reference or pointer where possible instead of copying, and ensuring the data structures weren't causing unexpected overheads. For example, I found that creating and destroying temporary strings during parsing was adding overhead, which I fixed by parsing directly into pre-allocated buffers or using faster parsing techniques. Getting the average down required several iterations over about two weeks, running benchmarks, identifying hotspots with basic timing, and tweaking the code. Reaching a consistent average below 1 microsecond for typical limit order and trade events felt like a significant milestone.

Overall, this project was a deep dive into performance-critical C++ and the specific challenges of building a system like an order book. It wasn't just about writing code; it was about understanding the performance characteristics of data structures, the costs of different operations, and how seemingly small details can have a large impact at high frequency. Debugging concurrency issues was avoided by keeping it single-threaded, but memory management with raw pointers (as in my `list<Order*>`) required extra care to prevent leaks, which I admittedly struggled with initially until I added more explicit cleanup logic. It was a demanding but incredibly rewarding experience that taught me a lot about building efficient systems from the ground up.