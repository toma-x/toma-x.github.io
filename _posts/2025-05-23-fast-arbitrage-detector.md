---
layout: post
title: Real-Time Arbitrage Detector
---

## Journey into Sub-Millisecond Arbitrage: Building a Real-Time Detector in C++

This has been, without a doubt, the most challenging and rewarding project I've undertaken so far. The idea of capturing arbitrage opportunities – those fleeting moments where an asset is priced differently on separate exchanges – has always fascinated me. But I wanted to go beyond theory. I wanted to build something that could *actually* detect these in real-time, which meant diving deep into low-latency C++. The goal was ambitious: a system capable of sub-millisecond processing for streaming market data.

### The Initial Plunge: Why C++?

The decision to use C++ was almost immediate. While I'm comfortable with Python for scripting and data analysis, the "sub-millisecond" target screamed for something closer to the metal. I briefly considered Go for its concurrency features, but my existing C++ knowledge, however academic at that point, felt like a more solid foundation for tackling the raw performance needed. I knew the path would be steeper, especially concerning memory management and the intricacies of high-performance networking, but the potential payoff in speed was too significant to ignore. The core idea was to identify situations where, for a given pair like BTC/USD, the bid price on one exchange was higher than the ask price on another, even after accounting for (hypothetical) trading fees.

### Grappling with the Data Torrent: Market Feeds

The first real hurdle was getting the market data. Most exchanges offer WebSocket APIs for streaming order book updates and trades. I decided to focus on two hypothetical exchanges for this project, "AlphaEx" and "BetaTrade." My initial attempts to connect and manage these streams were, frankly, a bit of a mess.

I started looking into C++ WebSocket libraries. `Boost.Beast` seemed incredibly powerful, but its asynchronous model felt a bit daunting to jump into immediately, especially with the added complexity of OpenSSL for wss:// streams. I found a slightly simpler library, `websocketpp`, which could be configured to work with Asio (either standalone or Boost.Asio). Getting the initial handshake and message parsing right took days. The documentation for AlphaEx's WebSocket stream, for instance, was a bit sparse on the exact JSON structure for delta updates to the order book.

I remember vividly spending an entire Saturday debugging why my order book for ETH/USDC on AlphaEx was consistently missing levels or showing wildly incorrect prices. It turned out I was misinterpreting the sequence numbers in their update messages. Their system expected you to buffer messages if you received one with a sequence number that wasn't *exactly* one greater than the last, and I was just dropping them, thinking it was a network glitch. That was a hard-won lesson in reading API docs *very* carefully.

Here's a simplified snippet of how I started to handle incoming JSON messages for order book updates using `nlohmann/json`. This is after the WebSocket client part (which I won't detail here, it was mostly boilerplate from the `websocketpp` examples):

```cpp
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm> // For std::stod

// Structure to hold a single order book level
struct OrderBookLevel {
    double price;
    double quantity;
};

// Maps to store bids and asks: price -> quantity
// Bids are typically sorted descending, asks ascending
std::map<double, double, std::greater<double>> bids_alpha; // Price -> Quantity
std::map<double, double> asks_alpha; // Price -> Quantity

void process_market_data_alpha(const std::string& msg_payload) {
    try {
        auto json_data = nlohmann::json::parse(msg_payload);
        // Assuming JSON structure like: {"stream":"ethusdc@depth","data":{"e":"depthUpdate","E":1678886400000,"s":"ETHUSDC","U":100,"u":102,"b":[["2000.10","1.5"], ...],"a":[["2000.50","2.0"], ...]}}
        // This is a simplification; real delta processing is more complex
        if (json_data.contains("data") && json_data["data"].contains("b")) {
            for (const auto& bid_entry : json_data["data"]["b"]) {
                if (bid_entry.is_array() && bid_entry.size() == 2) {
                    double price = std::stod(bid_entry.get<std::string>());
                    double qty = std::stod(bid_entry.get<std::string>());
                    if (qty > 0.0) {
                        bids_alpha[price] = qty;
                    } else {
                        bids_alpha.erase(price); // Remove level if quantity is zero
                    }
                }
            }
        }
        if (json_data.contains("data") && json_data["data"].contains("a")) {
            for (const auto& ask_entry : json_data["data"]["a"]) {
                 if (ask_entry.is_array() && ask_entry.size() == 2) {
                    double price = std::stod(ask_entry.get<std::string>());
                    double qty = std::stod(ask_entry.get<std::string>());
                    if (qty > 0.0) {
                        asks_alpha[price] = qty;
                    } else {
                        asks_alpha.erase(price); // Remove level
                    }
                }
            }
        }
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        // In a real system, much more robust error handling here
    } catch (const std::exception& e) {
        std::cerr << "Error processing market data: " << e.what() << std::endl;
    }
}
```
My early versions of this parser were incredibly naive and prone to crashing if the JSON structure varied even slightly. Using `std::stod` directly from JSON string values without enough validation also led to some `std::invalid_argument` exceptions that were tricky to track down initially. I learned the hard way to be extremely defensive when parsing external data.

### The Core Logic: Spotting the Fleeting Chance

Once I had somewhat reliable data streams (or so I thought), the next step was the arbitrage detection logic itself. For cross-exchange arbitrage, the basic condition is: `Bid_Price_Exchange_A > Ask_Price_Exchange_B`.
Of course, it's not that simple. You have to account for:
1.  **Trading Fees:** Each exchange charges a fee.
2.  **Order Size:** The available quantity at the best bid/ask matters. An opportunity isn't real if you can only fill a tiny amount.
3.  **Latency:** The time it takes for your order to reach the exchange and get filled. My project focused on *detection* latency, not execution, but it's a critical real-world factor.

I maintained separate order books for each exchange. `std::map<double, double>` seemed like a reasonable choice initially, with price as the key and quantity as the value. For bids, I used `std::greater<double>` to keep them sorted high-to-low, and the default `std::less<double>` for asks (low-to-high). This made getting the best bid (map.begin()) and best ask (map.begin()) straightforward.

```cpp
// Assuming bids_alpha, asks_alpha, bids_beta, asks_beta are populated
// std::map<double, double, std::greater<double>> bids_beta;
// std::map<double, double> asks_beta;

const double FEE_RATE = 0.001; // 0.1% fee, simplified

void check_for_arbitrage() {
    if (bids_alpha.empty() || asks_beta.empty()) {
        return; // Not enough data
    }

    auto best_alpha_bid_iter = bids_alpha.begin();
    double best_alpha_bid_price = best_alpha_bid_iter->first;
    double best_alpha_bid_qty = best_alpha_bid_iter->second;

    auto best_beta_ask_iter = asks_beta.begin();
    double best_beta_ask_price = best_beta_ask_iter->first;
    double best_beta_ask_qty = best_beta_ask_iter->second;

    // Potential opportunity: Buy on Beta, Sell on Alpha
    // Price to buy on Beta: best_beta_ask_price * (1 + FEE_RATE)
    // Price to sell on Alpha: best_alpha_bid_price * (1 - FEE_RATE)
    double effective_buy_price_beta = best_beta_ask_price * (1.0 + FEE_RATE);
    double effective_sell_price_alpha = best_alpha_bid_price * (1.0 - FEE_RATE);

    if (effective_sell_price_alpha > effective_buy_price_beta) {
        double potential_profit_per_unit = effective_sell_price_alpha - effective_buy_price_beta;
        double tradeable_qty = std::min(best_alpha_bid_qty, best_beta_ask_qty);
        
        if (tradeable_qty > 0.0001) { // Some minimum quantity
             std::cout << "Arbitrage DETECTED! Pair: XYZ/USD\n"
                      << "  BUY on BetaTrade: " << tradeable_qty << " @ " << best_beta_ask_price << " (Effective: " << effective_buy_price_beta << ")\n"
                      << "  SELL on AlphaEx: " << tradeable_qty << " @ " << best_alpha_bid_price << " (Effective: " << effective_sell_price_alpha << ")\n"
                      << "  Potential Profit/Unit (before network latency/slippage): " << potential_profit_per_unit << std::endl;
            // In a real trading system, this would trigger order placement logic
        }
    }

    // Need to check the other direction too: Buy on Alpha, Sell on Beta
    if (asks_alpha.empty() || bids_beta.empty()) {
        return;
    }

    auto best_alpha_ask_iter = asks_alpha.begin();
    double best_alpha_ask_price = best_alpha_ask_iter->first;
    double best_alpha_ask_qty = best_alpha_ask_iter->second;

    auto best_beta_bid_iter = bids_beta.begin();
    double best_beta_bid_price = best_beta_bid_iter->first;
    double best_beta_bid_qty = best_beta_bid_iter->second;

    double effective_buy_price_alpha = best_alpha_ask_price * (1.0 + FEE_RATE);
    double effective_sell_price_beta = best_beta_bid_price * (1.0 - FEE_RATE);

    if (effective_sell_price_beta > effective_buy_price_alpha) {
        double potential_profit_per_unit = effective_sell_price_beta - effective_buy_price_alpha;
        double tradeable_qty = std::min(best_alpha_ask_qty, best_beta_bid_qty);

        if (tradeable_qty > 0.0001) {
             std::cout << "Arbitrage DETECTED! Pair: XYZ/USD\n"
                      << "  BUY on AlphaEx: " << tradeable_qty << " @ " << best_alpha_ask_price << " (Effective: " << effective_buy_price_alpha << ")\n"
                      << "  SELL on BetaTrade: " << tradeable_qty << " @ " << best_beta_bid_price << " (Effective: " << effective_sell_price_beta << ")\n"
                      << "  Potential Profit/Unit (before network latency/slippage): " << potential_profit_per_unit << std::endl;
        }
    }
}
```
This logic gets called every time there's an update to either order book that could affect the top-of-book. A major challenge here was "false positives" due to stale data from one feed. If one exchange feed lagged even slightly, it could look like an arbitrage opportunity existed when it didn't. Timestamping messages accurately on receipt and having a mechanism to consider data "stale" if not updated within a very short window became crucial. I didn't implement a perfect solution for this, but I started by discarding opportunities if the last update from either relevant book was older than, say, 50 milliseconds.

### The Quest for Sub-Millisecond Processing

This was where C++ was supposed to shine, and also where most of my optimization efforts (and frustrations) lay. "Sub-millisecond processing" means that from the moment a new piece of market data (e.g., a WebSocket message fragment containing an order book update) arrives at my program's network buffer, to the moment the arbitrage check is complete based on that new data, should take less than 1000 microseconds.

My initial profiler runs (just using `std::chrono` around critical sections) were... humbling. JSON parsing was a big one. `nlohmann/json` is convenient, but not the absolute fastest. For truly critical paths, some systems use custom parsers or schema-based ones like FlatBuffers or Cap'n Proto, but that felt like overkill and a huge learning curve for this project. I stuck with `nlohmann/json` but became very careful about minimizing re-parsing and only parsing what was necessary.

String operations and memory allocations were another area. Every `std::string` manipulation or dynamic allocation (like nodes in `std::map`) in the critical path adds overhead.
I spent a lot of time on StackOverflow looking for advice on optimizing `std::map` usage. One suggestion I found interesting was using `emplace_hint` if I had an idea where the new element should go, potentially speeding up insertion. For frequently updated top-of-book levels, `std::map`'s O(log N) was generally okay for the book depths I was handling (maybe a few hundred levels), but I knew that for extreme performance, specialized data structures are used. I didn't go down the route of building a custom B-tree or skip list, but I did ensure my maps didn't grow excessively large by pruning very deep levels that were unlikely to be part of an arbitrage.

The critical path was:
1.  Receive WebSocket message bytes.
2.  Parse JSON (this was the most expensive step initially).
3.  Update the internal `std::map` representing the order book.
4.  Call `check_for_arbitrage()`.

To measure, I used `std::chrono::high_resolution_clock` extensively:
```cpp
// Inside the WebSocket message handler, for example
auto t1 = std::chrono::high_resolution_clock::now();

// ... (parse message, update book_alpha or book_beta) ...
// process_market_data_alpha(payload); // or beta

check_for_arbitrage();

auto t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
// std::cout << "Processing time: " << duration << " microseconds\n";
// Log this duration, aggregate it, etc.
```
A breakthrough moment came when I refactored the JSON parsing. Instead of parsing the whole giant snapshot message every time, I focused on correctly applying the delta updates. This dramatically reduced the workload per message once the initial book was built. Another small win was ensuring all string comparisons for symbols or exchange names were done once, converting them to enums or integer IDs early on.

I considered using multiple threads, one per exchange connection, with a central thread for arbitrage checking. However, the synchronization overhead (mutexes, condition variables) for the shared order book data seemed like it could easily negate the benefits and introduce a lot of complexity and potential for deadlock. I read a few forum posts on r/cpp and r/algotrading where people debated this, and the consensus for very low latency often leaned towards carefully crafted single-threaded event loops for the core logic, or thread-per-core designs with careful data partitioning, which was beyond my scope. So, I stuck to a single-threaded model for the core data processing and arbitrage logic, focusing on making each step as fast as possible.

Achieving *consistent* sub-millisecond processing was tough. Average times might be low, but P99 latencies (the time under which 99% of operations complete) would sometimes spike due to various factors – a larger JSON message, a `std::map` rebalance, or even just OS scheduling jitter. I didn't fully conquer these spikes, but by focusing on reducing allocations and minimizing work in the hot path, I got the average processing time for typical incremental updates down into the 200-700 microsecond range on my development machine. This felt like a huge victory.

### Hurdles, Headaches, and "Ah-Ha!" Moments

There were countless moments of sheer frustration. Debugging race conditions is never fun, even in a mostly single-threaded hot path (interactions with network I/O can still surprise you). One particularly nasty bug involved floating-point precision. Comparing `double`s for equality is risky, and my initial fee calculations were sometimes off by a tiny fraction, causing missed opportunities or phantom ones. I had to introduce an epsilon for comparisons and be very careful with the order of operations. I found an old StackOverflow answer detailing the pitfalls of floating-point arithmetic in financial calculations that was a real eye-opener.

The documentation for BetaTrade's API had an example for calculating message signatures that was just plain wrong. It took me two days of pulling my hair out, comparing my generated signatures with their "expected" ones, before I tried a common variation I'd seen for other exchanges, and it suddenly worked. That was a moment of immense relief mixed with annoyance at the documentation.

A big "ah-ha!" moment was when I finally managed to get GDB working properly with the asynchronous WebSocket callbacks. Debugging those felt like trying to catch smoke initially. Being able to step through the code line-by-line as messages arrived was invaluable.

### Testing and First "Live" (Simulated) Run

Testing this system was a challenge in itself. You can't just "run" it against live exchanges without risking real money, especially with experimental code. I spent a lot of time building a simple simulator that would replay captured market data or generate synthetic data with known arbitrage opportunities. Seeing the detector correctly identify these pre-programmed scenarios was incredibly satisfying. The first time it spat out "Arbitrage DETECTED!" on a (simulated) feed that I hadn't explicitly designed an immediate opportunity for, but which arose from the dynamic interplay of two slightly out-of-sync synthetic feeds, was a real thrill.

### Reflections and Future Paths

This project taught me more about C++, networking, and market microstructures than any textbook could. The sheer difficulty of working at the low-latency edge, even for a relatively simple detection task, gave me a new appreciation for the engineering that goes into real high-frequency trading systems.

If I were to continue, I'd explore:
*   **More robust feed handling:** Dealing with disconnects, sequence gaps, and varying message formats more gracefully.
*   **Faster JSON parsing or alternative serialization:** Looking into libraries like `simdjson` or even moving to something like FlatBuffers for the internal representation if this were to scale.
*   **Execution logic:** The current system only detects. Adding logic to (hypothetically) place orders would be the next giant leap, with all its complexities around order management, slippage, and risk.
*   **More sophisticated clock synchronization:** Using NTP libraries to get more accurate local timestamps and better correlate data from different exchange servers.
*   **Kernel bypass networking:** Far beyond a student project, but understanding how systems achieve even lower latencies by avoiding the kernel's network stack (like `DPDK` or `Solarflare Onload`) was fascinating to read about.

This journey from a theoretical interest to a tangible, (almost) sub-millisecond system has been intense. There were many dead ends and moments of doubt, but pushing through them and seeing those microsecond counts drop was an incredible learning experience. It's one thing to read about low-latency techniques, it's another entirely to wrestle with them in your own code.