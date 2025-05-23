---
layout: post
title: Real-Time Arbitrage Pathfinder
---

## Real-Time Arbitrage Pathfinder: My Journey into C++ and Crypto Markets

This project, the "Real-Time Arbitrage Pathfinder," has been a pretty intense undertaking for the past few months. The idea of finding and exploiting arbitrage opportunities in cryptocurrency markets, all in real-time, seemed like a fascinating blend of programming challenge and financial puzzle. I knew from the start that if I wanted to even attempt "real-time," C++ was going to be the way to go for performance.

### Wrestling with Real-Time Data: WebSockets and Boost.Asio

The very first step was getting live market data. Most crypto exchanges offer WebSocket APIs for this, which is great for low-latency updates. I decided to try and integrate with Kraken's API first. Their documentation seemed comprehensive enough. My plan was to use Boost.Asio for networking, as it's pretty much the standard for high-performance I/O in C++. Specifically, I looked into Boost.Beast, which sits on top of Asio and provides WebSocket functionalities.

Getting the initial WebSocket handshake and connection established with Kraken wasn't trivial. The Boost.Beast examples were helpful, but tailoring them to Kraken's specific subscription messages for order book data took a lot of trial and error.

```cpp
// Part of my WebSocket client class
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/strand.hpp>
#include <nlohmann/json.hpp> // For parsing messages

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

// ... inside the class ...
private:
    net::io_context& ioc_;
    tcp::resolver resolver_;
    websocket::stream<beast::ssl_stream<tcp::socket>> ws_stream_; // SSL needed for wss://
    beast::flat_buffer buffer_;
    std::string host_;
    std::string port_;
    std::string sub_message_; // JSON subscription message

    // ...

    void on_connect(beast::error_code ec, tcp::resolver::results_type::endpoint_type ep) {
        if(ec) {
            // Log error: std::cerr << "Connect error: " << ec.message() << std::endl;
            return; // Need robust reconnection logic here eventually
        }

        // Timeout for SSL handshake
        beast::get_lowest_layer(ws_stream_).expires_after(std::chrono::seconds(5));
        
        ws_stream_.next_layer().async_handshake(
            ssl::stream_base::client,
            [this](beast::error_code ec) {
                on_ssl_handshake(ec);
            });
    }
```
The `on_connect` callback chaining into `on_ssl_handshake` and then `on_websocket_handshake`... it took a while to get the sequence right and handle errors at each stage. I remember one particular evening getting stuck on an SSL handshake timeout. Turns out, I hadn't properly configured the SSL context or wasn't calling the handshake on the correct layer of the stream. Stack Overflow had a few similar questions, but none matched exactly. It was a lot of stepping through the Asio source (or trying to) and poring over examples.

Parsing the incoming JSON messages was another consideration. I opted for `nlohmann/json`. It's incredibly easy to use, and while I was initially worried about its performance for high-frequency messages, for parsing subscription confirmations and initial order book snapshots, it seemed fine. For the constant stream of individual trade updates, I knew I'd need to be careful.

### Building the Market Graph

Once data started flowing, I needed a way to represent the market for arbitrage detection. The concept is to treat currencies as nodes in a graph and trading pairs as directed edges. For example, an ETH/BTC pair gives you an edge from ETH to BTC (with weight related to ask price) and an edge from BTC to ETH (with weight related to bid price).

I chose an adjacency list representation for the graph. Something like `std::vector<std::map<int, double>> adj_list_`, where the outer vector index is the source currency ID, the map key is the destination currency ID, and the value is the rate (or transformed rate for pathfinding). I used a map for the inner structure because not all currencies trade directly with each other, so it's sparse.

Updating these rates in real-time was the tricky part. The WebSocket client runs in its own thread (or rather, its `io_context` runs on a thread), and the pathfinding algorithm would ideally run separately or periodically. This immediately brought up concerns about thread safety when accessing and updating the graph. For now, I've been a bit simplistic, perhaps queuing updates and processing them in batches, but a proper lock-free structure or careful mutex usage is something I know I need to refine.

When an order book update comes in from the WebSocket (e.g., for BTC/USD), I parse it, find the best bid and ask, and then update the corresponding edges in my graph. For example:

```cpp
// Simplified conceptual update
// Assume 'parsed_data' is a json object from the WebSocket
void update_graph_edge(const json& parsed_data) {
    // Example: {"pair": "XBT/USD", "best_bid": "60000.1", "best_ask": "60000.5"}
    std::string pair_name = parsed_data.at("pair").get<std::string>();
    double bid_price = std::stod(parsed_data.at("best_bid").get<std::string>());
    double ask_price = std::stod(parsed_data.at("best_ask").get<std::string>());

    // currency_to_id_ is a map<string, int>
    // Assume XBT is base, USD is quote for XBT/USD
    // For XBT -> USD (selling XBT for USD), we use the bid price
    // For USD -> XBT (buying XBT with USD), we use the ask price (1 / ask_price)

    int base_id = currency_to_id_[extract_base(pair_name)]; //  Helper to get "XBT"
    int quote_id = currency_to_id_[extract_quote(pair_name)]; // Helper to get "USD"

    // For arbitrage, we often use -log(rate) as edge weights
    // Edge: Base -> Quote (e.g., BTC -> USD means selling BTC for USD at BID price)
    // Weight: -log(bid_price)
    // Edge: Quote -> Base (e.g., USD -> BTC means buying BTC with USD at ASK price)
    // Weight: -log(1.0 / ask_price) which is log(ask_price)

    // This lock is a bit coarse, needs more granular approach later
    std::lock_guard<std::mutex> lock(graph_mutex_); 
    
    if (bid_price > 0) { // Basic sanity check
        adj_list_[base_id][quote_id] = -std::log(bid_price);
    }
    if (ask_price > 0) {
        adj_list_[quote_id][base_id] = std::log(ask_price);
    }
    // Also need to store fees, slippage considerations... this is simplified
}
```
The use of `-log(rate)` is key for using Bellman-Ford to find profitable arbitrage cycles. If `rate1 * rate2 * rate3 > 1` (a profitable cycle), then `log(rate1) + log(rate2) + log(rate3) > 0`. If we use weights `-log(rate)`, then `-log(rate1) - log(rate2) - log(rate3) < 0`, meaning we are looking for a negative cycle in the graph.

### The Hunt: Bellman-Ford for Arbitrage Paths

With the graph structure in place and updating (somewhat) in real-time, the next step was to implement a pathfinding algorithm to detect these negative cycles. Bellman-Ford is well-suited for this because it can handle negative edge weights and detect negative cycles.

My initial Bellman-Ford implementation was fairly textbook:
```cpp
// Conceptual Bellman-Ford snippet
// V is number of currencies (nodes), E is number of pairs (edges)
// dist_to_source_ is std::vector<double>, parent_of_ is std::vector<int>

bool find_negative_cycle(int start_node_id) {
    dist_to_source_.assign(V, std::numeric_limits<double>::infinity());
    parent_of_.assign(V, -1);
    dist_to_source_[start_node_id] = 0.0;

    for (int i = 0; i < V - 1; ++i) { // Relax V-1 times
        for (int u = 0; u < V; ++u) {
            if (dist_to_source_[u] == std::numeric_limits<double>::infinity()) continue;

            // graph_ is the adjacency list we discussed
            for (const auto& edge : graph_[u]) { // Iterate over neighbors of u
                int v = edge.first; // Destination node
                double weight = edge.second; // -log(rate)
                if (dist_to_source_[u] + weight < dist_to_source_[v]) {
                    dist_to_source_[v] = dist_to_source_[u] + weight;
                    parent_of_[v] = u;
                }
            }
        }
    }

    // Check for negative cycles
    for (int u = 0; u < V; ++u) {
        if (dist_to_source_[u] == std::numeric_limits<double>::infinity()) continue;
        for (const auto& edge : graph_[u]) {
            int v = edge.first;
            double weight = edge.second;
            if (dist_to_source_[u] + weight < dist_to_source_[v] - 1e-9) { // Added epsilon for float comparison
                // Negative cycle detected!
                // Backtrack using parent_of_ to reconstruct the path
                // Store it, log it, etc.
                // last_node_in_cycle_ = v; // To start backtracking
                return true; 
            }
        }
    }
    return false;
}
```
One subtle point was handling floating-point comparisons. Instead of `dist_to_source_[u] + weight < dist_to_source_[v]`, I had to add a small epsilon for stability: `dist_to_source_[u] + weight < dist_to_source_[v] - 1e-9` to detect actual improvements. Debugging this was painful; opportunities that looked real on paper weren't being found because of tiny precision errors.

Bellman-Ford runs in O(VE) time, which can be slow if the graph is dense or has many vertices. For crypto markets, `V` (number of currencies) might be a few hundred or thousand, and `E` (number of active pairs) can be significant. Running this continuously is computationally expensive. I'm currently running it periodically, but I know this isn't ideal for "real-time" discovery. Future optimizations might involve looking at algorithms like SPFA (Shortest Path Faster Algorithm) if it proves more suitable, or more intelligent ways to trigger the search.

### Major Hurdles and Small Victories

The learning curve for Boost.Asio was, frankly, brutal. Asynchronous operations, callbacks, strands, error propagation – it's a complex beast. There were many times my `io_context.run()` would just exit, or a handler wouldn't get called, and debugging felt like groping in the dark. The Boost documentation is very thorough, but it assumes a lot of prior knowledge. I heavily relied on examples from the Beast repository and numerous Stack Overflow posts (especially one about `async_read` not completing, which turned out to be a lifetime issue with a buffer).

Latency is another monster. Even if my C++ code is fast, network latency to the exchange and back is a huge factor. By the time my algorithm identifies an opportunity based on data that's already milliseconds old, it might be gone. This is a fundamental challenge in arbitrage that my student project can't fully solve without co-location, but optimizing every part of my software path is still crucial.

One of the biggest "aha!" moments was when I finally saw a (theoretically) profitable arbitrage path printed to my console from live data, correctly reconstructed by Bellman-Ford. It was something like ETH -> BTC -> USDT -> ETH. Even if it was tiny and probably not executable due to fees or slippage (which my model is still basic on), seeing the logic work end-to-end was incredibly rewarding after weeks of debugging connections and algorithms.

### Current State and What's Next

Right now, the system can connect to Kraken's WebSocket API, parse order book data for a predefined set of currency pairs, update an internal graph representation, and periodically run Bellman-Ford to detect arbitrage opportunities. It logs these potential paths.

It's far from a production-ready system, of course. Error handling needs to be much more robust, especially for WebSocket disconnections and API errors. The model for transaction costs (fees, slippage) is very rudimentary and needs significant improvement. I also need to implement a more efficient way to manage and query order book depth, rather than just top-of-book. Currently, I'm just looking at `std::map` for bids and asks to get the best price, but a full order book representation would be better.

Future plans include:
*   Integrating with more exchanges to find cross-exchange arbitrage.
*   Improving the sophistication of the cost model.
*   Developing a proper backtesting framework. Right now, testing is very ad-hoc.
*   Exploring optimizations for the pathfinding or alternative algorithms.
*   Better concurrency management. The current locking is very basic.

This project has been an incredible learning experience in C++, low-latency networking, and algorithmic thinking. It’s definitely pushed my coding skills and my understanding of how complex systems are built. There's still a mountain to climb, but the view from here is already pretty interesting.