---
layout: post
title: Optimized Order Matching Engine
---

## Building an Optimized Order Matching Engine: A Deep Dive into <10µs Latency

This semester, I embarked on what turned out to be one of my most challenging and rewarding personal projects: engineering an order matching engine in C++. The goal was ambitious from the start: achieve a median latency of less than 10 microseconds per order, processing high-volume FIX protocol message streams. It’s been a journey filled with late nights, perplexing bugs, and moments of genuine excitement when things finally clicked.

### The Initial Spark and Why C++

The idea originated from a fascination with high-frequency trading systems and the incredible engineering challenges they represent. While building a full HFT system is a monumental task, I wanted to tackle a core component: the matching engine. The <10µs latency target immediately pointed towards C++. I briefly considered other languages – perhaps Java for its ecosystem, or even Python for rapid prototyping. However, the need for direct memory control, the overhead of garbage collection in Java, and Python's performance characteristics for this kind of ultra-low latency work made C++ the clear choice. I knew I'd be fighting for every nanosecond, and C++ gives you the tools to do that, albeit with a steeper learning curve and more potential pitfalls.

### Core Logic: The Order Book

The heart of any matching engine is its order book. I needed a way to store buy and sell orders efficiently, sorted by price and then by time. My first instinct was to use `std::map<Price, std::deque<Order*>>`, with `Price` being a custom type that correctly handles buy (descending) and sell (ascending) price levels. The `std::deque` would hold orders at the same price level in FIFO order.

```cpp
// A simplified representation of an order
struct Order {
    uint64_t orderId;
    bool isBuySide;
    double price;
    uint32_t quantity;
    // other fields like timestamp, userId etc.
};

// For buy orders, higher price has priority
std::map<double, std::deque<Order*>, std::greater<double>> buyOrders;
// For sell orders, lower price has priority
std::map<double, std::deque<Order*>, std::less<double>> sellOrders;
```

This worked fine for basic functionality. However, as I started thinking about performance under load, the O(log N) complexity for `std::map` operations (insert, delete, find best price) became a concern. For truly high-frequency scenarios, some engines use custom data structures, like arrays of linked lists indexed by price, but this adds complexity in terms of price level management. For this project, I decided to stick with `std::map` initially, focusing on other areas for optimization first, with the understanding that this could be a future bottleneck to revisit. The reasoning was that the N (number of price levels) might not be *that* large in many common scenarios, and the logarithmic factor might be acceptable if other parts of the system were fast enough.

### The Matching Algorithm Itself

The matching logic is fairly standard:
1.  When a new buy order arrives, check the `sellOrders` book. If the best sell price (lowest sell) is less than or equal to the buy order's price, a match can occur.
2.  When a new sell order arrives, check the `buyOrders` book. If the best buy price (highest buy) is greater than or equal to the sell order's price, a match can occur.
3.  Orders are filled based on price-time priority. If quantities match, both orders are fully filled. If not, one is fully filled, and the other is partially filled, with the remainder staying in the book (or becoming a new order if it was a market order consuming liquidity).

Implementing this, especially handling partial fills and ensuring atomicity of updates to the order book and order statuses, was trickier than it sounds. My initial matching function was a bit of a monolith.

```cpp
void Matcher::processOrder(Order* newOrder) {
    if (newOrder->isBuySide) {
        while (newOrder->quantity > 0 && !sellOrders.empty()) {
            auto& bestSellLevel = sellOrders.begin()->second; // deque of orders at best price
            Order* topSellOrder = bestSellLevel.front();

            if (newOrder->price >= topSellOrder->price) {
                uint32_t tradeQuantity = std::min(newOrder->quantity, topSellOrder->quantity);
                // ... logic to record trade, update order quantities ...
                newOrder->quantity -= tradeQuantity;
                topSellOrder->quantity -= tradeQuantity;

                if (topSellOrder->quantity == 0) {
                    bestSellLevel.pop_front();
                    // TODO: Memory management for topSellOrder
                }
                if (bestSellLevel.empty()) {
                    sellOrders.erase(sellOrders.begin());
                }
            } else {
                break; // No more matches possible
            }
        }
        if (newOrder->quantity > 0) {
            // Add remaining to buy book
            buyOrders[newOrder->price].push_back(newOrder);
        } else {
            // TODO: Memory management for newOrder if fully filled
        }
    } else {
        // Similar logic for sell order matching against buyOrders
    }
}
```
One early bug I spent hours on was related to iterator invalidation in the `std::map` when a price level was completely cleared. If `bestSellLevel.empty()` led to `sellOrders.erase(sellOrders.begin())`, I had to be careful if I was still holding onto iterators or references from that map entry. This is where `std::map::erase` returning the next valid iterator became very helpful, though my initial loop structure didn't use it, leading to some crashes during aggressive testing.

### The Bottleneck: Concurrent Access and the Leap to Lock-Free Queues

The real beast was concurrency. A matching engine needs to process incoming orders from multiple sources simultaneously and update the shared order book. My first attempt used `std::mutex` to protect access to the order books and the matching logic.

```cpp
std::mutex orderBookMutex;

void Matcher::processOrderSafe(Order* newOrder) {
    std::lock_guard<std::mutex> lock(orderBookMutex);
    processOrder(newOrder); // The function shown previously
}
```

Benchmarking this version quickly showed that `orderBookMutex` became a massive point of contention. Even with a handful of concurrent threads sending orders, latency shot up, and throughput plummeted. The <10µs target felt miles away. It was clear that traditional locks were not going to cut it.

This led me down the rabbit hole of lock-free programming. The goal was to allow multiple threads to enqueue orders and have a matcher thread (or threads) process them without coarse-grained locks. I needed a multi-producer, multi-consumer (MPMC) lock-free queue.

I spent a lot of time reading about lock-free queue designs, including Michael-Scott queues. I found a few discussions on StackOverflow (one particular answer explaining memory ordering for a simpler SPSC queue was a lifesaver for understanding `std::atomic_thread_fence` and `memory_order_acquire`/`release`). Implementing a robust, general-purpose MPMC lock-free queue is notoriously difficult due to issues like the ABA problem and ensuring correct memory reclamation.

Given the project's time constraints, writing a production-ready MPMC queue from scratch felt like a Ph.D. thesis topic on its own. I initially tried a simplified CAS (Compare-And-Swap) loop based approach for enqueue and dequeue.

```cpp
// Highly simplified conceptual snippet for a lock-free queue node and operation
template<typename T>
struct LFNode {
    T data;
    std::atomic<LFNode<T>*> next;
    LFNode(T val) : data(val), next(nullptr) {}
};

template<typename T>
class SimpleLockFreeQueue {
    std::atomic<LFNode<T>*> head;
    std::atomic<LFNode<T>*> tail;
    // Missing: proper memory reclamation, ABA protection, etc. This is very basic.

public:
    SimpleLockFreeQueue() {
        LFNode<T>* dummy = new LFNode<T>(T{}); // Dummy node
        head.store(dummy);
        tail.store(dummy);
    }

    void enqueue(T val) {
        LFNode<T>* newNode = new LFNode<T>(val);
        LFNode<T>* currentTail;
        while (true) {
            currentTail = tail.load(std::memory_order_acquire);
            LFNode<T>* nextNode = currentTail->next.load(std::memory_order_acquire);
            if (currentTail == tail.load(std::memory_order_acquire)) { // Check if tail changed
                if (nextNode == nullptr) {
                    if (currentTail->next.compare_exchange_weak(nextNode, newNode, std::memory_order_release, std::memory_order_relaxed)) {
                        break; // Successfully linked
                    }
                } else {
                    // Help another thread complete its operation
                    tail.compare_exchange_weak(currentTail, nextNode, std::memory_order_release, std::memory_order_relaxed);
                }
            }
        }
        tail.compare_exchange_weak(currentTail, newNode, std::memory_order_release, std::memory_order_relaxed); // Swing tail
    }
    // dequeue is similarly complex and error-prone
};
```
My initial custom lock-free queue attempts were buggy and often led to deadlocks or incorrect behavior under stress tests. Debugging these was a nightmare. GDB is tricky with highly concurrent atomic operations. I relied a lot on `printf` debugging (I know, I know) and very careful reasoning about memory ordering. After burning a significant amount of time, I decided to evaluate existing libraries. I looked into Boost.Lockfree, but eventually settled on `moodycamel::ConcurrentQueue` for its reputation for performance and relative ease of use for MPMC scenarios. This was a pragmatic decision; while I learned a ton from trying to build one, using a well-tested library allowed me to focus on the rest of the matching engine.

The shift involved creating an input queue for new orders and an output queue for trade confirmations or acknowledgements.

```cpp
#include "concurrentqueue.h" // moodycamel's queue

moodycamel::ConcurrentQueue<Order*> incomingOrdersQueue;
// moodycamel::ConcurrentQueue<ExecutionReport*> outgoingReportsQueue; // For results

// Worker thread(s) would enqueue orders:
// incomingOrdersQueue.enqueue(newOrder);

// Matching thread would dequeue and process:
// Order* orderToProcess;
// if (incomingOrdersQueue.try_dequeue(orderToProcess)) {
//     processOrder(orderToProcess); // The non-mutexed version
// }
```
This separation significantly improved concurrent performance, as the order submission path became largely non-blocking, and the matching logic (still single-threaded access to the books in my final iteration for simplicity, but fed by a lock-free queue) could churn through orders much faster. For even more performance, one could shard the order book (e.g., by instrument) and have multiple matching threads, each with its own set of books and fed by its own queue, but that was a step too far for this project.

### Ingesting FIX Protocol Messages

The project brief specified benchmarking against "high-volume FIX protocol message streams." The FIX (Financial Information eXchange) protocol is a standard in the financial industry. Messages are tag-value pairs, delimited by SOH (ASCII 0x01). Example: `8=FIX.4.2|9=75|35=D|49=CLIENT|56=SERVER|38=100|40=2|44=105.50|54=1|55=XYZ|60=20230101-12:00:00.000|10=012|`

Writing a full FIX engine is a massive undertaking. For this project, I needed to parse just enough to extract the critical information for an order: symbol, side (buy/sell), quantity, price, order type (I focused on Limit orders). I wrote a simple FIX message parser.

```cpp
Order* parseFIX(const std::string& fixMsg) {
    Order* order = new Order(); // Simplified, needs proper memory management
    // Super basic parser, not robust for production
    std::vector<std::string> KVPairs;
    std::string currentPair;
    for (char c : fixMsg) {
        if (c == '|') { // Using | as a delimiter for simplicity here, FIX SOH is \x01
            if (!currentPair.empty()) KVPairs.push_back(currentPair);
            currentPair.clear();
        } else {
            currentPair += c;
        }
    }
    if (!currentPair.empty()) KVPairs.push_back(currentPair);


    for (const auto& pair : KVPairs) {
        size_t eqPos = pair.find('=');
        if (eqPos == std::string::npos) continue;
        int tag = std::stoi(pair.substr(0, eqPos));
        std::string value = pair.substr(eqPos + 1);

        switch (tag) {
            case 54: // Side
                order->isBuySide = (value == "1");
                break;
            case 44: // Price
                order->price = std::stod(value);
                break;
            case 38: // Quantity
                order->quantity = std::stoul(value);
                break;
            // ... other tags like 55 (Symbol), 11 (ClOrdID) ...
            case 11: // ClOrdID - let's use this for orderId for simplicity
                order->orderId = std::stoull(value); // Assuming ClOrdID can be numeric for an ID
                break;
        }
    }
    // Basic validation omitted for brevity
    return order;
}
```
This parser is very naive. It doesn't handle repeating groups, checksums, or the myriad of other FIX features. It splits by `|` for readability in the example, but in reality, it should split by the SOH character. The main challenge was making this parsing step fast enough not to become the new bottleneck. `std::string` operations can be costly. For real HFT systems, FIX parsing is often heavily optimized, sometimes with custom generated parsers. For my tests, I pre-generated a stream of simplified FIX messages and focused the latency measurement on the `processOrder` call after parsing.

### Benchmarking: The Moment of Truth

To benchmark, I set up a test harness that would fire a large number of pre-parsed `Order` objects (or FIX messages if I wanted to include parsing time) into the `incomingOrdersQueue`. The matching thread would dequeue and process them. I used `std::chrono::high_resolution_clock` to measure the time taken from just before an order is ready to be processed by the core matching logic to just after all its resulting events (fills, acknowledgements) would theoretically be generated.

```cpp
// Inside the matching thread loop
auto startTime = std::chrono::high_resolution_clock::now();
matcher.processOrder(orderToProcess); // The core matching logic
auto endTime = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
latencies.push_back(duration);```

After processing, say, a million orders, I'd sort the `latencies` vector and find the median.
The first few runs with the lock-free queue (using MoodyCamel's) and the `std::map`-based order book were promising, but not quite there. I was seeing median latencies around 15-20µs.

Optimization iterations involved:
1.  **Memory Allocations:** `new Order()` for every incoming order and then `delete` upon fill or cancellation can be slow. I experimented with an object pool for `Order` objects. This gave a noticeable improvement, bringing latency down by a few microseconds. The idea is to pre-allocate a large number of `Order` objects and recycle them.
2.  **Compiler Optimizations:** Ensuring I was compiling with full optimizations (`-O3` or `-Ofast` with g++) and link-time optimization (LTO) was crucial. This is standard, but easy to forget during development cycles.
3.  **CPU Affinity:** Pinning the matching thread to a specific CPU core helped reduce cache thrashing and context switching, providing more consistent low latencies. This can be done with `pthread_setaffinity_np` on Linux.
4.  **Careful Data Structure Access:** Even with `std::map`, minimizing redundant lookups or unnecessary temporary object creations within the matching loop helped shave off nanoseconds.

One specific "aha!" moment came when profiling. I noticed that even with the object pool, the occasional dynamic allocation for `std::deque` within the `std::map` (when a new price level was created) could cause latency spikes. While not directly impacting the *median* much once the book was warm, it affected the tail latencies. For the median, the key was the efficiency of the lock-free queue and the core matching logic on already existing price levels.

After these refinements, and careful measurement of just the matching engine's core processing time (dequeuing from an internal queue, matching, enqueuing results), I was finally able to consistently hit the **<10µs median latency** mark. My measurements typically hovered around 7-9µs for the core matching operation on my test machine (an Intel Core i7-9700K).

### Struggles and Key Learnings

*   **Debugging Concurrency:** This was, by far, the hardest part. Lock-free code is subtle. Even using a library, understanding how it interacts with your system requires care. One issue I faced was ensuring the lifecycle management of `Order` objects passed through the queue was correct – who deletes it, and when? This led to some memory leaks and use-after-free bugs that were painful to track. `std::shared_ptr` could help but adds overhead, which I was trying to avoid in the hot path. I ended up using raw pointers with very strict ownership rules and relying on the object pool for recycling.
*   **The "It Works on My Machine" Syndrome:** Performance is highly dependent on the specific hardware (CPU, cache sizes, memory speed). What was <10µs on my desktop might be different elsewhere. This highlighted the importance of specifying the test environment.
*   **Premature Optimization vs. Informed Optimization:** While the mantra "premature optimization is the root of all evil" is often true, for a project with such a strict performance target, you have to think about performance from the beginning. However, it's crucial to *profile* before diving into complex optimizations. My initial focus on `std::map` might have been a premature de-optimization if I had tried to build a custom B-tree variant from day one without knowing if it was the true bottleneck.
*   **The Value of Good Libraries:** While I learned a lot struggling with my own lock-free queue, `moodycamel::ConcurrentQueue` saved me a huge amount of time and allowed me to reach my performance goals much faster and more reliably. Knowing when to build vs. when to use an existing, well-tested component is a key engineering skill.
*   **C++ Nuances:** This project forced me to delve deeper into C++ features like move semantics, perfect forwarding (when I was designing some template helper functions, not shown here), and the intricacies of `std::atomic` and memory models. There was a particular post on a forum about `std::memory_order_seq_cst` versus `acquire`/`release` that really helped clarify why sequential consistency is often overkill and can impede performance.

### Future Directions

There's always more that can be done.
*   **More Sophisticated Order Book:** Implementing a more cache-friendly order book structure could yield further improvements, especially for very deep books.
*   **Multi-Threaded Matching:** Sharding the order book by instrument and having multiple matcher threads.
*   **Resiliency:** Adding proper error handling, logging, and perhaps snapshotting/recovery mechanisms.
*   **Full FIX Compliance:** Using a proper FIX engine library like QuickFIX/C++ for parsing and session management if this were to become a more serious application.

This project was a fantastic learning experience. Pushing C++ to its limits to achieve ultra-low latency was incredibly challenging but also deeply satisfying. The combination of algorithmic thinking, low-level system details, and careful benchmarking made it a comprehensive dive into high-performance computing.