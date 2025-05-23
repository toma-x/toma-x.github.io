---
layout: post
title: DeFi Arbitrage Bot Simulation
---

## Diving Deep: Simulating DeFi Arbitrage with a Python and Rust Hybrid

For my latest personal project, I decided to venture into the world of Decentralized Finance (DeFi), specifically targeting arbitrage opportunities. The idea of bots automatically exploiting tiny price discrepancies across different exchanges has always fascinated me, so I set out to build a simulator for triangular arbitrage on mock Decentralized Exchange (DEX) order books. The goal wasn't just to see if it could be done, but to understand the performance implications and tackle them head-on.

### The Starting Point: Understanding Triangular Arbitrage

Triangular arbitrage, in a nutshell, is about exploiting price differences between three assets. For example, if you can trade Asset A for Asset B, Asset B for Asset C, and then Asset C back to Asset A, and end up with more of Asset A than you started with (after accounting for fees, though I simplified this initially), you've found an arbitrage opportunity. In the context of DEXs, this would mean looking at currency pairs like ETH/DAI, DAI/USDC, and USDC/ETH on simulated order books.

### Version 1: The All-Python Simulator

I kicked things off with Python because, well, it's familiar and great for rapid prototyping. My first task was to model the DEX order books. I opted for a simple structure: dictionaries where keys were price levels, and values were the total quantity available at that price.

```python
# mock_dex.py
from collections import defaultdict

class OrderBook:
    def __init__(self):
        self.bids = defaultdict(float) # Price -> Quantity
        self.asks = defaultdict(float) # Price -> Quantity

    def add_bid(self, price, quantity):
        self.bids[price] += quantity

    def add_ask(self, price, quantity):
        self.asks[price] += quantity

    # More methods for matching orders, etc. would go here
    # For the simulation, I mostly just needed to read these.

# Setting up mock exchanges
dex1_eth_dai = OrderBook()
dex2_dai_usdc = OrderBook()
dex3_usdc_eth = OrderBook()

# Populating with some initial dummy data
dex1_eth_dai.add_ask(1600.0, 10.0) # ETH/DAI: Ask 10 ETH @ 1600 DAI
dex1_eth_dai.add_bid(1599.0, 5.0)  # ETH/DAI: Bid 5 ETH @ 1599 DAI
# ... and so on for other books and pairs
```

With the order books in place, the core logic involved iterating through potential paths, calculating expected outcomes, and identifying profitable cycles. An event could be a new trade posted or an update to an order book. My initial Python loop for detecting an opportunity looked something like this (highly simplified for brevity):

```python
# main_simulator.py (initial Python version)
initial_eth = 1.0

def check_eth_dai_usdc_eth_opportunity(amount_in_eth, book_eth_dai, book_dai_usdc, book_usdc_eth):
    # Assume we are buying DAI with ETH, then USDC with DAI, then ETH with USDC
    
    # Trade 1: ETH -> DAI (buy DAI, so look at ETH/DAI asks)
    # This part got complicated fast with matching against order book depth
    # For now, let's imagine a function that gives best price for a quantity
    
    dai_received = amount_in_eth * get_effective_price(book_eth_dai.asks, amount_in_eth, 'buy') 
    # Simplified: assumes simple multiplication, real logic iterates through order book levels

    # Trade 2: DAI -> USDC
    usdc_received = dai_received * get_effective_price(book_dai_usdc.asks, dai_received, 'buy')

    # Trade 3: USDC -> ETH
    eth_received_final = usdc_received * get_effective_price(book_usdc_eth.asks, usdc_received, 'buy')
    
    if eth_received_final > amount_in_eth:
        print(f"Arbitrage found! Start ETH: {amount_in_eth}, End ETH: {eth_received_final}")
        return eth_received_final - amount_in_eth
    return 0.0

# The `get_effective_price` would need to iterate through sorted book_eth_dai.asks, 
# consume quantities at each price level until the `amount_in_eth` (or equivalent value) is filled,
# and then return the average price. This was a bit fiddly to get right.
```

The simulation would generate mock events (new orders, trades) and feed them into the system. For each event, especially one that significantly changed prices on a relevant order book, the arbitrage detection logic would run.

### The Inevitable Slowdown: Python Hits Its Limits

As I added more complexity to the order book interactions and increased the frequency of simulated events, the Python implementation started to show its age. Profiling with `cProfile` revealed that the arbitrage calculation loop, especially when trying to simulate matching against multiple levels of the order book for a given trade volume, was a major bottleneck. Each event, if it triggered a re-evaluation of potential arbitrage paths, was taking tens of milliseconds, sometimes more. In the fast-paced (even simulated) world of DeFi, this was glacial. I knew that real arbitrage bots operate on much, much tighter timeframes. My target was to get event processing and opportunity detection under 1 millisecond.

I remember staring at the profiler output and thinking there was no way pure Python was going to get me there, especially with the Global Interpreter Lock (GIL) being a potential concern if I ever wanted to explore true parallelism for, say, checking multiple arbitrage paths simultaneously (though that wasn't the immediate problem; raw computation speed was).

### The Quest for Speed: Enter Rust and PyO3

I considered a few options:
1.  **Cython:** Could translate parts of my Python code to C, which often gives good speedups. I'd used it briefly before, but the thought of managing the `.pyx` files and the build process for just a section of the code felt a bit like a patch rather than a robust solution.
2.  **Numba:** Uses a JIT compiler to speed up numerical Python code. This was tempting, as it often requires minimal code changes. However, my logic wasn't purely numerical; it involved a lot of dictionary lookups, sorting (implied, for order books), and conditional logic that I wasn't sure Numba would optimize as effectively as I needed.
3.  **Rewriting the critical path in a faster language:** This felt like the "proper" engineering solution, albeit the most time-consuming. Go and C++ were contenders, but Rust, with its promises of performance, memory safety without a garbage collector, and growing ecosystem, caught my eye. Specifically, PyO3, a library for creating Python extension modules in Rust, seemed like a clean way to bridge the two languages.

I'd been wanting an excuse to learn Rust properly, and this seemed like the perfect opportunity. The performance of C/C++, but with memory safety features that would hopefully save me from shooting myself in the foot with pointer errors. I found a few blog posts and the PyO3 documentation which, while a bit dense at times, gave me enough to get started. The "Calling Rust from Python" examples in the PyO3 guide were my starting point.

### Taking the Plunge: The Rust Rewrite

The first step was identifying the absolute critical path. This was the function that, given the current state of three order books and an initial amount of Asset A, would calculate the potential profit from a triangular arbitrage.

My Rust data structures for order books mirrored the Python ones, but using `BTreeMap` for sorted keys (prices) which is crucial for order books.

```rust
// src/lib.rs (Rust part)
use pyo3::prelude::*;
use std::collections::BTreeMap;

// A simplified representation for an order book side
// In a real scenario, prices would likely be represented differently (e.g., fixed-point)
// For simplicity, f64 is used here.
type OrderBookSide = BTreeMap<u64, f64>; // Price (as u64, e.g., scaled integer) -> Quantity

#[pyfunction]
fn calculate_triangular_arbitrage_rust(
    initial_amount_asset1: f64,
    // Representing order books as tuples of (bids, asks)
    // Bids: (price, quantity), Asks: (price, quantity)
    // For an ETH -> DAI -> USDC -> ETH loop:
    // book1: ETH/DAI (we sell ETH for DAI, so we look at bids for ETH, or effectively asks for DAI if pair is DAI/ETH)
    // More accurately, if pair is ETH/DAI, and we sell ETH, we hit bids.
    // If pair is ETH/DAI, and we buy ETH, we hit asks.
    // Let's define clearly:
    // book1_asks: ETH/DAI asks (price of ETH in DAI, quantity of ETH) - used if buying ETH with DAI
    // book1_bids: ETH/DAI bids (price of ETH in DAI, quantity of ETH) - used if selling ETH for DAI
    
    // For ETH -> DAI (selling ETH for DAI): use ETH/DAI bids
    // For DAI -> USDC (selling DAI for USDC): use DAI/USDC bids
    // For USDC -> ETH (selling USDC for ETH): use USDC/ETH bids
    // This is one interpretation. Or, if you always "cross the spread":
    // ETH -> DAI: buy DAI with ETH (use asks of ETH/DAI, where price is DAI per ETH)
    // DAI -> USDC: buy USDC with DAI (use asks of DAI/USDC, where price is USDC per DAI)
    // USDC -> ETH: buy ETH with USDC (use asks of USDC/ETH, where price is ETH per USDC)
    // The latter seems more standard for arbitrage calc.

    // Let's use asks for "buying the base asset"
    // path: A -> B -> C -> A
    // Trade 1: Start with A, buy B. Look at A/B asks (price of B in A).
    // No, if I have ETH and want DAI, I sell ETH. This means someone *bids* for my ETH.
    // Or, I am buying DAI with my ETH. The price is ETH/DAI. Ask price is what sellers want for ETH. Bid price is what buyers offer for ETH.
    // If I sell 1 ETH for DAI, I look for the highest bid price on the ETH/DAI book.
    // amount_dai = amount_eth * bid_price_eth_dai
    
    // Let's simplify and assume these are the prices you get.
    // A real implementation needs to walk the book.
    
    // This function will receive sorted lists of (price, quantity) tuples for the relevant side of each book.
    // For example, if selling asset1 for asset2, we'd use the bids for asset1/asset2.
    // If buying asset2 with asset1, we'd use the asks for asset1/asset2.
    
    // This gets very confusing quickly without very strict definitions of what each "book" represents.
    // Let's assume the Python side pre-selects the correct side (bids or asks) and sorts them.
    // book_1_levels: Vec<(f64, f64)>, // (price, quantity) for Asset1 -> Asset2
    // book_2_levels: Vec<(f64, f64)>, // for Asset2 -> Asset3
    // book_3_levels: Vec<(f64, f64)>, // for Asset3 -> Asset1
    
    // Let's refine. Assume prices are "how much of quote asset for 1 unit of base asset".
    // ETH/DAI: price is DAI per ETH.
    // Start with 1 ETH.
    // 1. Sell ETH for DAI: Use ETH/DAI bids. Price: DAI_per_ETH. Amount DAI = amount_ETH * price.
    //    book_eth_dai_bids: Vec<(f64 price_dai_per_eth, f64 quantity_eth)>
    // 2. Sell DAI for USDC: Use DAI/USDC bids. Price: USDC_per_DAI. Amount USDC = amount_DAI * price.
    //    book_dai_usdc_bids: Vec<(f64 price_usdc_per_dai, f64 quantity_dai)>
    // 3. Sell USDC for ETH: Use USDC/ETH bids. Price: ETH_per_USDC. Amount ETH = amount_USDC * price.
    //    book_usdc_eth_bids: Vec<(f64 price_eth_per_usdc, f64 quantity_usdc)>

    book1_bids_eth_dai: Vec<(f64, f64)>, // (price_dai_per_eth, quantity_eth_at_price)
    book2_bids_dai_usdc: Vec<(f64, f64)>, // (price_usdc_per_dai, quantity_dai_at_price)
    book3_bids_usdc_eth: Vec<(f64, f64)>  // (price_eth_per_usdc, quantity_usdc_at_price)

) -> PyResult<f64> {
    let mut current_amount_asset1 = initial_amount_asset1;
    let mut amount_asset2 = 0.0;
    let mut amount_asset3 = 0.0;
    let mut final_amount_asset1 = 0.0;

    // Trade 1: Asset1 (ETH) -> Asset2 (DAI)
    // We are selling asset1, so we look at bids (buyers for asset1)
    // Prices should be sorted descending for bids
    let mut amount_to_trade1 = current_amount_asset1;
    for (price, quantity) in book1_bids_eth_dai.iter().rev() { // bids: high to low
        if amount_to_trade1 == 0.0 { break; }
        let tradeable_qty = if amount_to_trade1 > *quantity { *quantity } else { amount_to_trade1 };
        amount_asset2 += tradeable_qty * price;
        amount_to_trade1 -= tradeable_qty;
        if amount_to_trade1 <= 1e-9 { break; } // Epsilon for float comparison
    }
    if amount_to_trade1 > 1e-9 { return Ok(0.0); } // Not enough liquidity

    // Trade 2: Asset2 (DAI) -> Asset3 (USDC)
    // We are selling asset2
    let mut amount_to_trade2 = amount_asset2;
    for (price, quantity) in book2_bids_dai_usdc.iter().rev() { // bids: high to low
        if amount_to_trade2 == 0.0 { break; }
        let tradeable_qty_asset2 = if amount_to_trade2 > *quantity { *quantity } else { amount_to_trade2 };
        // quantity here is quantity of asset2 (DAI)
        amount_asset3 += tradeable_qty_asset2 * price; // price is asset3/asset2
        amount_to_trade2 -= tradeable_qty_asset2;
        if amount_to_trade2 <= 1e-9 { break; }
    }
    if amount_to_trade2 > 1e-9 { return Ok(0.0); } // Not enough liquidity for asset2

    // Trade 3: Asset3 (USDC) -> Asset1 (ETH)
    // We are selling asset3
    let mut amount_to_trade3 = amount_asset3;
    for (price, quantity) in book3_bids_usdc_eth.iter().rev() { // bids: high to low
        if amount_to_trade3 == 0.0 { break; }
        let tradeable_qty_asset3 = if amount_to_trade3 > *quantity { *quantity } else { amount_to_trade3 };
        // quantity here is quantity of asset3 (USDC)
        final_amount_asset1 += tradeable_qty_asset3 * price; // price is asset1/asset3
        amount_to_trade3 -= tradeable_qty_asset3;
        if amount_to_trade3 <= 1e-9 { break; }
    }
    if amount_to_trade3 > 1e-9 { return Ok(0.0); } // Not enough liquidity for asset3

    if final_amount_asset1 > initial_amount_asset1 {
        Ok(final_amount_asset1 - initial_amount_asset1)
    } else {
        Ok(0.0)
    }
}

#[pymodule]
fn arbitrage_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_triangular_arbitrage_rust, m)?)?;
    Ok(())
}
```
One of the trickiest parts was ensuring the data types matched up and were handled correctly by PyO3, especially converting Python lists of tuples (representing order book levels) into Rust `Vec<(f64, f64)>`. The `iter().rev()` on bids assumes bids are stored in ascending price order from the Python side, which I had to ensure. Or, the Python side should pre-sort them descendingly before passing. I also spent a good while wrestling with what "price" meant in each pair (e.g., ETH/DAI vs DAI/ETH) and ensuring the multiplication was correct to get the amount of the target asset. Small logical errors here would completely invalidate the simulation. My initial Rust code for walking the order book was also a bit naive and I iterated a few times on how to correctly consume liquidity.

Building this with `maturin develop` was surprisingly smooth once I got the `Cargo.toml` and PyO3 boilerplate set up.

The Python side then became much simpler for the core calculation:
```python
# main_simulator.py (with Rust integration)
# import the Rust module, name depends on Cargo.toml
# Assuming the lib name in Cargo.toml is "arbitrage_engine"
import arbitrage_engine 

# ... OrderBook class and population ...

# Prepare data for Rust:
# The Rust function expects bids sorted ascendingly for its .iter().rev() logic
# Or, if Rust expects them descending, Python sorts descendingly.
# Let's assume Python sorts them ready for consumption.
# For selling ETH (asset1) for DAI (asset2), we need ETH/DAI bids, sorted high to low.
# book_eth_dai.bids is a dict {price: quantity}. Convert to sorted list of tuples.
eth_dai_bids_sorted = sorted(dex1_eth_dai.bids.items(), key=lambda x: x, reverse=True)

# For selling DAI (asset2) for USDC (asset3), we need DAI/USDC bids.
dai_usdc_bids_sorted = sorted(dex2_dai_usdc.bids.items(), key=lambda x: x, reverse=True)

# For selling USDC (asset3) for ETH (asset1), we need USDC/ETH bids.
usdc_eth_bids_sorted = sorted(dex3_usdc_eth.bids.items(), key=lambda x: x, reverse=True)

initial_eth_amount = 1.0
profit = arbitrage_engine.calculate_triangular_arbitrage_rust(
    initial_eth_amount,
    eth_dai_bids_sorted,
    dai_usdc_bids_sorted,
    usdc_eth_bids_sorted
)

if profit > 0:
    print(f"Rust-powered arbitrage found! Profit: {profit} ETH")

```
The first time I ran this and saw the "Rust-powered arbitrage found!" message, with a calculation that matched my (slower) Python version, was a huge relief. There was a lot of `println!` debugging in the Rust code and careful checking of intermediate values passed back and forth.

### The "It Works... But Is It Right?" Moment

A particularly nasty bug I encountered was related to floating-point precision and how I was accumulating amounts. Initially, in my Rust code, small discrepancies were leading to situations where a path looked profitable by a tiny fraction, but it was just noise, or I wasn't fully consuming an order level correctly. This involved meticulous step-through debugging. Another issue was the definition of "price". If `book1_eth_dai` has a price of 1600, does that mean 1 ETH = 1600 DAI, or 1 DAI = 1600 ETH? Standardizing this across Python and Rust, and in my own head, was critical. For DEX pairs, it's typically `BASE/QUOTE`, so price is "how much QUOTE for 1 BASE". Getting this wrong in one leg of the triangle throws everything off. I remember spending an entire afternoon just drawing diagrams and re-confirming which side of which book to hit and how to calculate the resulting amount. I found a few forum discussions on how people structure their order book data for HFT systems that gave me some clarity.

### Smashing the Latency Target

With the core arbitrage logic now in Rust, the change was dramatic. I used `time.perf_counter()` in Python before and after the call to the Rust function.

```python
import time

# ...
start_time = time.perf_counter()
profit = arbitrage_engine.calculate_triangular_arbitrage_rust(
    initial_eth_amount,
    eth_dai_bids_sorted,
    dai_usdc_bids_sorted,
    usdc_eth_bids_sorted
)
end_time = time.perf_counter()
duration_ns = (end_time - start_time) * 1_000_000_000 # nanoseconds
duration_ms = duration_ns / 1_000_000 # milliseconds
print(f"Rust calculation took: {duration_ms:.4f} ms")

if profit > 0:
     print(f"Potential profit: {profit}")
# ...
```
The calculation time for the core logic dropped from many milliseconds to well under one millisecond – often in the range of a few hundred microseconds, depending on the depth of the order book slices being processed. This was exactly what I was hoping for. The Python overhead for preparing the data (sorting lists of tuples from the dictionaries) and handling the PyO3 call added a little, but the total event processing latency was now comfortably within my desired sub-millisecond range.

### Simulating the DEX Environment

For the mock DEX environment, I had a simple event loop. New orders would be generated randomly (within certain price bands around a "true" market price) and added to the order books. My simulator didn't fully model gas fees or transaction execution times on a blockchain, as that would add another layer of complexity. The focus was primarily on the speed of *detecting* the opportunity given fresh order book data. I also simplified slippage by having the Rust function walk the provided book levels; a more advanced system might predict slippage more dynamically.

### Reflections and Learnings

This project was a fantastic learning experience.
*   **Python is great for orchestration, Rust is a beast for performance:** The combination felt very powerful. Python handled the overall simulation flow, event generation, and data management, while Rust crunched the numbers for the time-critical part.
*   **PyO3 is effective:** While there's a learning curve, PyO3 provides a relatively clean interface between Python and Rust. The documentation is key.
*   **Details matter immensely:** Especially in finance-related calculations, small errors in logic (like which side of the book to use, or how price is defined) can lead to completely wrong results. I learned to be extremely methodical.
*   **Profiling is essential:** Don't guess where bottlenecks are. `cProfile` pointed me in the right direction.
*   **Understanding the problem domain:** Even for a simulation, understanding the basics of market microstructure (order books, bid-ask spread, price impact) was crucial.

The simulation did show that, under the right (simplified) conditions, arbitrage opportunities would appear, and the Rust-powered core was fast enough to identify them. Future extensions could involve more realistic market dynamics, multiple exchanges, or even exploring more complex arbitrage paths. But for now, achieving that <1ms latency felt like a significant milestone. It was definitely a challenging but rewarding project, and a great dive into both DeFi concepts and practical performance engineering.