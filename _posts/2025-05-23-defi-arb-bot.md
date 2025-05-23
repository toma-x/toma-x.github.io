---
layout: post
title: DeFi Arbitrage Bot
---

## DeFi Arbitrage Bot: A Journey into Web3.py, Rust, and Flash Loans

It's been a while since my last post, mostly because I've been deep in the weeds with a project that’s been both incredibly frustrating and hugely rewarding: building a real-time arbitrage bot for Decentralized Exchanges (DEXs). This wasn't just about stringing together a few API calls; I wanted to explore flash loan strategies and see if I could actually make something that could, in theory, identify and execute profitable arbitrage opportunities across EVM-compatible chains. The stack? Web3.py for its Ethereum interaction capabilities and Rust for the performance-critical parts.

### The Allure of Arbitrage and the "Magic" of Flash Loans

The idea of arbitrage in DeFi has always fascinated me. Price discrepancies between DEXs, even fleeting ones, seemed like a complex puzzle to solve. When I first read about flash loans – the ability to borrow substantial amounts of capital with no upfront collateral, provided the loan is repaid within the same transaction block – it felt like unlocking a new level in financial engineering. The main resources I leaned on were the Aave V2 documentation and a few articles on platforms like Medium explaining the execution flow. My initial thought was, "This is too good to be true," but the atomicity requirement made it clear it was a feature, not a bug, of blockchain transactions.

### Choosing the Tools: Web3.py and Rust

My go-to for blockchain interaction has usually been Python, largely due to the ease of use of **Web3.py**. For quick prototyping and managing smart contract interactions, it felt like the natural choice. I was already familiar with its API for fetching data, formatting transactions, and interacting with contracts.

However, I knew that identifying arbitrage opportunities in real-time, especially if I wanted to scan multiple pairs across multiple DEXs, would require serious speed. Python, for all its conveniences, wasn't going to cut it for the core detection logic. This led me to **Rust**. I'd been meaning to get deeper into Rust for its performance and memory safety promises. This project felt like the perfect excuse. The learning curve was steep, no doubt. Concepts like ownership and the borrow checker took a good while to click. I remember spending hours on the Rust forums and StackOverflow, particularly trying to understand lifetimes in the context of data structures I was using for price feeds.

For the target environment, I decided to focus on EVM-compatible chains. Initially, I thought about Ethereum mainnet, but the gas fees quickly made me reconsider for a student project. So, I started looking at Polygon and then later, Avalanche, as more viable testing and potential deployment grounds.

### Weaving It Together: Architecture and Early Hurdles

The basic architecture I landed on involved a Python process managing the overall orchestration, using Web3.py to subscribe to new block events and fetch prices from DEXs like UniswapV2 and Sushiswap. I focused on these first because their router and pair contract interfaces are quite similar, which simplified the initial development.

The Python script would then feed this pricing data to a Rust module. This was my first big integration challenge. I explored a few options, including simple IPC via stdin/stdout, but eventually settled on using **PyO3**. This allowed me to compile my Rust logic into a Python-loadable module. Setting up the `Cargo.toml` and the `lib.rs` file with the `#[pymodule]` and `#[pyfunction]` macros was finicky. I recall a lot of compilation errors related to type conversions between Python objects and Rust structs until I found some clear examples in the PyO3 documentation.

My Rust module was responsible for the heavy lifting: maintaining an internal representation of market states and rapidly calculating potential arbitrage paths. For example, if I had prices for ETH/DAI on SushiSwap and ETH/USDC and DAI/USDC on Uniswap, the Rust code would check for triangular arbitrage opportunities.

Here's a very simplified snippet of how I started fetching pair reserves using Web3.py. I was initially just printing them to make sure I was getting sensible data.

```python
from web3 import Web3

# Assuming w3 is an initialized Web3 instance connected to a node
# and I have the pair_contract_address and pair_contract_abi

def get_reserves(w3_instance, pair_address, pair_abi):
    pair_contract = w3_instance.eth.contract(address=pair_address, abi=pair_abi)
    try:
        reserves = pair_contract.functions.getReserves().call()
        # reserves is reserve0, reserves is reserve1
        return reserves, reserves
    except Exception as e:
        print(f"Error fetching reserves for {pair_address}: {e}")
        return None, None

# Later on, I would feed these reserves into my Rust module
```
My Rust code for processing this was, at first, very basic. I was just trying to get the data flow right.

```rust
// In my Rust lib.rs, after PyO3 setup

struct TokenPair {
    token0_address: String,
    token1_address: String,
    reserve0: u128,
    reserve1: u128,
    dex_name: String,
}

// This function would be called from Python with data for multiple pairs
fn find_arbitrage_opportunities_rs(pairs_data: Vec<TokenPair>) -> Option<String> {
    // Simplified logic: just an example of iterating
    // Real logic involved graph traversal or more complex comparisons
    if pairs_data.len() < 2 {
        return None;
    }

    // Imagine complex calculations here to find an arbitrage path
    // For instance, comparing (pairs_data.reserve0 / pairs_data.reserve1)
    // against (pairs_data.reserve0 / pairs_data.reserve1) after accounting for fees, etc.
    // This is where the speed of Rust was intended to shine.

    // For now, let's just pretend we found something (very naively)
    // if some_condition_met {
    //     return Some("Found an opportunity!".to_string());
    // }
    None
}
```
Obviously, the actual arbitrage logic in Rust became much more complex, involving iterating through possible trade paths and calculating expected output amounts based on constant product formulas.

### The Flash Loan Challenge

This was the part that felt like walking a tightrope. The core idea was:
1.  Spot an arbitrage opportunity (e.g., TokenA -> TokenB on DEX1 is cheaper than TokenA -> TokenB on DEX2, effectively meaning you can buy TokenB cheaper on DEX1 and sell it higher on DEX2).
2.  Borrow TokenA using a flash loan (e.g., from Aave).
3.  Execute the first leg of the arbitrage: Swap TokenA for TokenB on DEX1.
4.  Execute the second leg: Swap TokenB back to TokenA on DEX2 (ending up with more TokenA than borrowed).
5.  Repay the flash loan (original TokenA amount + fee).
6.  All of this *must* happen in a single atomic transaction. If any step fails, the whole thing reverts, and the loan isn't actually taken (apart from wasted gas for the failed transaction attempt).

I decided to handle the flash loan interaction logic directly within a smart contract that my Python script would call. The Python script would prepare the parameters for this contract (which tokens, amounts, routes for swaps). Web3.py's `build_transaction` and `send_raw_transaction` were my bread and butter here.

One of the first major hurdles was calculating profitability accurately. It wasn't just about price differences; gas fees were a killer, especially on Ethereum. Then there was the flash loan fee itself (Aave's was 0.09% at the time I was looking). My profit calculation had to be pessimistic and account for worst-case gas scenarios. I spent a lot of time trying to simulate transactions using `w3.eth.call` before actually sending them, to get a gas estimate and check for reverts, but even this wasn't foolproof.

A specific moment of confusion I remember vividly was dealing with token decimals. I had a bug where my profit calculations were wildly off for certain pairs. It turned out I was naively assuming all tokens had 18 decimals. I had to add logic to dynamically fetch the `decimals()` for each token involved in a potential arbitrage and normalize all amount calculations. That was a facepalm moment, but a crucial learning experience. I found a helpful thread on a Geth forum that discussed common pitfalls with `eth_call` for simulations, which pointed me towards checking for subtle issues like that.

### The Never-Ending Battle: Gas, Latency, and Front-Running

Gas fees were a constant headache. An opportunity might look profitable, but by the time my transaction was mined, gas prices could have spiked, or the opportunity itself vanished. This led me to explore chains like Polygon more seriously.

Node latency was another beast. If my price data was even a few seconds stale, I was working with illusions. I started with Infura, but for more critical, real-time data, I briefly considered the complexity of running my own Geth/OpenEthereum (now Nethermind) node. The resource requirements and maintenance seemed too high for this project iteration, so I stuck with third-party providers but tried to optimize how frequently and efficiently I polled for data.

Then there's front-running. I was aware of Generalized Front-Runners (GFRs) and MEV (Maximal Extractable Value) bots. Any profitable transaction I broadcasted to the mempool was a target. For a student project, building sophisticated front-running protection (like using Flashbots RPC or private relays) felt out of scope, but it was a sobering realization that simply finding an opportunity wasn't enough; you also had to get it mined without being exploited. My naive attempts involved things like slightly varying gas prices, which, as I learned, don't do much against sophisticated actors.

### A Glimmer of Success (and a Lot of Learning)

Did I get rich? No, definitely not. The bot did manage to identify and execute a few *theoretically* profitable flash loan arbitrages on a testnet (forked mainnet environment using Hardhat, which was invaluable for this). When I saw the first one execute successfully from start to finish in my local Hardhat node – borrow, swap, swap, repay, profit – it was a huge moment of satisfaction.

Actually making consistent profit on a live mainnet (even a cheaper L2) was a different story, given the competition and gas fee volatility. However, the primary goal was learning, and in that respect, the project was a massive success.

Here’s a piece of the Python code that I used to estimate gas and call the arbitrage contract function. This went through many iterations.

```python
# w3, arbitrage_contract, account are already set up
# 'data' would be the encoded function call to my arbitrage contract

def execute_arbitrage_tx(w3_instance, contract_instance, function_signature_data, user_account, private_key):
    nonce = w3_instance.eth.get_transaction_count(user_account.address)
    
    # Building the transaction
    # Gas price strategy was a huge pain point - initially used w3.eth.gas_price
    # then tried to be more dynamic based on network conditions
    tx_fields = {
        'from': user_account.address,
        'to': contract_instance.address,
        'nonce': nonce,
        'data': function_signature_data,
        # 'gas': 0, # Will be estimated
        'gasPrice': w3_instance.to_wei('50', 'gwei') # Example fixed gas price, bad for real world
    }

    try:
        # Estimate gas - this can fail or give misleading results too!
        # I spent ages debugging why estimate_gas would sometimes be way off or just revert.
        # Often it was an issue *inside* the contract logic that only manifested with real parameters.
        gas_estimate = w3_instance.eth.estimate_gas(tx_fields)
        tx_fields['gas'] = int(gas_estimate * 1.2) # Adding a 20% buffer, a bit arbitrary
    except Exception as e:
        print(f"Gas estimation failed: {e}")
        return None

    signed_tx = w3_instance.eth.account.sign_transaction(tx_fields, private_key)
    
    try:
        tx_hash = w3_instance.eth.send_raw_transaction(signed_tx.rawTransaction)
        print(f"Transaction sent! Hash: {w3_instance.to_hex(tx_hash)}")
        receipt = w3_instance.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        return receipt
    except Exception as e:
        print(f"Transaction failed: {e}")
        return None

```
The `gasPrice` and `gas` limit settings were a constant source of tinkering. Too low, and the transaction wouldn't get picked up. Too high, and it would eat any potential profit. For a while, I was just hardcoding values based on what Etherscan was showing, which is obviously not a sustainable strategy.

### Reflections and What’s Next

This project pushed my understanding of DeFi, smart contracts, and low-level transaction mechanics much further than any course could have. Wrestling with Rust's async features using `tokio` for some experimental concurrent price fetching (which I eventually simplified because integrating it with the PyO3 part became too complex for my timeline) was a challenge in itself. Debugging was often a case of `print` statements in Python and `println!` in Rust, then slowly graduating to using the Hardhat console and Tenderly for transaction simulation and inspection when I really got stuck on contract interactions.

If I were to iterate, I'd focus more on:
1.  **Robust Gas Strategies:** Something more adaptive than what I had.
2.  **MEV Mitigation:** Researching and integrating with services like Flashbots.
3.  **Broader DEX & Chain Support:** Expanding the Rust core to handle more complex routing and more chains.
4.  **More Sophisticated Rust Logic:** Perhaps exploring more advanced pathfinding algorithms for arbitrage.

It was an intense experience, with many late nights fueled by coffee and the dim glow of my monitor, staring at transaction logs. But seeing those pieces click together, from Web3.py calls to Rust calculations to successful flash loan executions on a testnet, was incredibly validating. It’s one thing to read about these concepts, and quite another to try and build them from the ground up.