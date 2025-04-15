---
layout: post
title: Best Time to Buy and Sell Stock with Cooldown
---

Hey everyone! We're now at part 5 of our stock trading LeetCode series. Check out the previous posts if you missed them: [part 1](/2025/04/14/best-time-to-buy-and-sell-stock/), [part 2](/2025/04/15/best-time-to-buy-and-sell-stock-ii/), [part 3](/2025/04/15/best-time-to-buy-and-sell-stock-iii/), and [part 4](/2025/04/15/best-time-to-buy-and-sell-stock-iv/).

## The Problem

[Problem on LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

> You are given an array prices where prices[i] is the price of a given stock on the ith day.
> Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
> After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
> Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

For example, with prices [1,2,3,0,2]:
- Buy on day 1 (price = 1)
- Sell on day 2 (price = 2), profit = 1
- Cooldown on day 3
- Buy on day 4 (price = 0)
- Sell on day 5 (price = 2), profit = 2
- Total profit = 3

## First Thoughts

This problem introduces a new constraint - the cooldown period. My first instinct is that we need to track an additional state: whether we're in a cooldown period or not.

We can't use the simple approach from Problem II (just summing up all positive price differences) because now we have to consider the opportunity cost of selling today versus waiting for potentially better returns after the cooldown.

## Dynamic Programming Approach

Let's try using a state machine approach. At each day, we can be in one of three states:
1. **Hold**: We're holding a stock.
2. **Sold**: We just sold a stock.
3. **Ready**: We're ready to buy (not in cooldown).

The transitions would be:
- Hold → Hold (keep holding) or Hold → Sold (sell)
- Sold → Ready (cooldown)
- Ready → Ready (do nothing) or Ready → Hold (buy)

Let's implement this:

```python
def maxProfit(prices):
    if not prices:
        return 0
    
    n = len(prices)
    
    # Initialize our three states
    hold = -prices[0]  # Maximum profit if holding a stock
    sold = float('-inf')  # Maximum profit if just sold a stock
    ready = 0  # Maximum profit if ready to buy (not in cooldown)
    
    for i in range(1, n):
        prev_hold, prev_sold, prev_ready = hold, sold, ready
        
        # Maximum profit if holding a stock today
        hold = max(prev_hold, prev_ready - prices[i])
        
        # Maximum profit if just sold a stock today
        sold = prev_hold + prices[i]
        
        # Maximum profit if ready to buy today (not in cooldown)
        ready = max(prev_ready, prev_sold)
    
    # At the end, we want to be either ready to buy or just sold
    return max(ready, sold)
```

This approach tracks our state at each day and computes the maximum profit based on the previous day's states.

## Understanding the Solution

Let's break down the recurrence relations:

1. `hold = max(prev_hold, prev_ready - prices[i])`: We either continue holding from the previous day, or we were ready to buy and decided to buy at today's price.

2. `sold = prev_hold + prices[i]`: If we're selling today, we must have been holding a stock yesterday, and we gain the current price.

3. `ready = max(prev_ready, prev_sold)`: We're ready to buy if either we were ready yesterday and decided not to buy, or we sold a stock the day before yesterday (cooldown period is over).

The final result is the maximum of being ready to buy or having just sold, since we don't want to end up holding a stock.

## Alternative DP Approach

Another way to think about this is to use a more traditional DP approach with arrays:

```python
def maxProfit(prices):
    if not prices:
        return 0
    
    n = len(prices)
    
    # Initialize DP arrays
    # buy[i] = max profit ending on day i with a buy or holding
    # sell[i] = max profit ending on day i with a sell or cooldown
    buy = [0] * n
    sell = [0] * n
    
    buy[0] = -prices[0]
    
    for i in range(1, n):
        # Either keep the stock we had, or buy a new one (if it's not from a sell on the previous day)
        buy[i] = max(buy[i-1], sell[i-2] - prices[i] if i >= 2 else -prices[i])
        
        # Either don't sell, or sell the stock we're holding
        sell[i] = max(sell[i-1], buy[i-1] + prices[i])
    
    # We want to end with a sell or in cooldown
    return sell[n-1]
```

This solution explicitly models the cooldown by looking two days back when deciding whether to buy. We can't buy immediately after a sell, so we use `sell[i-2]` when considering a buy on day i.

## Optimization

The above solution uses O(n) space, but we can optimize it to O(1) space since we only need the last two values of each array:

```python
def maxProfit(prices):
    if not prices:
        return 0
    
    n = len(prices)
    
    # Initialize our values
    buy_prev = -prices[0]  # buy[i-1]
    sell_prev = 0  # sell[i-1]
    sell_prev_prev = 0  # sell[i-2]
    
    for i in range(1, n):
        buy_curr = max(buy_prev, sell_prev_prev - prices[i])
        sell_curr = max(sell_prev, buy_prev + prices[i])
        
        # Update for next iteration
        buy_prev, sell_prev, sell_prev_prev = buy_curr, sell_curr, sell_prev
    
    return sell_prev
```

## Comparison with Previous Problems

This problem is similar to Problem II (multiple transactions allowed), but with the added constraint of a cooldown period. It forces us to think more carefully about when to buy and sell, because selling today impacts our ability to buy tomorrow.

The cooldown mechanism is an interesting twist that makes the problem more realistic - in real markets, there might be settlement periods or other constraints that prevent immediate re-entry into a position.

## Conclusion

The cooldown constraint adds an interesting layer of complexity to the stock trading problem. It forces us to consider the temporal dependencies between our actions more carefully.

Dynamic programming again proves to be a powerful approach for these types of problems, allowing us to break down the complex decision-making process into manageable steps.

In the [final post](/2025/04/15/best-time-to-buy-and-sell-stock-with-transaction-fee/) of this series, we'll tackle a variation where there's a transaction fee for each trade. See you then! 