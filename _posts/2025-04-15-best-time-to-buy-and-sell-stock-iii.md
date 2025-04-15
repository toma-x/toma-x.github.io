---
layout: post
title: Best Time to Buy and Sell Stock III - At Most Two Transactions
---

Hey folks! Continuing our journey through the LeetCode stock trading problems, today we're tackling "Best Time to Buy and Sell Stock III". If you haven't read my earlier posts, check out [part 1](/2025/04/14/best-time-to-buy-and-sell-stock/) and [part 2](/2025/04/15/best-time-to-buy-and-sell-stock-ii/) first.

## The Problem

[Problem on LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

<blockquote>
You are given an array prices where prices[i] is the price of a given stock on the ith day.
Find the maximum profit you can achieve. You may complete at most two transactions.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
</blockquote>

For example, with prices [3,3,5,0,0,3,1,4]:
- Buy on day 4 (price = 0)
- Sell on day 6 (price = 3), profit = 3
- Buy on day 7 (price = 1)
- Sell on day 8 (price = 4), profit = 3
- Total profit = 6

## First Thoughts

My first instinct was to try to adapt the approach from part 2, but that doesn't work here because we're limited to just two transactions. We need to carefully decide when to make each transaction to maximize profit.

We cannot try all possible combinations of two transactions, since that would be O(n⁴), which is way too inefficient.

## Dynamic Programming Approach

This problem definitely calls for dynamic programming. We need to track our state at each day, including:
1. Which day we're on
2. How many transactions we've made so far
3. Whether we're currently holding a stock

Let's define some state variables:
- `T[i][j][0]` = maximum profit on day i, with j transactions completed, and not holding a stock
- `T[i][j][1]` = maximum profit on day i, with j transactions completed, and holding a stock

Here's how we might implement it:

```python
def maxProfit(prices):
    if not prices:
        return 0
    
    # Initialize our dp array
    # T[i][j][k] = max profit on day i with j transactions and k stocks (0 or 1)
    n = len(prices)
    T = [[[0 for _ in range(2)] for _ in range(3)] for _ in range(n)]
    
    # Base cases
    # On day 0, if we have a stock, our profit is -prices[0]
    T[0][0][0] = 0
    T[0][0][1] = -prices[0]
    
    # Cannot have completed any transactions on day 0
    T[0][1][0] = T[0][1][1] = T[0][2][0] = T[0][2][1] = float('-inf')
    
    for i in range(1, n):
        # No transactions yet
        T[i][0][0] = T[i-1][0][0]  # Still not holding
        T[i][0][1] = max(T[i-1][0][1], T[i-1][0][0] - prices[i])  # Buy or keep holding
        
        # One transaction completed (or about to be)
        T[i][1][0] = max(T[i-1][1][0], T[i-1][0][1] + prices[i])  # Sell or stay without stock
        T[i][1][1] = max(T[i-1][1][1], T[i-1][1][0] - prices[i])  # Buy again or keep holding
        
        # Two transactions completed (or about to be)
        T[i][2][0] = max(T[i-1][2][0], T[i-1][1][1] + prices[i])  # Sell or stay without stock
    
    # Return the maximum profit with at most 2 transactions and no stock at the end
    return max(0, T[n-1][0][0], T[n-1][1][0], T[n-1][2][0])
```

But this approach can be a bit complex and hard to wrap your head around. Let's try a cleaner approach.

## A More Intuitive Approach

We can track four variables that represent the state we can be in:
1. `buy1` - Maximum profit after buying first stock
2. `sell1` - Maximum profit after selling first stock
3. `buy2` - Maximum profit after buying second stock
4. `sell2` - Maximum profit after selling second stock

```python
def maxProfit(prices):
    buy1 = buy2 = float('-inf')
    sell1 = sell2 = 0
    
    for price in prices:
        # We want to minimize the cost of first buy
        buy1 = max(buy1, -price)
        
        # After first sell, we want to maximize profit
        sell1 = max(sell1, buy1 + price)
        
        # For second buy, we consider the profit from first transaction
        buy2 = max(buy2, sell1 - price)
        
        # For second sell, maximize final profit
        sell2 = max(sell2, buy2 + price)
    
    return sell2
```

This solution is much more elegant and easier to understand! 

## Understanding the Solution

Let's break down how this works:

1. `buy1 = max(buy1, -price)`: We're trying to minimize the cost of our first purchase, so we keep track of the maximum value of -price (i.e., the minimum price).

2. `sell1 = max(sell1, buy1 + price)`: We maximize the profit of our first transaction by selling at the best possible price, considering our buying price (`buy1`).

3. `buy2 = max(buy2, sell1 - price)`: For our second purchase, we factor in the profit from our first transaction. This is key - we're not just buying again in isolation, but considering our overall profit so far.

4. `sell2 = max(sell2, buy2 + price)`: Finally, we maximize our total profit by selling our second stock at the best possible price.

This approach is dynamic programming at its core - we're breaking down the problem into subproblems (each day's decision), and building up the solution incrementally.

## Generalizing the Pattern

I'm starting to see a pattern here. This approach could be extended for k transactions by adding more buy/sell variables. In fact, that's exactly what we'll need for the next problem in the series where we allow k transactions.

## Conclusion

This problem was definitely a step up in complexity from the first two. The limitation of only two transactions forces us to think more carefully about the optimal points to buy and sell. Dynamic programming provides a clean solution by tracking the state at each step.

I'm looking forward to tackling [the next problem](/2025/04/15/best-time-to-buy-and-sell-stock-iv/) where we'll generalize to k transactions. See you then! 