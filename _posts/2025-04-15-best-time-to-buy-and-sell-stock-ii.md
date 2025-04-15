---
layout: post
title: Best Time to Buy and Sell Stock II - Multiple Transactions
---

Hi again! Following up on my [previous post](/2025/04/14/best-time-to-buy-and-sell-stock/) about the first stock trading problem, today I'm going to talk about "Best Time to Buy and Sell Stock II". This version introduces a new twist to the problem.

## The Problem

[Problem on LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

>
You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
Find and return the maximum profit you can achieve.

For example, with prices [7,1,5,3,6,4]:
- Buy on day 2 (price = 1)
- Sell on day 3 (price = 5), profit = 4
- Buy on day 4 (price = 3)
- Sell on day 5 (price = 6), profit = 3
- Total profit = 7

## Initial Thoughts

At first, I was thinking this would require a complex dynamic programming solution with lots of state tracking. We cannot simply buy at every local minimum and sell at every local maximum... or can we?

We cannot use a greedy approach to just find all increasing sequences because we need to consider the opportunity cost of holding versus selling and buying again. For instance, if prices are [1,2,3,4,5], should I:
1. Buy at 1, sell at 5 (profit 4)
2. Buy at 1, sell at 2, buy at 2, sell at 3... and so on (also profit 4)

Hmm, actually, both seem to give the same result in this case.

## The Dynamic Programming Approach

After working through some examples, I realized something interesting: the maximum profit is just the sum of all positive price differences between consecutive days.

Let's try to understand why with dynamic programming:

- At each day, we have two possible states: holding a stock or not holding a stock
- For each state, we want to maximize our profit

Let's track two variables through the days:
1. `cash` - maximum profit if we don't have a stock
2. `hold` - maximum profit if we have a stock

Here's the solution:

```python
def maxProfit(prices):
    cash = 0           # Maximum profit if we don't have a stock
    hold = -float('inf')  # Maximum profit if we have a stock
    
    for price in prices:
        prev_cash = cash
        
        # Max profit if we don't have a stock: either keep not having, or sell what we had
        cash = max(cash, hold + price)
        
        # Max profit if we have a stock: either keep what we have, or buy with our cash
        hold = max(hold, prev_cash - price)
    
    # At the end, we should not be holding any stock
    return cash
```

But there's actually a simpler solution too:

```python
def maxProfit(prices):
    max_profit = 0
    
    for i in range(1, len(prices)):
        # If we can make profit, make the transaction
        if prices[i] > prices[i-1]:
            max_profit += prices[i] - prices[i-1]
    
    return max_profit
```

This works because we're allowed to buy and sell on consecutive days. So we're essentially capturing every uptick in the market. On any day the price is higher than the previous day, we just imagine we bought yesterday and sold today.

This is still a form of dynamic programming, just heavily simplified. We're making the optimal local decision at each step, which happens to lead to the globally optimal solution.

## Why This Works

I spent some time thinking about why the second approach works. Imagine the price sequence [1,2,3,4,5]:

- Classic approach: Buy at 1, sell at 5. Profit = 4.
- New approach: Buy at 1, sell at 2, buy at 2, sell at 3, etc. Profit = 1+1+1+1 = 4.

Both give the same answer! That's because (5-1) = (2-1)+(3-2)+(4-3)+(5-4).

The reason why this problem is different from the first one is that we're no longer restricted to just one transaction. So we can capture every single profit opportunity in the market.

## Wrapping Up

This problem taught me that sometimes the simple solution is staring you right in the face. What initially seemed like a complex dynamic programming problem turned out to have an elegant and simple solution.

In the [next post](/2025/04/15/best-time-to-buy-and-sell-stock-iii/), I'll tackle the next problem in the series, where we can make at most two transactions. I have a feeling things will get a bit more complex! 