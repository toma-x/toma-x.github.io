---
layout: post
title: Best Time to Buy and Sell Stock - LeetCode Problem
---

Hey there! Today I thought I'd write about one of the classic LeetCode problems I've been tackling: Best Time to Buy and Sell Stock. It's one of those problems that initially seems straightforward but has some nice subtleties to it.

## The Problem

[Problem on LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

For example, if you have prices [7,1,5,3,6,4]:
- Buy on day 2 (price = 1)
- Sell on day 5 (price = 6)
- Profit = 6 - 1 = 5

## First Approach

My initial instinct was to use a nested loop - check every possible buying day against every possible future selling day. Something like:

```python
def maxProfit(prices):
    max_profit = 0
    for i in range(len(prices)):
        for j in range(i+1, len(prices)):
            profit = prices[j] - prices[i]
            max_profit = max(max_profit, profit)
    return max_profit
```

But this approach is O(n²), which isn't great for larger inputs. We can't just look at the minimum and maximum values in the array, since we need to buy before we sell.

## A Dynamic Programming Approach

We can actually solve this with a dynamic programming approach, though it's a simpler form than many DP problems.

We need to track two things as we iterate through the prices:
1. The minimum price we've seen so far (potential buying point)
2. The maximum profit we could make by selling at the current price

```python
def maxProfit(prices):
    if not prices:
        return 0
        
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        # Update the minimum price seen so far
        min_price = min(min_price, price)
        
        # Calculate potential profit if we sell at current price
        current_profit = price - min_price
        
        # Update maximum profit if current profit is better
        max_profit = max(max_profit, current_profit)
    
    return max_profit
```

The beauty of this approach is that we only need a single pass through the array, making it O(n) time complexity.

This is actually a form of dynamic programming because we're:
1. Breaking down the problem into simpler subproblems (what's the best decision at each day)
2. Storing and reusing previously computed results (min_price)
3. Building up our solution incrementally

I haven't mastered dynamic programming yet, but this problem helped me understand the concept better. There's probably more elegant ways to explain or implement this, but I found this approach intuitive.

## Takeaway

This problem is a good introduction to the stock trading series on LeetCode. It teaches an important lesson: sometimes we don't need elaborate data structures or complex algorithms - just a careful tracking of a few key values as we iterate through the data.

In my [next post](/2023/06/16/best-time-to-buy-and-sell-stock-ii/), I'll tackle the follow-up problem where we can make multiple transactions. Stay tuned! 