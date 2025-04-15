---
layout: post
title: Best Time to Buy and Sell Stock IV - At Most k Transactions
---

Hello everyone! We're now on part 4 of our LeetCode stock trading series. If you're just joining, you might want to check out the previous posts: [part 1](/2025/04/14/best-time-to-buy-and-sell-stock/), [part 2](/2025/04/15/best-time-to-buy-and-sell-stock-ii/), and [part 3](/2025/04/15/best-time-to-buy-and-sell-stock-iii/).

## The Problem

[Problem on LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

>
You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.
Find the maximum profit you can achieve. You may complete at most k transactions: i.e. you may buy at most k times and sell at most k times.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

For example, with prices [3,2,6,5,0,3] and k = 2:
- Buy on day 2 (price = 2)
- Sell on day 3 (price = 6), profit = 4
- Buy on day 5 (price = 0)
- Sell on day 6 (price = 3), profit = 3
- Total profit = 7

## Initial Approach

In the previous problem, we found a pattern for handling exactly 2 transactions. Now we need to generalize this for any number k of transactions.

My first instinct is to extend the approach from the previous problem by creating arrays to track the state after each buy and sell operation, but that would require us to hardcode k different variables, which isn't scalable.

Instead, we need a more general approach that can handle any value of k.

## Dynamic Programming Solution

We can use a 2D array to keep track of our states:
- `dp[i][j]` = maximum profit up to day i with at most j transactions

The recurrence relation would be:
- `dp[i][j] = max(dp[i-1][j], max(dp[i-1][j-1] + prices[i] - prices[m]) for m in range(0, i))`

This means that on day i, with j transactions allowed, we either:
1. Don't do anything today (dp[i-1][j])
2. Sell today, having bought on some previous day m (dp[i-1][j-1] + prices[i] - prices[m])

But this approach is O(n²k) which is too slow for larger inputs. We can optimize it:

```python
def maxProfit(k, prices):
    if not prices or k == 0:
        return 0
    
    n = len(prices)
    
    # If k is large enough, we can make as many transactions as we want
    # This becomes the same as problem II
    if k >= n // 2:
        return sum(max(0, prices[i] - prices[i-1]) for i in range(1, n))
    
    # Initialize dp array
    # dp[i][j] = max profit with i transactions on day j
    dp = [[0 for _ in range(n)] for _ in range(k+1)]
    
    for i in range(1, k+1):
        # Initialize max difference with the first price
        local_max = -prices[0]
        
        for j in range(1, n):
            # Either we don't make a transaction on day j, or we sell on day j
            dp[i][j] = max(dp[i][j-1], prices[j] + local_max)
            
            # Update maximum difference
            local_max = max(local_max, dp[i-1][j-1] - prices[j])
    
    return dp[k][n-1]
```

I like this solution, but I think there's a more intuitive way to think about it. Let's approach it differently.

## A More Intuitive Approach

Let's think of our state as:
- `dp[i][j][0]` = maximum profit up to day i using at most j transactions, and not holding a stock
- `dp[i][j][1]` = maximum profit up to day i using at most j transactions, and holding a stock

The transitions would be:
- `dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])`
- `dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])`

Here's the implementation:

```python
def maxProfit(k, prices):
    if not prices or k == 0:
        return 0
    
    n = len(prices)
    
    # Special case: k is large enough to make as many transactions as we want
    if k >= n // 2:
        return sum(max(0, prices[i] - prices[i-1]) for i in range(1, n))
    
    # dp[j][0] = max profit with j transactions and not holding stock
    # dp[j][1] = max profit with j transactions and holding stock
    dp = [[-float('inf') for _ in range(2)] for _ in range(k+1)]
    dp[0][0] = 0
    
    for price in prices:
        for j in range(k, 0, -1):
            # Update for selling
            dp[j][0] = max(dp[j][0], dp[j][1] + price)
            
            # Update for buying
            dp[j][1] = max(dp[j][1], dp[j-1][0] - price)
    
    # Return maximum profit with at most k transactions and not holding stock
    return max(0, max(dp[j][0] for j in range(k+1)))
```

We cannot blindly update our states without considering the order of updates, since this can lead to using the updated values from the same day. Let's fix that:

```python
def maxProfit(k, prices):
    if not prices or k == 0:
        return 0
    
    n = len(prices)
    
    # Special case: k is large enough to make as many transactions as we want
    if k >= n // 2:
        return sum(max(0, prices[i] - prices[i-1]) for i in range(1, n))
    
    # Initialize dp array
    # buy[i] = maximum profit after making i purchases
    # sell[i] = maximum profit after making i sales
    buy = [-float('inf')] * (k + 1)
    sell = [0] * (k + 1)
    
    for price in prices:
        for i in range(1, k + 1):
            # Maximum profit if we buy the stock with i purchases
            buy[i] = max(buy[i], sell[i-1] - price)
            
            # Maximum profit if we sell the stock with i sales
            sell[i] = max(sell[i], buy[i] + price)
    
    # Return maximum profit after k transactions
    return sell[k]
```

This solution is much cleaner and more efficient. It has a time complexity of O(nk) and space complexity of O(k).

## Understanding the Solution

The approach we're using is still dynamic programming, but with a more compact representation:

1. `buy[i]` keeps track of the maximum profit after making i purchases. We initialize it to negative infinity since we can't make a profit just by buying.

2. `sell[i]` keeps track of the maximum profit after making i sales. We initialize it to 0 since without any transactions, the profit is 0.

3. For each price, we iterate through our transaction limits and update our states:
   - To update `buy[i]`, we either keep our previous state or buy at the current price after having made i-1 sales.
   - To update `sell[i]`, we either keep our previous state or sell at the current price after having made i purchases.

4. Finally, we return `sell[k]`, which represents the maximum profit after making k transactions.

The optimization to handle large k values is important - if k is at least n/2, we can buy and sell as many times as we want (similar to problem II), so we just sum up all the positive price differences.

## Conclusion

This has been the most challenging problem in the series so far. It requires us to carefully track our states and understand the transitions. The key insight is to handle the buy and sell states separately, and to optimize for the case when k is large.

In the [next post](/2025/04/15/best-time-to-buy-and-sell-stock-with-cooldown/), we'll look at a variation with a cooldown period after selling. Stay tuned! 