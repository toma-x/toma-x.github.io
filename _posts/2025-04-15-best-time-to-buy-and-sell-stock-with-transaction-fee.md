---
layout: post
title: Best Time to Buy and Sell Stock with Transaction Fee
---

Hi all! Welcome to the final installment of my LeetCode stock trading problem series. If you're joining in the middle, be sure to check out the previous posts: [part 1](/2025/04/14/best-time-to-buy-and-sell-stock/), [part 2](/2025/04/15/best-time-to-buy-and-sell-stock-ii/), [part 3](/2025/04/15/best-time-to-buy-and-sell-stock-iii/), [part 4](/2025/04/15/best-time-to-buy-and-sell-stock-iv/), and [part 5](/2025/04/15/best-time-to-buy-and-sell-stock-with-cooldown/).

## The Problem

[Problem on LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

>
You are given an array prices where prices[i] is the price of a given stock on the ith day, and an integer fee representing a transaction fee.
Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.
Note:
You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
The transaction fee is only charged once for each stock purchase and sale.

For example, with prices [1,3,2,8,4,9] and fee = 2:
- Buy on day 1 (price = 1)
- Sell on day 4 (price = 8), profit = 8 - 1 - 2 = 5
- Buy on day 5 (price = 4)
- Sell on day 6 (price = 9), profit = 9 - 4 - 2 = 3
- Total profit = 8

## Initial Thoughts

This problem introduces yet another real-world constraint - transaction fees. In real trading, brokers charge fees for executing trades, and these fees can significantly impact the profitability of a trading strategy.

My first instinct is that this is somewhat similar to Problem II (multiple transactions), but with the transaction fee, we need to be more selective about which transactions we make. Small price increases might not be worth the fee.

## Dynamic Programming Approach

As with the previous problems, we can use dynamic programming. Let's define two states:
- `hold[i]` = maximum profit at the end of day i if we're holding a stock
- `cash[i]` = maximum profit at the end of day i if we're not holding a stock

The transitions would be:
- `hold[i] = max(hold[i-1], cash[i-1] - prices[i])` (continue holding or buy)
- `cash[i] = max(cash[i-1], hold[i-1] + prices[i] - fee)` (stay in cash or sell and pay fee)

Here's how we can implement it:

```python
def maxProfit(prices, fee):
    if not prices:
        return 0
    
    n = len(prices)
    
    # Initialize our states
    hold = -prices[0]  # Maximum profit if holding a stock
    cash = 0  # Maximum profit if not holding a stock
    
    for i in range(1, n):
        prev_hold, prev_cash = hold, cash
        
        # Maximum profit if holding a stock at the end of day i
        hold = max(prev_hold, prev_cash - prices[i])
        
        # Maximum profit if not holding a stock at the end of day i
        cash = max(prev_cash, prev_hold + prices[i] - fee)
    
    # At the end, we want to be in the cash state
    return cash
```

This approach is simple and elegant. It has a time complexity of O(n) and a space complexity of O(1).

## Understanding the Solution

Let's break down how this solution works:

1. We initialize `hold` to -prices[0], which represents the state where we buy the stock on day 0.
2. We initialize `cash` to 0, which represents the state where we don't buy anything on day 0.
3. For each subsequent day, we update our states:
   - `hold`: We either continue holding the stock from the previous day, or we buy a new stock using our cash.
   - `cash`: We either continue not holding a stock, or we sell the stock we were holding and pay the transaction fee.
4. At the end, we return `cash` since we want to end up not holding any stock.

The transaction fee is subtracted from our profit when we sell a stock. This makes us think twice about selling a stock if the profit isn't enough to cover the fee.

## Generalization of Previous Problems

This problem is a generalization of Problem II (multiple transactions). In fact, if we set fee = 0, it becomes exactly the same as Problem II.

Thinking back on the entire series, we can see that all these problems are variations of the same core problem, with different constraints:
- Problem I: At most 1 transaction
- Problem II: Unlimited transactions
- Problem III: At most 2 transactions
- Problem IV: At most k transactions
- Problem V: Unlimited transactions with cooldown
- Problem VI: Unlimited transactions with transaction fee

Each constraint requires us to adapt our approach, but the underlying dynamic programming framework remains the same.

## Alternative Perspective

Another way to think about the transaction fee is to incorporate it into the buying price. Instead of paying the fee when selling, we could pay it when buying:

```python
def maxProfit(prices, fee):
    if not prices:
        return 0
    
    n = len(prices)
    
    # Initialize our states
    hold = -(prices[0] + fee)  # Maximum profit if holding a stock
    cash = 0  # Maximum profit if not holding a stock
    
    for i in range(1, n):
        prev_hold, prev_cash = hold, cash
        
        # Maximum profit if holding a stock at the end of day i
        hold = max(prev_hold, prev_cash - prices[i] - fee)
        
        # Maximum profit if not holding a stock at the end of day i
        cash = max(prev_cash, prev_hold + prices[i])
    
    # At the end, we want to be in the cash state
    return cash
```

This approach gives the same result but conceptually charges the fee at a different point in the transaction. It's a matter of preference which version to use.

## Real-World Applications

These LeetCode stock trading problems teach us about dynamic programming, but they also provide insights into real-world trading strategies. Factors like transaction fees, cooldown periods (settlement periods), and limited capital (limited number of transactions) all affect trading decisions.

Of course, real-world trading is much more complex, with factors like market impact, slippage, partial fills, and more. But the principles we've learned here - careful state tracking and optimal decision-making at each step - are broadly applicable.

## Conclusion

And with that, we've completed our journey through the LeetCode stock trading problem series! I hope these posts have helped you understand these problems and the dynamic programming approaches to solve them.

Dynamic programming can be challenging to grasp at first, but it's an incredibly powerful technique for solving optimization problems like these. The key is to identify the states we need to track and the transitions between them.

Thanks for reading, and happy coding! If you have any questions or insights about these problems, feel free to reach out.

In the next series of posts, I'll tackle a different family of LeetCode problems. Stay tuned! 