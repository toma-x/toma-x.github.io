---
layout: post
title: AI-Driven Factor Discovery
---

## AI-Driven Factor Discovery: The JAX Journey

This one took a while. The `AI-Driven Factor Discovery` project finally feels like it's at a point where I can write about it. The core idea was to use symbolic regression to find new alpha factors from market data. I'd seen [GPlearn](https://gplearn.readthedocs.io/en/stable/) mentioned in a few papers and it seemed like a powerful approach for this kind of "equation discovery" problem. My initial thought was, "Great, I'll just use that." But then there was this self-imposed constraint (or maybe it was from that advanced ML elective I took last semester, the one that really pushed functional programming) – I wanted to do all the core machine learning bits in Python using *only* JAX.

Why JAX? Well, partly curiosity, partly the promise of speed with `jax.jit`, and honestly, partly because I thought it would look good on my resume. Professor A mentioned in an office hour how JAX is gaining traction for research because of its flexibility with `jax.grad` and transformations like `jax.vmap`. I wasn't even sure if symbolic regression, being a genetic programming (GP) thing, would really benefit from autodiff in the traditional sense, but I figured at least the numerical evaluations of the candidate factors could be sped up.

The first hurdle was realizing that there isn't really a mature, ready-to-go symbolic regression library for JAX quite like GPlearn. I spent a good week searching, going through [JAX's ecosystem page](https://github.com/google/jax#ecosystem) and various GitHub repos. Found some interesting JAX-based evolutionary libraries, but nothing that quite fit what I envisioned for traditional GP for symbolic regression. That was a bit of a "uh oh" moment. My friend, Sarah, who was working on a vision project with Flax, suggested I look into just wrapping GPlearn and using JAX for specific computations. But that felt like cheating my "JAX only" rule for the ML part.

So, I decided to try and build the core GP components myself in JAX. How hard could it be, right? (Famous last words.)

My starting point was to define how to represent the "programs" or "factors." In GP, these are typically trees. I thought about how to do this in a JAX-friendly way. Since JAX loves pure functions, representing programs as callable functions seemed natural. The actual *structure* of these functions (the tree) would be manipulated by the GP operations. I decided on a simple set of primitives: `add`, `sub`, `mul`, `protected_div` (to avoid `NaN`s), `neg`, maybe `jnp.sin`, `jnp.cos`, `jnp.log`, `jnp.exp` from `jax.numpy`. The terminals would be various market data features – things like lagged prices, moving averages, RSI, etc., which I pre-calculated using Pandas from my daily OHLCV data. For instance, `x[:, 0]` might be a 5-day ROC, `x[:, 1]` could be volatility.

A candidate factor (an individual in GP terms) could be represented as a tree structure, and its evaluation would be a JAX function. For instance, a simple factor like `(close - open) / volume` would translate to a tree, and then a JAX function that takes the market data arrays as input.

The genetic operations – crossover and mutation – were the tricky part. How do you swap subtrees or modify nodes in an immutable, JAX-compatible way? I didn't use `jax.tree_util` for the trees themselves initially, which was a mistake I later revisited. I ended up doing a lot of manual index-based manipulation on a list-of-lists representation of the trees, which was clunky and error-prone. Debugging `jax.jit`-compiled functions doing these manipulations... let's just say I became very familiar with `jax.debug.print` and the "banana" error messages.

Managing the PRNG keys for mutation and crossover was another JAX-specific learning curve. You can't just call `np.random` everywhere. Everything needs a `jax.random.PRNGKey`, and it needs to be split and passed down. My initial population generation looked something like this (conceptually, my actual code was messier):

```python
import jax
import jax.numpy as jnp

# Simplified example of generating a random program (tree)
# This is not how I actually ended up doing it, this is just for illustration
# of the PRNGKey handling I had to learn.
# My actual program representation was more complex to allow for JAX evaluation.

def create_random_program(key, depth, primitives, terminals, x_shape_dim1):
    # key needs to be split for every random choice
    # ... logic to recursively build a tree using JAX random functions ...
    # For instance, choosing a primitive or a terminal:
    key_op, key_choice = jax.random.split(key)
    is_primitive = jax.random.uniform(key_op) > 0.5 # simplified
    
    prog_structure = [] # this would be more structured in reality
    if is_primitive and depth > 0:
        # choose a primitive
        op_idx = jax.random.randint(key_choice, shape=(), minval=0, maxval=len(primitives))
        # ... recursively call for children, splitting keys ...
        prog_structure.append(primitives[op_idx])
    else:
        # choose a terminal
        term_idx = jax.random.randint(key_choice, shape=(), minval=0, maxval=x_shape_dim1)
        prog_structure.append(f"x[:, {term_idx}]") # placeholder for actual terminal representation
    return prog_structure

# This is a conceptual snippet, my actual implementation for tree generation
# and evaluation was much more involved to be JAX-traceable and support GP operations.
# For actual evaluation, I compiled the tree into a JAX callable.
```
Honestly, that `create_random_program` snippet is massively simplified. The real challenge was making the *evaluation* of these dynamically generated programs efficient in JAX. `jax.lax.switch` or masked operations came up when trying to evaluate programs of different structures within a `vmap`ped fitness evaluation across the population. I remember a [StackOverflow thread](https://stackoverflow.com/questions/something-similar-to-this-but-real) (okay, maybe not this exact one, but similar) that discussed dynamic computation in JAX which gave me some ideas, but it was still a struggle.

For the fitness function, the goal was to maximize the Sharpe ratio of the generated factor. This involved evaluating the factor over historical data (my training set), calculating its returns, and then the Sharpe. The evaluation of *many* factors (the population) over *many* timesteps of data seemed like a perfect candidate for `jax.vmap` and `jax.jit`. My fitness function looked roughly like `fitness = calculate_sharpe(evaluate_factor(program, market_data_train))`. `evaluate_factor` would be the JAX-compiled function derived from the tree.

I initially tried to make parts of the GP process itself differentiable using JAX's `grad`, thinking I could somehow "learn" better mutation operators or something. That turned out to be a rabbit hole leading nowhere for this project and I quickly abandoned it. GP is fundamentally a search algorithm, not a gradient descent one. The gradients I needed were just for any internal JAX operations I might have built, not for the evolutionary process itself. Sticking to JAX for fast numerical computation was the real win.

The selection mechanism was tournament selection, fairly standard GP stuff. Crossover involved picking two parents, selecting random crossover points in their tree representations, and swapping the subtrees. Mutation could involve changing a node (e.g., an operator to another operator, a terminal to another terminal) or regenerating an entire subtree. Doing this while respecting JAX's immutability meant creating new program instances rather than modifying them in-place. This felt very functional, but also memory-intensive if not managed carefully.

There were many false starts. My first attempt at tree representation was just a nested Python list/tuple structure. JAX could `jit` functions that *used* these, but manipulating them *within* `jit`-compiled GP operations like mutation was a nightmare. I eventually moved to a more flattened representation for some operations, inspired by how some compilers handle abstract syntax trees.

After (what felt like) an eternity of debugging and tuning the JAX-based GP system, I started getting some interesting-looking factors. The expressions were often quite complex and non-intuitive, which is exactly what I was hoping for – things I wouldn't have come up with manually.

Then came the backtesting. This part I did *outside* of JAX, using the more traditional stack of [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), and [Statsmodels](https://www.statsmodels.org/) for OLS regressions and more detailed performance statistics. The process was:
1.  Take the best JAX-generated factor expressions (which were essentially Python callables that took NumPy arrays as input, since JAX NumPy is so compatible).
2.  Apply these factor functions to my out-of-sample market data (a separate holdout set).
3.  Calculate daily returns based on the factor values (e.g., go long top quintile, short bottom quintile).
4.  Compute the Sharpe ratio, drawdown, turnover, etc.

I did hit a high Sharpe on a few of them, which was incredibly rewarding. Of course, there's always the specter of overfitting with GP, given the massive search space. I tried to be careful with a strict train/validation/test split, and the validation set was used for the GP fitness, while the final test set was only for the very end evaluation. The [Wikipedia page on Symbolic Regression](https://en.wikipedia.org/wiki/Symbolic_regression) has a good overview of the method, and it often mentions the risk of bloat (programs getting overly complex), which I definitely saw and tried to penalize in my fitness function.

One specific factor that came out looked something like `jnp.log(rank(volume_roc_10) * rank(close_std_20)) - rank(adv_5_roc_3)`. It's not immediately obvious why it would work, but the backtest on the holdout period was surprisingly good. `rank` here refers to cross-sectional ranking, and `adv_5_roc_3` was average daily volume 5-day rate of change over 3 days (or something similar, I had a convention for naming my pre-calculated terminals).

What I learned:
*   Building a GP system from scratch, even a simplified one, is a *lot* more work than just `from gplearn.genetic import SymbolicRegressor`.
*   JAX is incredibly powerful for numerical computation, but it has a steep learning curve, especially around PRNG, state management, and debugging `jit`-compiled code. The "pure functional" paradigm takes getting used to.
*   The "think in arrays" mentality of NumPy, when supercharged by JAX's transformations, is very effective for financial data.
*   Symbolic regression is a fascinating technique for generating novel hypotheses.
*   Don't underestimate the infrastructure needed around the core algorithm (data loading, preprocessing, backtesting framework). I spent as much time on that as on the JAX GP core.

Would I do it this way again? For the learning experience with JAX, absolutely. For production, I'd probably look much harder for existing robust JAX-native evolutionary libraries, or contribute to one. This project was a deep dive, and while the results are promising, the engineering effort to get the JAX GP part working was substantial. It felt like I was wrestling with the machine a lot of the time, especially when a `ConcretizationTypeError` would pop up from deep inside a `jit`-compiled function after a tiny change. But seeing those Sharpe ratios at the end, from factors generated by *my own* JAX code... yeah, that was pretty cool. Now to write it up properly for my actual thesis...