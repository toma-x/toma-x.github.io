---
layout: post
title: Financial Counterparty Risk Simulation
---

Implementing an optimized pricing kernel in C++ and binding it to Python, as I discussed in my previous post, was a challenging but rewarding experience focused on raw computational speed for specific tasks. That project really highlighted the performance gains possible by moving out of pure Python for number-crunching. Building on that interest in quantitative finance applications, I recently tackled another personal project: developing a Monte Carlo simulation framework in Python specifically for financial counterparty risk modeling.

The basic idea behind this project was to simulate thousands of possible future market scenarios and, within each scenario, calculate the potential exposure I might have to a hypothetical counterparty at various points in time. Aggregating these exposures across all scenarios allows you to estimate metrics like Potential Future Exposure (PFE), which is essentially a high percentile (like the 95th or 99th) of the exposure distribution at a future date. This is a crucial part of understanding and managing credit risk in financial transactions.

Coming off the C++ kernel project, my initial thought was whether parts of this simulation, especially the market path generation or instrument pricing *within* the simulation, could benefit from compiled code. However, after mapping out the structure – simulating paths for potentially multiple underlying assets, calculating exposures for potentially multiple trades at multiple future dates *for each path* – I realized the complexity wasn't just in a single, tight calculation loop like the Black-Scholes price. It involved a lot of array manipulation, conditional logic based on trade types, and aggregation. Python, particularly with its scientific computing stack, felt like a much better fit for rapid prototyping and managing this higher level of complexity, even if it meant sacrificing some raw speed compared to C++. My comfort level with data handling in Python (thanks to pandas, though I tried to minimize its use in the core simulation for performance) also pushed me this way.

So, the core stack became Python 3, leveraging NumPy for array operations and SciPy for statistical functions (primarily distribution sampling and percentiles). I briefly considered dedicated financial modeling libraries, but they often felt too high-level or required specific license types. Building it myself, while slower to develop, allowed me to understand every piece of the simulation logic, which was a key learning goal.

The first major hurdle was generating correlated market scenarios. For simplicity, I modeled asset prices (like stocks) using a Geometric Brownian Motion (GBM) process. Generating *uncorrelated* random paths using `numpy.random.standard_normal` was straightforward, maybe taking an hour or two to get the basic loop structure right and ensure the time stepping (`dt`) was handled correctly. The complexity came with introducing correlation. I needed to generate random variables that had a specific correlation structure matching historical market data.

This led me down the rabbit hole of correlation matrices and techniques like the Cholesky decomposition. The idea is you take your desired correlation matrix, compute its Cholesky decomposition (`L` such that `L @ L.T` equals the correlation matrix), generate independent standard normal random numbers, and then multiply them by `L`. The resulting numbers will have the desired correlation. Implementing this in NumPy using `numpy.linalg.cholesky` was relatively painless *once I understood the math*, but debugging why my *output* paths didn't perfectly match the input correlation took some time. I initially messed up the matrix multiplication order or applied it incorrectly within the time loop. It took perhaps a solid evening (3-4 hours) of rereading documentation and StackOverflow posts on generating correlated random numbers to get the Cholesky approach correctly integrated into my path generation loop.

Here's a simplified snippet of how I ended up generating the correlated paths. Note the use of `np.dot` for matrix multiplication, which is key here.

```python
import numpy as np
# import scipy.stats as sp_stats # Initially thought I might need scipy for sampling, but numpy.random is sufficient

def generate_correlated_paths(S0, mu, sigma, corr_matrix, T, dt, num_paths):
    """
    Generates correlated asset price paths using Geometric Brownian Motion.

    S0: Initial prices (NumPy array)
    mu: Drift rates (NumPy array)
    sigma: Volatilities (NumPy array)
    corr_matrix: Target correlation matrix (NumPy array)
    T: Total time horizon (float)
    dt: Time step (float)
    num_paths: Number of simulation paths (int)
    """
    num_assets = len(S0)
    num_steps = int(T / dt)
    # I forgot to add a check here initially, T must be divisible by dt
    # Led to weird errors with array sizes later if T/dt wasn't integer.
    # Added a round() + int() conversion and a warning message later.

    # Compute Cholesky decomposition of the correlation matrix
    # This gave me trouble initially, ensuring the matrix was PSD (positive semi-definite)
    # if it came from real data - numpy can throw errors if not.
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        print("Error: Correlation matrix is not positive semi-definite. Cannot compute Cholesky decomposition.")
        return None # Or handle error appropriately

    # Array to store paths
    # Shape: (num_steps + 1, num_paths, num_assets) - steps first, then paths, then assets
    # Deciding array shape upfront is crucial for numpy performance.
    # Took me a while to settle on this indexing order for efficiency.
    paths = np.zeros((num_steps + 1, num_paths, num_assets))
    paths[0, :, :] = S0 # Set initial prices for all paths

    # Generate random numbers - shape (num_steps, num_paths, num_assets)
    # Need independent normals first
    independent_normals = np.random.standard_normal((num_steps, num_paths, num_assets))

    # Apply Cholesky decomposition to induce correlation
    # This was the part that took time to get right with broadcasting/dot product
    # Initially, I tried looping or using element-wise multiplication, which was wrong.
    # np.dot does matrix multiplication correctly here.
    # Reshape L for broadcasting: (1, 1, num_assets, num_assets)
    # Reshape normals for multiplication: (num_steps, num_paths, num_assets, 1)
    # The output should be (num_steps, num_paths, num_assets)
    # After experimenting, using einsum or just carefully shaped dot was faster than naive loops.
    # Stuck with a simple np.dot broadcast here by reshaping normals to (num_steps*num_paths, num_assets)
    # applying dot with L, then reshaping back. More efficient.
    # Correlated_normals = np.einsum('ij,kjl->kil', L, independent_normals) # Tried einsum, settled on simpler reshape+dot
    correlated_normals = np.dot(independent_normals.reshape(-1, num_assets), L.T).reshape(num_steps, num_paths, num_assets)


    # Simulate paths step by step
    for t in range(num_steps):
        # This is the core GBM step, fully vectorized across paths and assets
        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion_term = sigma * np.sqrt(dt) * correlated_normals[t, :, :]

        # Update prices - forgot np.exp initially, simple addition is wrong!
        paths[t+1, :, :] = paths[t, :, :] * np.exp(drift_term + diffusion_term)

        # Edge case: prices going negative? GBM shouldn't, but numerical errors?
        # Added a np.maximum(paths[t+1, :, :], 1e-6) safeguard just in case after seeing weird outputs once.


    return paths

# Example Usage (parameters need to be defined)
# S0 = np.array([100.0, 50.0])
# mu = np.array([0.05, 0.06])
# sigma = np.array([0.2, 0.25])
# corr_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
# T = 1.0 # 1 year
# dt = 1/252.0 # Daily steps
# num_paths = 10000
# market_paths = generate_correlated_paths(S0, mu, sigma, corr_matrix, T, dt, num_paths)
# if market_paths is not None:
#     print("Generated paths shape:", market_paths.shape) # Should be (253, 10000, 2)
```

Once I had the market paths, the next step was calculating the exposure to the counterparty along each path at specific future dates. This involved pricing the financial instruments (like options or simple swaps) we assume are outstanding with the counterparty using the simulated market prices at each time step. For simplicity in this framework, I focused on plain vanilla options, which meant using Black-Scholes again.

Initially, I wrote a simple Python function to price *one* option at a time, taking single values for S, K, T, etc. I quickly realized that calling this function repeatedly inside the simulation loop (for each path, for each time step, for each option) would be incredibly slow. This mirrored my experience with the pure-Python pricing from the previous project, but this time the need for speed was within the simulation itself, not for a one-off calculation.

I considered integrating my C++ pricing kernel. The `black_scholes_batch` function I'd built was designed for speed with large arrays. However, the structure of the simulation loop required pricing potentially many *different* options (different strikes, maturities remaining, etc.) at each time step, and the inputs (S, T remaining) changed with the path and time step. Preparing the input arrays for a single batch call to the C++ kernel at each internal simulation step felt complicated and potentially incurred significant Python/C++ boundary crossing overhead if the batches weren't perfectly optimized.

Instead, I pivoted to fully vectorizing the Black-Scholes calculation *within* the Python simulation loop itself using NumPy. This meant rewriting the Black-Scholes formula to accept NumPy arrays for S, K, T, etc., and perform calculations element-wise or using NumPy functions (`np.log`, `np.sqrt`, `np.exp`, etc.) and a vectorized normal CDF function (found a stable public domain implementation online and ported it, similar to my C++ project, avoiding SciPy's potentially slower `cdf` function for large arrays). This vectorized Python pricing function could then take the relevant slice of market paths (`paths[t, :, asset_index]`) and the parameters for a specific trade type (strikes, volatilities are fixed per trade but applied across all paths), calculate prices for all paths at time `t` in one go, and return an array of prices.

Refactoring the pricing logic to be fully vectorized took about two days (around 10-12 hours), largely spent figuring out the correct NumPy array broadcasting rules and implementing the vectorized normal CDF reliably. Debugging involved comparing the output array prices against a simple single-option Black-Scholes calculator for various inputs to ensure the vectorized version was correct.

After calculating the value of each trade for each path at each time step, the next step was to aggregate these values to get the total portfolio exposure to the counterparty along each path. For simple trades like options, this might just be summing up the positive values (since exposure is typically defined as the cost to replace the position if the counterparty defaults, which only happens if the counterparty owes *me* money). This aggregation was relatively straightforward using `numpy.sum` and conditional masking (`np.maximum(value_array, 0)`), maybe taking half a day (4 hours).

Finally, with the array of exposures for each path at each future date, calculating the PFE involves taking a specific percentile across the paths for each date. SciPy's `scipy.stats.scoreatpercentile` or simply `numpy.percentile` worked perfectly here. This was one of the easier parts, taking only an hour or two to implement and verify.

Here's a rough outline of the overall simulation structure in Python:

```python
# Main simulation script idea

# 1. Define market parameters (S0, mu, sigma, corr_matrix)
# 2. Define simulation parameters (T, dt, num_paths)
# 3. Define counterparty trades (list of dicts or objects) - type, strike, maturity, etc.

# Calculate number of time steps
# num_steps = int(round(T / dt)) # Added round() after initial error

# Generate correlated market paths
# market_paths = generate_correlated_paths(...)
# if market_paths is None:
#    exit() # Handle the Cholesky error case

# Array to store portfolio exposure for each path at each time step
# Shape: (num_steps + 1, num_paths)
portfolio_exposures = np.zeros((num_steps + 1, num_paths))

# Loop through each time step
# This is the main loop, needs to be efficient inside
for t in range(num_steps + 1):
    current_time = t * dt # What time is it in the simulation?

    # Get market prices for all paths at the current time step
    current_prices = market_paths[t, :, :] # Shape: (num_paths, num_assets)

    # Calculate value of each trade for all paths at current time
    # This part needs the vectorized pricing functions
    trade_values_across_paths = []
    for trade in counterparty_trades:
        # Need a function like calculate_option_value_vectorized(...)
        # which takes current_prices (or relevant slices), trade params, current_time
        # and returns an array of values for this trade across all paths.
        # This was the core vectorization effort.
        # Example: Assuming a simple call option trade
        # Requires remaining time to maturity: T_rem = trade['maturity'] - current_time
        # Need to handle T_rem <= 0 carefully - option expires, value is intrinsic or 0.
        # This handling inside the loop added complexity.
        if trade['type'] == 'call':
             values = calculate_black_scholes_call_vectorized(
                         S=current_prices[:, trade['asset_idx']], # Get prices for this asset
                         K=trade['strike'], # K is scalar, numpy handles broadcasting
                         T=trade['maturity'] - current_time, # T_rem is scalar per trade, important to calculate correctly
                         R=0.03, # Example rate
                         Sigma=trade['volatility']) # Sigma is scalar
             # Need to handle T_rem <= 0 explicitly in the vectorized func
             # My initial vectorized func didn't handle T_rem=0 well, returned NaNs or errors.
             # Had to add checks: if T_rem <= 0, value is max(S-K, 0).

        # Add values to list (or pre-allocated array)
        trade_values_across_paths.append(values)

    # Sum up values across trades for each path
    # Convert list of value arrays to a single array (num_paths, num_trades)
    if trade_values_across_paths: # Check if there are any trades
        all_trades_values = np.stack(trade_values_across_paths, axis=-1) # Shape (num_paths, num_trades)

        # Calculate exposure for each path: Sum of positive trade values
        # This assumes I am long these trades w.r.t counterparty - cost to replace.
        # If counterparty is long, exposure is max(0, -portfolio_value).
        # Had to be clear on my definition of exposure. Stuck to my long case for simplicity.
        portfolio_value = np.sum(all_trades_values, axis=1) # Sum across trades, shape (num_paths,)
        portfolio_exposures[t, :] = np.maximum(portfolio_value, 0) # Exposure is max(0, positive value)

    # If no trades, exposure is 0
    else:
        portfolio_exposures[t, :] = 0.0


# 4. Calculate PFE at specific horizons (optional, could do for all steps)
# pf_horizons = [0.25, 0.5, 1.0] # e.g., 3 months, 6 months, 1 year
# pfe_results = {}
# for horizon_years in pf_horizons:
#     # Find the closest time step index
#     step_idx = int(round(horizon_years / dt))
#     if step_idx < portfolio_exposures.shape:
#          # Get exposures across all paths at this time step
#          exposures_at_horizon = portfolio_exposures[step_idx, :]
#          # Calculate 95th percentile
#          pfe_value = np.percentile(exposures_at_horizon, 95) # Used numpy.percentile
#          pfe_results[horizon_years] = pfe_value
#     else:
#          pfe_results[horizon_years] = np.nan # Horizon beyond simulation time

# print("PFE Results:", pfe_results)
```

The performance, while not C++ speed, was quite acceptable for typical analysis runs (e.g., 10,000 paths, 250 time steps, handful of trades) on my laptop, thanks to NumPy's vectorization. A run like that usually finished in under a minute. I briefly looked into Numba as a potential next step for speeding up the core loops (like the path generation or the vectorized pricing if it became a bottleneck), but decided the current performance was sufficient for the project's scope and adding Numba felt like another significant time investment I couldn't justify just yet.

Validation was tricky for a complex simulation. I validated the market path generation by plotting sample paths and checking that the statistical properties (mean return, volatility, correlation) of the *simulated* paths matched the input parameters over long runs. I validated the vectorized pricing against single-option calculators. The final PFE numbers were harder to validate rigorously without a benchmark, but the *shape* of the exposure distribution and the PFE curve over time made intuitive sense – PFE often rises, then falls as trades approach maturity.

Looking back, this project took around three to four weeks of dedicated evenings and weekends. The biggest lessons were about structuring a larger simulation framework in Python, effectively using NumPy for vectorization across multiple dimensions (paths, assets, trades), and appreciating that performance bottlenecks can shift – sometimes it's the core calculation, sometimes it's the data management or loop structure around it. While the C++ kernel project was about optimizing a single function, this project was about optimizing the flow and computation across a more complex, multi-step process, highlighting the power of Python's scientific stack for such tasks. There's plenty more to add – different asset models, more complex trades, netting agreements, collateral modeling – but having this basic simulation engine is a solid foundation.