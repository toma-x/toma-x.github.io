---
layout: post
title: Volatility Spillover Network Analysis
---

Right, finally getting around to documenting this Volatility Spillover Network project. It's been a bit of a slog, and I’m still not sure I’ve got all the kinks worked out, but it’s definitely been… a learning experience. The initial idea came after that guest lecture on financial econometrics – Dr. Evans mentioned dynamic conditional correlations and it just sounded more interesting than the standard CAPM stuff we’d been doing. I wanted to see if I could model how shocks actually *move* between assets, not just if they're correlated.

My first thought was to look into multivariate GARCH models. I remember flipping through [Bauwens, Laurent, and Jean-Philippe Rombouts' "Multivariate GARCH models: a survey"](https://www.core.ucl.ac.be/services/psfiles/dp06/dp2006_07.pdf) and getting slightly overwhelmed. The VEC and CCC models seemed a bit too simple for what I wanted – capturing asymmetric responses and dynamic dependencies. The BEKK model, from [Engle and Kroner (1995)](https://www.jstor.org/stable/2951676), felt like the right balance of complexity and interpretability, especially for spillover effects. The `H_t = C'C + A'\epsilon_{t-1}\epsilon_{t-1}'A + B'H_{t-1}B` formulation seemed like it directly gave you the shock (A matrix) and volatility (B matrix) transmissions.

Now, the *implementation*. This is where things got tricky. Most existing packages I found were either in R, or if in Python, they used TensorFlow or PyTorch. I’d been meaning to get my hands dirty with [JAX](https://jax.readthedocs.io/en/latest/) for a while, after seeing some impressive benchmarks and hearing about its functional programming style in one of the advanced ML study groups. Plus, the automatic differentiation with `jax.grad` seemed perfect for the complex likelihood function of GARCH models. This project felt like a good excuse to dive in. JAX *only* for the ML part was the goal.

For data, I just grabbed a few years of daily closing prices for a handful of tech stocks and an ETF – say, AAPL, MSFT, NVDA, and SPY – using the `yfinance` library. Standard stuff: calculate log returns `r_t = log(P_t/P_{t-1})`, demean them to get the residuals `epsilon_t` (though some GARCH models estimate the mean equation jointly, I simplified here to focus on volatility).

The core of the BEKK model is estimating the conditional covariance matrix `H_t`. My parameter set `theta` consisted of the elements of `C` (a lower triangular matrix to ensure positive semi-definiteness of `C'C`), `A`, and `B`. The negative log-likelihood for `T` observations and `N` assets, assuming multivariate normality of residuals, is something like `neg_log_likelihood = 0.5 * sum_t (N * log(2*pi) + log(det(H_t)) + epsilon_t' * H_t_inv * epsilon_t)`.

Actually coding this in JAX was a trip. The main challenge was the recursive nature of `H_t`. `jax.lax.scan` turned out to be the way to go for iterating through time steps.

```python
import jax
import jax.numpy as jnp
import optax
from jax.scipy.stats import multivariate_normal

# N_assets would be, say, 4
# N_params_C = N_assets * (N_assets + 1) // 2 # for lower triangular C
# N_params_A = N_assets * N_assets
# N_params_B = N_assets * N_assets

def unpack_params_bekk(params_flat, N_assets):
    # This was fiddly. Had to be careful with indices.
    idx = 0
    C_vec = params_flat[idx : idx + N_assets * (N_assets + 1) // 2]
    idx += N_assets * (N_assets + 1) // 2
    A_flat = params_flat[idx : idx + N_assets**2]
    idx += N_assets**2
    B_flat = params_flat[idx : idx + N_assets**2]

    C = jnp.zeros((N_assets, N_assets))
    C = C.at[jnp.tril_indices(N_assets)].set(C_vec)
    A = A_flat.reshape(N_assets, N_assets)
    B = B_flat.reshape(N_assets, N_assets)
    return C, A, B

def bekk_recursion_step(carry, epsilon_t_minus_1):
    H_prev, C, A, B = carry
    # epsilon_t_minus_1 is a row vector, make it a column vector for outer product
    eps_outer = jnp.outer(epsilon_t_minus_1, epsilon_t_minus_1)
    H_t = C @ C.T + A.T @ eps_outer @ A + B.T @ H_prev @ B
    return (H_t, C, A, B), H_t

# residuals_ts is my (T, N_assets) array of demeaned returns
# initial_H0 could be the unconditional covariance of residuals_ts

def neg_log_likelihood_bekk(params_flat, residuals_ts, initial_H0):
    N_assets = residuals_ts.shape
    C, A, B = unpack_params_bekk(params_flat, N_assets)

    # The scan function takes (carry, x_t)
    # Here, x_t is epsilon_{t-1} effectively.
    # We need residuals_ts to calculate H_1 using H_0 and residuals_ts (as eps_0)
    # Then residuals_ts (as eps_1) to calculate H_2 using H_1 and residuals_ts
    # So, the 'epsilons' fed to scan should be residuals_ts itself.
    # The first H_t calculated will be H_1 using H_0 and eps_0
    
    # Shift residuals to get eps_{t-1} for H_t
    # For H_1, we use eps_0. For H_T, we use eps_{T-1}.
    # So the `epsilons_for_scan` are residuals_ts[:-1] if H_t depends on eps_{t-1}
    # But my `bekk_recursion_step` takes epsilon_t_minus_1 as input for H_t
    # My residuals_ts are (T x N), eps_t = residuals_ts[t]
    
    # Let's align: H_t uses eps_{t-1}. So, when calculating H_for_eps_t, use eps_{t-1}
    # The likelihood uses H_t and eps_t.
    
    _, H_ts = jax.lax.scan(
        bekk_recursion_step,
        (initial_H0, C, A, B),
        residuals_ts[:-1] # eps_0 to eps_{T-2} to compute H_1 to H_{T-1}
    )
    # This produces H_1, H_2, ..., H_{T-1}
    # I need to align H_t with epsilon_t for the likelihood
    # So H_ts from scan corresponds to H for residuals_ts[1:]

    # For H_0, it is not used in likelihood directly.
    # H_1 corresponds to residuals_ts (if we define t=0 as first observation)
    # Let's redefine: epsilons for scan are eps_{t-1}
    # H_t = f(H_{t-1}, eps_{t-1})
    # L_t = f(H_t, eps_t)
    
    # This part was super confusing, getting the indices right.
    # Let scan compute H_t based on eps_{t-1}
    # Initial carry: (H_0, C, A, B)
    # Input to scan: eps_0, eps_1, ..., eps_{T-1} (this is residuals_ts)
    # Output H_ts will be: H_1, H_2, ..., H_T
    
    initial_carry = (initial_H0, C, A, B)
    # `epsilons_for_scan` should be `residuals_ts`. Each `epsilon_t_minus_1` in the scan function
    # will actually be `residuals_ts[t]` which is `epsilon_t`.
    # So, the model `H_t = C'C + A' eps_t eps_t' A + B' H_{t-1} B`
    # Wait, no, standard BEKK is `H_t` depends on `eps_{t-1}`.
    # So, `epsilons_for_scan` should be `residuals_ts[:-1, :]` which are eps_0...eps_{T-2}
    # This would generate H_1...H_{T-1}.
    # And the likelihood `sum_t log L(eps_t | H_t)`. So I need H_t for eps_t.
    # If scan gets eps_0 ... eps_{T-2}, it produces H_1 ... H_{T-1}.
    # These H_t's (H_1 to H_{T-1}) would be used with eps_1 to eps_{T-1}.
    # What about eps_0? It seems one observation is lost or H_0 is used for it.
    # Standard practice: sum likelihood from t=1 to T. H_1 is based on H_0 (fixed) and eps_0.
    
    # Let's assume residuals_ts = [eps_0, eps_1, ..., eps_{T-1}]
    # `scan_inputs` = residuals_ts (as a proxy for eps_{t-1}, so it's shifted)
    # The first `epsilon_t_minus_1` passed to `bekk_recursion_step` will be `residuals_ts` (eps_0).
    # The first H_t computed will be H_1.
    # So `all_H_t` will be [H_1, ..., H_T] if `scan_inputs` is `residuals_ts`.
    
    initial_params_for_scan = (initial_H0, C, A, B)
    _, H_t_series = jax.lax.scan(bekk_recursion_step, initial_params_for_scan, residuals_ts)
    # H_t_series now contains [H_1, H_2, ..., H_T] where H_t is based on eps_{t-1}
    # (where eps_{t-1} was taken from residuals_ts[t-1] essentially, and H_1 used initial_H0 and residuals_ts as eps_0)

    # The log_prob needs H_t and eps_t.
    # So, H_t_series[i] is H_{i+1} and should be paired with residuals_ts[i+1] (eps_{i+1})
    # This is still messy.
    
    # Simpler: define L_t(params | F_{t-1}). F_{t-1} gives H_t. Then use eps_t.
    # Let H_t be computed based on eps_{t-1}.
    # Then likelihood term for time t is `log_pdf(residuals_ts[t] | mean=0, cov=H_t)`.
    # `H_t_series` as computed by scan above, if `residuals_ts` are `eps_0, ..., eps_{T-1}`:
    # `H_t_series` is `H_1` (uses `eps_0`)
    # `H_t_series[T-1]` is `H_T` (uses `eps_{T-1}`)
    # This means `H_t_series[i]` is `H_{i+1}` (uses `eps_i`).
    # So I need to pair `H_t_series[i]` with `residuals_ts[i+1]` (eps_{i+1}).
    # This means my sum starts effectively from the second residual observation.
    # Or, I need to be very careful about the definition of `H_t` from scan.

    # Let's re-think `scan` for GARCH log-likelihood. A common pattern:
    # `scan_fn(carry, eps_t)` where `carry` is `H_t`. `eps_t` is `residuals_ts[t]`.
    # `scan_fn` computes `log_like_term_t` using `H_t` and `eps_t`, then updates `H_t` to `H_{t+1}` using `eps_t`.
    
    def ll_scan_fn(carry_H_prev, eps_t): # eps_t = residuals_ts[t]
        # carry_H_prev is H_t (conditional on info up to t-1)
        # C,A,B are fixed for the whole sum, so they shouldn't be in carry for this type of scan
        # They should be passed as part of the `params_flat` closure.
        
        # Okay, back to the first approach with `bekk_recursion_step` producing H_ts
        # This is simpler, `H_ts` are `[H_1, ..., H_T]` where `H_t` uses `eps_{t-1}` (and `H_0` for `H_1`).
        # `eps_input_for_scan` should be `residuals_ts[:-1]` to generate `H_1...H_{T-1}`.
        # Then `log_det_H` and `quad_form` should use `residuals_ts[1:]` and these generated `H_s`.

        # For now, let's assume H_t_series contains H_t for each epsilon_t
        # For a simpler first pass, I might have made a mistake in aligning H_t and eps_t
        # but the optimizer would still try to do *something*.
        
        # This was a major headache. Let's use the structure from a JAX GARCH example I found on a forum once.
        # The likelihood term at time t uses H_t (covariance for eps_t) and eps_t.
        # H_t is calculated based on H_{t-1} and eps_{t-1}.

        def single_step_log_likelihood(h_prev, eps_tm1_and_eps_t):
            eps_tm1, eps_t = eps_tm1_and_eps_t # eps_tm1 is used to calculate H_t, eps_t for likelihood
            # C, A, B are from the outer scope (neg_log_likelihood_bekk)
            eps_outer = jnp.outer(eps_tm1, eps_tm1)
            h_curr = C @ C.T + A.T @ eps_outer @ A + B.T @ h_prev @ B
            
            # Ensure H is positive definite - this is the real problem.
            # Adding a small jitter in case of numerical issues. Not ideal.
            # h_curr = h_curr + jnp.eye(N_assets) * 1e-8 
            
            log_det_h = jnp.linalg.slogdet(h_curr) # is sign, is log-abs-det
            # If not PD, log_det_h could be -inf or NaN if slogdet fails.
            # And inv(h_curr) would fail.
            
            # Forcing PD for C'C by C being lower triangular is good.
            # But A and B terms can still make H_t not PD.
            # This is a known issue with BEKK estimation. Some use constrained optimization.
            # I am just using `optax.adam` so no explicit constraints.
            
            inv_h_curr = jnp.linalg.inv(h_curr)
            quad_form = eps_t.T @ inv_h_curr @ eps_t
            
            log_like_term = N_assets * jnp.log(2 * jnp.pi) + log_det_h + quad_form
            return h_curr, log_like_term # new H becomes carry, log_like_term is output per step

        # We need (eps_{t-1}, eps_t) pairs.
        # eps_tm1_series = residuals_ts[:-1]
        # eps_t_series   = residuals_ts[1:]
        # This means we sum from t=1 to T-1 (if T is original length). Or define T as T-1 effectively.
        eps_pairs = jnp.stack([residuals_ts[:-1], residuals_ts[1:]], axis=1)

        # The first H for the scan (h_initial_for_scan) is H_1, calculated from H_0 and eps_0.
        eps0_outer = jnp.outer(residuals_ts, residuals_ts)
        h1_for_scan_init = C @ C.T + A.T @ eps0_outer @ A + B.T @ initial_H0 @ B
        
        # The scan will iterate over (eps_1,eps_2), (eps_2,eps_3) ... (eps_{T-2},eps_{T-1})
        # Using h1_for_scan_init as the H for the first (eps_0, eps_1) pair.
        # This is getting complicated. Let's assume residuals are eps_1, ..., eps_T.
        # H_1 is calculated using initial_H0 and initial_eps0 (e.g. zeros or avg).
        
        # Let's try again with a simpler view, assuming H_t is precomputed using `bekk_recursion_step`
        # and then fed into the likelihood summation. This is what I recall doing eventually.
        # So, `H_t_series = [H_1, ..., H_T]` where H_t depends on eps_{t-1} (and residuals_ts is eps_0)
        
        # `H_dynamic_list` are H_1, H_2, ..., H_T, computed using eps_0, ..., eps_{T-1}
        _, H_dynamic_list = jax.lax.scan(bekk_recursion_step, 
                                         (initial_H0, C, A, B), 
                                         residuals_ts) # residuals_ts are eps_0,...,eps_{T-1}
        
        # Now, `H_dynamic_list[t]` is `H_{t+1}`, which is the covariance for `eps_{t+1}`.
        # This seems to be a common off-by-one source of pain.
        # If residuals_ts are `e_0, e_1, ..., e_{T-1}`
        # H_dynamic_list are `H_1(e_0), H_2(e_1), ..., H_T(e_{T-1})`
        # The log-likelihood term for `e_t` uses `H_t`.
        # So `residuals_ts[t]` (which is `e_t`) should use `H_dynamic_list[t]` (which is `H_{t+1}(e_t)`). No this is not right.
        
        # One final attempt at clarity for the loop:
        # H_t depends on epsilon_{t-1}. The likelihood term for epsilon_t uses H_t.
        # So, loop t from 1 to T (num_obs).
        # H_1 uses H_0 (unconditional) and epsilon_0.
        # L_1 uses H_1 and epsilon_1.
        # If `residuals_ts` = [eps_1, ..., eps_T]. Need eps_0.
        # Assume `residuals_ts` = [eps_0, eps_1, ..., eps_{T-1}] to make it easy. T = number of observations.
        # `eps_for_H_calc` = `residuals_ts[:-1]` (i.e., eps_0 to eps_{T-2})
        # `H_list_for_likelihood` = `scan(..., eps_for_H_calc)` -> gives [H_1, ..., H_{T-1}]
        # `eps_for_likelihood` = `residuals_ts[1:]` (i.e., eps_1 to eps_{T-1})
        # Then sum log_prob(eps_for_likelihood[i] | cov=H_list_for_likelihood[i])

        eps_for_H_update = residuals_ts[:-1] # eps_0 to eps_{T-2}
        eps_for_logL = residuals_ts[1:]      # eps_1 to eps_{T-1}

        # Compute H_1, ..., H_{T-1} based on eps_0, ..., eps_{T-2} and H_0
        _, H_forecasts = jax.lax.scan(bekk_recursion_step, (initial_H0, C, A, B), eps_for_H_update)
        
        # Now, H_forecasts[i] is H_{i+1}, which is the covariance for eps_{i+1} (eps_for_logL[i])
        
        # Vectorized computation of log_likelihood terms:
        # `jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)`
        # `x` would be `eps_for_logL` (shape M x N_assets, where M=T-1)
        # `cov` would be `H_forecasts` (shape M x N_assets x N_assets)
        # `mean` would be `jnp.zeros(N_assets)`
        
        # `vmap` over the time dimension (M observations)
        log_pdf_terms = jax.vmap(lambda eps, H: multivariate_normal.logpdf(eps, mean=jnp.zeros(N_assets), cov=H))(
            eps_for_logL, H_forecasts
        )
        
        # Check for NaNs or Infs which can happen if H_t is not PD.
        # A Prof once said: "If you get NaNs, your model or your data is telling you something."
        # Forcing positive definiteness is hard. For BEKK, C must be triangular. A and B can be anything.
        # Sometimes, the sum B.T @ H_prev @ B + A.T @ eps_outer @ A can lead to non-PD matrices.
        # If log_pdf_terms contains NaN, the sum will be NaN.
        total_log_likelihood = jnp.sum(log_pdf_terms)
        
        # If total_log_likelihood is NaN, return a very large number for minimization.
        # This helps the optimizer steer away from bad regions.
        # This was a tip from a StackOverflow thread I can't find anymore.
        return jnp.where(jnp.isnan(total_log_likelihood), jnp.inf, -total_log_likelihood)

    # Optimizer
    # initial_params_flat = ... (randomly initialize, or from some simpler model)
    # I started with small random values for A and B elements, and C from cholesky of unconditional cov.
    # value_and_grad_fn = jax.value_and_grad(neg_log_likelihood_bekk)
    # optimizer = optax.adam(learning_rate=1e-3)
    # opt_state = optimizer.init(initial_params_flat)

    # for i in range(num_iterations):
    #   loss_val, grads = value_and_grad_fn(params_flat, residuals_ts, initial_H0)
    #   updates, opt_state = optimizer.update(grads, opt_state, params_flat)
    #   params_flat = optax.apply_updates(params_flat, updates)
    #   if i % 10 == 0:
    #       print(f"Step {i}, Loss: {loss_val}") # Using jax.debug.print is better for inside jit
```

The hardest part, by far, was ensuring positive definiteness of `H_t` at each step. JAX is very good at telling you when you have a `NaN` because your matrix inversion failed or `log(det(H_t))` blew up. I didn't implement any specific constraints on `A` and `B` (like diagonal BEKK or scalar BEKK initially, though I read about them). I just hoped `optax.adam` would find a region where things behaved. Sometimes it worked, sometimes the loss just went to `NaN` and stayed there. Reducing the learning rate, or re-initializing parameters, sometimes helped. I spent a lot of time on [JAX's FAQ on NaNs](https://jax.readthedocs.io/en/latest/errors.html#faq-float64-to-float32-issues-or-nans). Using `jax.debug.print` within the jitted likelihood function was also a lifesaver.

One trick I saw mentioned, but didn't rigorously implement, was adding a tiny diagonal matrix `diag(delta)` to `H_t` at each step to keep it numerically stable, but that feels like cheating the model. Another was to check `jnp.linalg.eigvalsh(H_t)` and if any eigenvalue was non-positive, return a massive loss, but that makes gradients tricky. My `jnp.where(jnp.isnan(total_log_likelihood), jnp.inf, -total_log_likelihood)` was a cruder version of that.

After (eventually) getting the model to converge to *something* that wasn't `NaN`, I had my estimated `C`, `A`, and `B` matrices. The off-diagonal elements of `A` (for shock spillover) and `B` (for volatility spillover) were what I was interested in. For `N` assets, these are `N x N` matrices. `A_ij` (or `A_ji` depending on convention) shows how a shock in asset `j` affects asset `i`'s conditional variance in the next period.

To visualize this, I turned to [NetworkX](https://networkx.org/). I built a directed graph where nodes are the assets. An edge from asset `j` to asset `i` would have a weight derived from `A_ij` and `B_ij`. I tried a few things for edge weights:
1.  Just `abs(A_ij)`.
2.  `A_ij^2` (since `A` enters quadratically in `H_t`). Same for `B`.
I decided to create two separate networks: one for shock spillovers (from `A`) and one for volatility spillovers (from `B`). For instance, for the shock spillover network, an edge `(j,i)` has weight `A_ij^2`.

```python
import networkx as nx
import numpy as np # For initial data, but JAX arrays for model params

# Assume fitted_A_matrix and fitted_B_matrix are the estimated parameters (numpy arrays now)
# asset_names = ['AAPL', 'MSFT', 'NVDA', 'SPY']

# G_shock = nx.DiGraph()
# for i, source_asset in enumerate(asset_names):
#     G_shock.add_node(source_asset) # Ensure all nodes exist
#     for j, target_asset in enumerate(asset_names):
#         if i == j: continue # No self-loops for this visualization
#         # A_ij refers to impact of j on i (if A is A_target_source)
#         # This depends on how A was defined in H_t = ... A' eps eps' A ...
#         # If A_ij means effect from j to i in the matrix, then:
#         weight = fitted_A_matrix[i, j]**2 # Or whatever metric I decided
#         if weight > threshold: # Some threshold to prune weak connections
#             G_shock.add_edge(asset_names[j], asset_names[i], weight=weight, type='shock')
#
# # Similarly for G_volatility using fitted_B_matrix
# G_volatility = nx.DiGraph()
# # ... (similar loop using fitted_B_matrix[i,j]**2)
```

Then, I exported these graphs to `.gexf` format, which [Gephi](https://gephi.org/) handles beautifully. `nx.write_gexf(G_shock, "shock_spillover_network.gexf")`. Playing around with layouts in Gephi (ForceAtlas2 is usually good for this kind of network) and mapping edge weights to thickness or color helped visualize the strongest spillover pathways. It was pretty satisfying to finally see some structure after all the JAX wrestling.

What I learned:
*   Implementing complex econometric models from scratch in JAX is a steep learning curve but very rewarding. `jax.jit` is magic, but debugging it requires patience.
*   The math needs to be *exactly* right. Off-by-one errors in time series indexing for the likelihood function were a nightmare. I probably still have some subtle bugs there.
*   Positive definiteness in multivariate GARCH is a real pain. I didn't solve it robustly; I mostly relied on good initializations and the optimizer not straying too far. A friend doing a PhD in stats mentioned something about reparameterizing the matrices `A` and `B` using matrix logarithms or other techniques to enforce PD, but that was way over my head for this project.
*   BEKK provides direct measures of spillover, which is cool, but the number of parameters explodes with `N_assets^2`. For more than 5-6 assets, full BEKK becomes computationally very expensive. I stuck to 3-4.

If I were to do it again, I'd spend more time researching constrained optimization in JAX, or perhaps look into more numerically stable GARCH variants or parameterizations if sticking with BEKK. Maybe even explore some of the JAX-based probabilistic programming libraries like [NumPyro](https://num.pyro.ai/en/stable/) or [BlackJAX](https://github.com/blackjax-devs/blackjax) to see if they offer more robust ways to handle these models, though that might be overkill.

This write-up is already way too long. But yeah, that was the journey. Lots of dead ends, StackOverflow deep dives, and "why is this NaN?" moments. But the final Gephi graphs were pretty neat to look at.