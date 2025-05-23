---
layout: post
title: GPU-Accelerated Options Pricing via PINNs
---

It's been a while since my last update, mostly because I've been deep in the weeds with a project that's consumed a good chunk of my time: trying to get GPU acceleration working for options pricing using Physics-Informed Neural Networks. The main goal was to see if I could leverage JAX and CUDA to get a speed-up, particularly for some exotic options where traditional Monte Carlo methods can get really slow.

The initial idea came after a financial engineering class where we discussed the computational cost of pricing path-dependent options. Monte Carlo simulations are the workhorse, but for some of the more complex derivatives, or when you need to re-price rapidly for changing market conditions, the time taken can be a real bottleneck. I’d read a few papers on PINNs (Raissi et al.'s work was a starting point) and the concept of baking the governing PDE directly into the neural network's loss function seemed like a really elegant solution, potentially side-stepping the need for discretizing the domain like in traditional PDE solvers or running thousands of simulations.

My first step was to tackle the Black-Scholes PDE for a European call option. The PDE itself isn't too scary:
`∂V/∂t + rS ∂V/∂S + 0.5 σ² S² ∂²V/∂S² - rV = 0`
where `V` is the option price, `S` is the stock price, `t` is time, `r` is the risk-free rate, and `σ` is the volatility. The challenge with PINNs isn't just the PDE, but also encoding the boundary and terminal conditions correctly. For a call option, this means `V(S, T) = max(S - K, 0)` at expiry `T`, `V(0, t) = 0`, and `V(S, t) -> S - K * exp(-r(T-t))` for large `S` (or rather, its derivative `∂V/∂S -> 1` for large S).

I decided to use JAX for this. I'd been wanting a good project to really learn it, and its automatic differentiation capabilities (`jax.grad`) and XLA compilation for GPU/TPU seemed perfect for PINNs. PyTorch was an option, and I'm more familiar with it, but the functional nature of JAX and tools like `vmap` and `pmap` felt more aligned with the mathematical operations I'd need.

Getting the environment set up with JAX and CUDA on my personal machine was the first hurdle. I spent an evening wrestling with `nvidia-smi` outputs, CUDA toolkit versions, and cuDNN libraries until `jax.devices()` finally showed my GPU. Turns out, the pre-built JAX wheels have specific CUDA version dependencies, and I had to downgrade my drivers a bit. A few StackOverflow threads later, it was sorted.

Then came the actual PINN implementation. I started with a simple multi-layer perceptron (MLP) using `flax.linen` since it integrates nicely with JAX. Something like:

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax # For the optimizer

class SimplePINN(nn.Module):
    num_neurons: int = 64
    num_layers: int = 4

    @nn.compact
    def __call__(self, S, t): # S and t are my inputs
        x = jnp.concatenate([S, t], axis=-1)
        for _ in range(self.num_layers - 1):
            x = nn.Dense(features=self.num_neurons)(x)
            x = nn.tanh(x)
        x = nn.Dense(features=1)(x) # Output is the option price V
        return x

key = jax.random.PRNGKey(0)
dummy_S = jnp.ones((1,1))
dummy_t = jnp.zeros((1,1))
model = SimplePINN()
params = model.init(key, dummy_S, dummy_t)['params']
# print(params) # Just to see the structure initially
```
The network takes stock price `S` and time `t` as inputs (normalized, of course, which I learned the hard way later was crucial) and outputs the option price `V`. I used `tanh` activation functions as I saw that recommended in a few PINN papers for smoother derivatives.

The real brain-twister was constructing the loss function. It's not just about matching data points. The loss has to include:
1.  The PDE residual: How well the network's output `V(S,t)` satisfies the Black-Scholes equation. This involved using `jax.grad` multiple times to get `∂V/∂t`, `∂V/∂S`, and `∂²V/∂S²`.
2.  The boundary/terminal conditions: Ensuring `V(S,T)` is close to `max(S-K,0)`, `V(0,t)` is close to 0, etc.

My first attempt at the PDE residual function was a bit naive:

```python
# V_model is a function that takes params, S, t and returns V
# This is a simplified conceptual version of what I was trying
def pde_residual(params, S, t, K, r, sigma, V_model_apply):
    # Get V and its derivatives
    V_and_grad_S = jax.value_and_grad(lambda s_val: V_model_apply({'params': params}, s_val, t).squeeze(), argnums=0)
    V_val, dVdS_val = V_and_grad_S(S)
    
    # For second derivative of S, and first of t
    # This got complicated quickly. I had to be careful about how grad was applied.
    # My initial attempts here were full of shape mismatches or wrong derivative calculations.
    # d2VdS2_val = jax.grad(lambda s_val: V_and_grad_S(s_val))(S) # This is conceptually what I wanted for d2V/dS2
    
    # To get dV/dt, had to use grad w.r.t. the 't' argument
    # dVdt_val = jax.grad(lambda t_val: V_model_apply({'params': params}, S, t_val).squeeze(), argnums=0)(t)

    # Placeholder for actual derivative calculations which took a lot of debugging
    # For instance, jax.jacfwd and jax.jacrev for Hessians, or multiple grads.
    # The actual implementation involved creating helper functions for each derivative.
    
    # Let's assume I got these derivatives correctly after some pain:
    # V, dV_dt, dV_dS, d2V_dS2 = compute_derivatives_correctly(params, S, t, V_model_apply)
    
    # This is where I define the actual Black-Scholes operator
    # residual = dV_dt + r * S * dV_dS + 0.5 * sigma**2 * S**2 * d2V_dS2 - r * V
    # return residual

    # Actually computing derivatives for the blog post:
    # Need to ensure S and t are treated as separate inputs for differentiation
    # V_fn = lambda s_scalar, t_scalar: V_model_apply({'params': params}, s_scalar.reshape(1,1), t_scalar.reshape(1,1))

    # dV/dt
    # This requires V_fn to handle scalar inputs for grad to work easily here for illustration
    # For batched inputs, vmap is the way but makes the example longer.
    # Let's assume S and t are single points (collocation points) for this snippet.
    
    V = V_model_apply({'params': params}, S, t)

    # Using jax.grad for partial derivatives.
    # Need to be careful with function signatures for jax.grad.
    # Often easier to define small lambda functions.
    get_V_for_grad_S = lambda s_val, t_val: V_model_apply({'params': params}, s_val, t_val)
    get_V_for_grad_t = lambda s_val, t_val: V_model_apply({'params': params}, s_val, t_val)

    dVdS = jax.grad(get_V_for_grad_S, argnums=0)(S, t)
    # For d2VdS2, we differentiate dVdS with respect to S again.
    # This requires dVdS to be a function of S.
    get_dVdS_for_grad = lambda s_val, t_val: jax.grad(get_V_for_grad_S, argnums=0)(s_val, t_val)
    d2VdS2 = jax.grad(get_dVdS_for_grad, argnums=0)(S, t)
    
    dVdt = jax.grad(get_V_for_grad_t, argnums=1)(S, t)
    
    # Black-Scholes PDE residual
    # Note: S might need to be jnp.squeeze(S) if it has an extra dim from batching
    # For this example, assuming S, t, V, and derivatives are conformable scalars or squeezed arrays.
    # S_val = S.item() # if S is a 0-dim array
    residual_val = dVdt + r * S * dVdS + 0.5 * sigma**2 * S**2 * d2VdS2 - r * V
    return jnp.mean(residual_val**2) # Mean squared error for the residual
```
The derivative calculations were finicky. `jax.grad` is powerful, but you need to be precise about `argnums` and how your function handles inputs, especially when going for higher-order derivatives or derivatives with respect to multiple variables. My first few attempts had `jax.grad` complaining about non-scalar outputs from functions I was trying to differentiate, or I was accidentally taking gradients with respect to the wrong thing. I spent a good amount of time in the JAX documentation on automatic differentiation and looking at examples online. One key insight was to ensure the function being differentiated returned a scalar if I wanted a simple gradient; for Jacobians/Hessians, the approach changes.

The total loss function then became a weighted sum:
`loss = w_pde * mse_pde_residuals + w_bc * mse_boundary_conditions + w_ic * mse_initial_condition`
Getting these weights (`w_pde`, `w_bc`, `w_ic`) right was another week of trial and error. If `w_pde` was too large, the network would learn the PDE in regions that didn't matter and ignore the crucial boundary values. If `w_bc` was too large, it would fit the boundaries perfectly but produce garbage in the interior. There's probably a more systematic way to tune these, but I mostly did it by feel and observing the training dynamics.

Normalization of inputs (`S` and `t`) and outputs (`V`) was also something I initially overlooked. Stock prices can range from tens to hundreds, while time `t` (as fraction of year to expiry) is usually between 0 and 1. Without normalizing these to, say, `[-1, 1]` or `[0, 1]`, the network had a really hard time learning. Gradients would either vanish or explode. A simple min-max scaling on `S` based on a reasonable domain (e.g., `0` to `2*K`) and `t` (e.g., `0` to `T`) made a significant difference.

Training was done with `optax.adam`. The training loop itself is fairly standard JAX: define a `@jax.jit` decorated `train_step` function that computes the loss and gradients, then updates the parameters.

```python
# Simplified training step structure
@jax.jit
def train_step(params, opt_state, S_colloc, t_colloc, S_boundary, t_boundary, V_boundary_target, K, r, sigma):
    def loss_fn_wrapper(p):
        # Collocation points for PDE loss
        # This is where my pde_residual function would be called
        # For brevity, I am not re-defining pde_residual here, but it uses V_model_apply with 'p'
        # loss_pde = calculate_pde_loss(p, S_colloc, t_colloc, K, r, sigma, model.apply) 
        
        # Boundary condition loss
        # V_pred_boundary = model.apply({'params': p}, S_boundary, t_boundary)
        # loss_boundary = jnp.mean((V_pred_boundary - V_boundary_target)**2)
        
        # Placeholder for actual loss calculation logic
        loss_pde = 0.0
        loss_boundary = 0.0
        # A real implementation would calculate these based on network output and targets/PDE form
        # Example of calling the V_model_apply within the loss function:
        V_pde = model.apply({'params': p}, S_colloc, t_colloc)
        # ... then compute derivatives of V_pde to get loss_pde ...
        
        # For boundary, let's say we have terminal condition points (S_term, T_term)
        # V_terminal_pred = model.apply({'params': p}, S_term, T_term_vals)
        # V_terminal_actual = jax.nn.relu(S_term - K) # max(S-K, 0)
        # loss_terminal = jnp.mean((V_terminal_pred - V_terminal_actual)**2)

        # total_loss = w_pde * loss_pde + w_bc * loss_boundary # plus other conditions
        total_loss = loss_pde + loss_boundary # Assume weights are 1 for simplicity here
        return total_loss

    grad_loss = jax.grad(loss_fn_wrapper)(params)
    updates, new_opt_state = optimizer.update(grad_loss, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    # return new_params, new_opt_state, total_loss # total_loss would be returned by loss_fn_wrapper itself
    current_loss = loss_fn_wrapper(params) # get current loss value before updating params
    return new_params, new_opt_state, current_loss


# optimizer = optax.adam(learning_rate=1e-3)
# opt_state = optimizer.init(params)

# Training loop
# for epoch in range(num_epochs):
#   # Generate new collocation points, boundary points etc. for this epoch
#   # S_col, t_col = generate_collocation_points(...)
#   # S_bnd, t_bnd, V_target_bnd = generate_boundary_points(...)
#   params, opt_state, loss_val = train_step(params, opt_state, S_col, t_col, S_bnd, t_bnd, V_target_bnd, K, r, sigma)
#   if epoch % 100 == 0:
#     print(f"Epoch {epoch}, Loss: {loss_val.item()}")
```

The `generate_collocation_points` function would typically sample points randomly from the `(S, t)` domain. The `@jax.jit` compilation was a massive speed-up for the training step. My first few runs without it were painfully slow, as each gradient calculation and update was being interpreted in Python. Once `jit` was applied, it flew, especially because JAX could compile the whole graph down to efficient XLA operations for the GPU.

One critical "aha!" moment was with `jax.vmap`. My initial PDE residual and loss calculations were done point-wise using Python loops over batches of collocation points. This was terribly inefficient and didn't leverage the GPU properly. Refactoring to use `jax.vmap` to vectorize the computation of derivatives and residuals across all collocation points at once was key. The GPU utilization shot up, and training times dropped dramatically. It took a bit of thought to make my functions `vmap`-compatible, mainly ensuring they operated on individual samples and then letting `vmap` handle the batching.

For comparison, I implemented a straightforward Monte Carlo pricer for European options. While simple to code, getting high accuracy required a large number of simulation paths, and it was noticeably slower than the *inference time* of the trained PINN. The PINN takes time to *train*, yes, but once trained, it can price options for new `(S,t)` pairs (within its trained domain) almost instantaneously.

The real test was moving towards an exotic option. I focused on a type of Asian option (average price option). The PDE for this is more complex as it involves an additional state variable for the running average, effectively increasing the dimensionality. This is where Monte Carlo methods start to become even more computationally intensive if high accuracy is needed. My PINN approach, extended to this higher-dimensional PDE, started to show a more significant speed advantage during inference after the initial (and admittedly more complex) training phase. For specific configurations of this Asian option, the PINN was considerably faster at generating prices across a range of current stock prices and volatilities compared to re-running the MC simulation for each point.

It's not a perfect solution by any means. The training of PINNs can be finicky, highly sensitive to network architecture, optimizer choice, learning rate, and those pesky loss weights. Generalization outside the strict domain and parameter range (like `r`, `sigma`) it was trained on can also be an issue without re-training or using techniques like parameter-informed PINNs, which is a whole other level of complexity.

But as a learning experience, this project was fantastic. Wrestling with JAX's functional paradigm, CUDA setup, the math of PINNs, and then seeing the speed-ups on the GPU for a problem I found interesting was incredibly rewarding. There were many moments of "why is this not working?" followed by small breakthroughs. For example, debugging the derivative computations often involved printing out intermediate `jax.jvp` or `jax.vjp` results to ensure the directional derivatives made sense, which felt like stepping through the calculus manually.

Next steps? Perhaps trying to apply this to American options, which introduces free boundaries and makes the problem even harder (often formulated as a linear complementarity problem). Or exploring more advanced PINN architectures and adaptive sampling techniques for collocation points, as random sampling might not always be the most efficient. For now, though, I'm pretty pleased with how this turned out.