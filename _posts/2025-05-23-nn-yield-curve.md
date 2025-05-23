---
layout: post
title: Arbitrage-Free Yield Curve Smoothing with Neural Networks
---

## Arbitrage-Free Yield Curve Smoothing with Neural Networks: A Deep Dive

For the past few months, I've been neck-deep in a project that sits at the intersection of quantitative finance and machine learning: attempting to build a more robust yield curve smoothing model. The goal was to develop an arbitrage-free Nelson-Siegel model, but with a modern twist using PyTorch neural networks, and then train it on real-world European Central Bank (ECB) swap data. The main motivation was to see if I could achieve better fixed income pricing accuracy, smoother curves, and ultimately, reduced hedging errors when compared against established benchmarks like QuantLib. It’s been a journey, to say the least, with plenty of head-scratching moments and a few small victories along the way.

### The Starting Point: Why Nelson-Siegel and Why Arbitrage-Free?

The Nelson-Siegel model is a popular choice for yield curve modeling because it's relatively simple and provides a good fit for typical yield curve shapes. However, the standard implementation doesn't inherently guarantee an arbitrage-free curve. This is a big deal because arbitrage opportunities, even if they're just artifacts of the model, can lead to serious mispricings and flawed hedging strategies. So, the "arbitrage-free" part was non-negotiable.

My initial plan was to implement the Svensson model, an extension of Nelson-Siegel, as it can capture more complex curve shapes with its two hump-related components. However, ensuring arbitrage-free conditions for Svensson through parameter constraints is trickier than for the basic Nelson-Siegel. I found a few papers that discussed constrained optimization for Nelson-Siegel parameters to ensure positive forward rates, which is a key condition for an arbitrage-free curve. This seemed like a more manageable starting point.

### The Data Hurdle: ECB Swap Data

Before any modeling, there's data. I decided to use ECB swap data because it's publicly available and represents a significant chunk of the Euro interest rate market. Getting the data wasn't too bad – the ECB Statistical Data Warehouse is fairly straightforward. The challenge was cleaning and preprocessing it. Swap rates come with different tenors, and not all tenors are quoted every day. There were gaps, and some outright outliers that looked suspicious.

My preprocessing pipeline in Python involved:
1.  Filtering for EUR-denominated interest rate swaps.
2.  Pivoting the data to get a matrix of tenors vs. dates.
3.  Interpolating missing short-end rates using a simple linear interpolation if the gap wasn't too large. For longer gaps, I initially just dropped those days, but this reduced my dataset size significantly. I later revisited this to use a cubic spline for interpolation on days where at least a few key tenors were present, which felt like a reasonable compromise between accuracy and data retention.
4.  Converting swap rates to zero-coupon yields. This step itself is an iterative process (bootstrapping), and doing it consistently was crucial. I spent a good week debugging my bootstrapping script because the long-end of the curve kept blowing up. Turns out, I had a subtle error in how I was calculating discount factors for semi-annual coupon bonds from annualized swap rates. A classic case of thinking I understood the formula but messing up the implementation details.

### The Core Model: Nelson-Siegel in Python

I started by implementing the standard Nelson-Siegel model. The formula for the zero-coupon yield *y(t)* at maturity *t* is:

*y(t) = β₀ + β₁ * ( (1 - exp(-t/τ₁)) / (t/τ₁) ) + β₂ * ( ( (1 - exp(-t/τ₁)) / (t/τ₁) ) - exp(-t/τ₁) )*

(Actually, I used the Diebold-Li variation where τ₁ is fixed and you optimize for λ, which relates to τ. The parameters become β₀ (level), β₁ (slope), and β₂ (curvature).)

My first attempt was a direct translation of the formula into Python, using `scipy.optimize.minimize` to fit the β parameters and τ (or λ) to the observed market yields for each day.

```python
import numpy as np
from scipy.optimize import minimize

# Simplified Nelson-Siegel yield function
def nelson_siegel_yield(params, t):
    beta0, beta1, beta2, tau1 = params
    if tau1 < 1e-6: # Prevent division by zero or extremely small tau
        tau1 = 1e-6
    term1 = (1 - np.exp(-t / tau1)) / (t / tau1)
    term2 = term1 - np.exp(-t / tau1)
    return beta0 + beta1 * term1 + beta2 * term2

# Objective function for minimization
def objective_ns(params, market_tenors, market_yields):
    model_yields = nelson_siegel_yield(params, market_tenors)
    return np.sum((model_yields - market_yields)**2)

# Example usage for a single day's curve
# market_tenors = np.array([0.25, 0.5, 1, 2, 5, 10, 30]) # in years
# market_yields = np.array([0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01])
# initial_params_ns = [0.01, -0.01, -0.01, 1.5]
# result = minimize(objective_ns, initial_params_ns, args=(market_tenors, market_yields), method='SLSQP')
# optimal_params_ns = result.x
```
This worked okay for fitting individual curves, but the arbitrage-free constraint was still missing. The SLSQP method in `scipy.optimize.minimize` allows for constraints, and my first approach was to try and constrain the parameters directly based on conditions for positive forward rates derived from the Nelson-Siegel formula. This got complicated quickly, and the constraints often led to the optimizer getting stuck or producing unrealistic parameters.

### Enter PyTorch: The Neural Network Enhancement

This is where the idea of using neural networks came in. Instead of just fitting the parameters β₀, β₁, β₂, and τ independently for each day using `scipy.optimize`, what if a neural network could learn to output these parameters (or some transformation of them) in a way that's more stable and conducive to arbitrage-free conditions? Or, even better, what if the network could directly learn a representation that helps ensure positive forward rates?

I decided to have the neural network predict the Nelson-Siegel parameters (βs and τ). The input to the network for each day would be the set of observed market yields for that day.

My first PyTorch model was quite simple: a few fully connected layers.

```python
import torch
import torch.nn as nn

class NelsonSiegelNet(nn.Module):
    def __init__(self, num_market_tenors, num_ns_params=4):
        super(NelsonSiegelNet, self).__init__()
        self.layer1 = nn.Linear(num_market_tenors, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, num_ns_params)
        # For tau, ensure it's positive. Applying softplus or exp later.
        # For betas, their ranges are less constrained initially.

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        params_pred = self.output_layer(x)
        
        # We need to be careful with the parameter 'tau'
        # It must be positive. Let's assume the last output parameter is tau.
        # A common trick is to pass it through a Softplus or exp to ensure positivity.
        # Let's say params_pred has [beta0, beta1, beta2, raw_tau]
        # We can transform raw_tau in the training loop or here.
        # For now, I'll just output them and handle transformation + NS calculation outside.
        return params_pred
```

The initial idea was that the network would learn the mapping from noisy market rates to "optimal" Nelson-Siegel parameters. The loss function was initially just Mean Squared Error (MSE) between the yields generated by the network's predicted NS parameters and the actual market yields.

### The Arbitrage-Free Constraint with Neural Networks

This was the hardest part. How do you make a neural network output arbitrage-free curves? Just penalizing negative forward rates in the loss function is a start, but it can be a bit like playing whack-a-mole.

I spent a lot of time reading. One approach I found promising was to not predict the Nelson-Siegel parameters directly, but to predict parameters of a *different* representation of the discount curve that is inherently arbitrage-free, and then map that back to yields. For instance, some papers propose parameterizing the instantaneous forward rate curve in a way that ensures positivity, e.g., by modeling log(forward rate) or (forward rate)^2.

Given my existing Nelson-Siegel framework, I opted for a hybrid approach initially:
1.  The NN predicts the NS parameters (β₀, β₁, β₂, τ).
2.  The τ parameter (or more precisely λ related to it) was transformed using `torch.exp()` or `torch.nn.functional.softplus` on the raw NN output to ensure it was positive, as τ must be > 0.
3.  The primary loss was still MSE on yields.
4.  **Crucially**, I added a penalty term to the loss function. This penalty was based on calculating a series of short-term forward rates across the curve (e.g., f(t, t+Δt)) and heavily penalizing any that were negative.

Calculating these forward rates from the predicted NS parameters within the PyTorch computation graph was key, so gradients could flow back.
The instantaneous forward rate *f(t)* from Nelson-Siegel is:
*f(t) = β₀ + β₁ * exp(-t/τ₁) + β₂ * (t/τ₁) * exp(-t/τ₁)*

I had to ensure this *f(t)* remained positive for all *t > 0*. Analytically deriving constraints on β₀, β₁, β₂, τ for this is non-trivial and often too restrictive. So, the penalty approach seemed more practical for a neural network.

My training loop looked something like this conceptually:

```python
# Pseudocode for training part
# model = NelsonSiegelNet(num_tenors)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion_mse = nn.MSELoss()

# for epoch in range(num_epochs):
#     for market_data_batch in dataloader:
#         optimizer.zero_grad()
#         observed_yields = market_data_batch # Input to NN
        
#         # NN predicts raw NS parameters
#         raw_ns_params_pred = model(observed_yields) 
        
#         # Transform raw_ns_params_pred to ensure positivity for tau, etc.
#         # For instance, params_pred[:, 3] = torch.exp(raw_ns_params_pred[:, 3]) + 1e-4 # for tau
#         # This transformation is important and I played around with it a lot.
#         # Sometimes softplus was better than exp to avoid explosions.
#         ns_params_pred = transform_raw_params(raw_ns_params_pred) # Placeholder for actual transformation

#         # Calculate model yields using predicted NS params
#         # This involves iterating through tenors and applying the NS formula
#         # Ensure this is all done with PyTorch tensors for autograd
#         model_yields_pred = calculate_ns_yields_from_params_torch(ns_params_pred, tenors_tensor)
        
#         loss_fit = criterion_mse(model_yields_pred, observed_yields)
        
#         # Arbitrage penalty calculation
#         # 1. Derive instantaneous forward rates at several points on the curve
#         #    f(t) = beta0 + beta1*exp(-t/tau) + beta2*(t/tau)*exp(-t/tau)
#         # 2. Penalize if f(t) < 0
#         # This required careful implementation of the forward rate formula using torch tensors.
#         forward_rates_pred = calculate_forward_rates_torch(ns_params_pred, test_forward_tenors_tensor)
#         penalty_arbitrage = torch.mean(torch.relu(-forward_rates_pred)) * penalty_weight 
#         # penalty_weight was a hyperparameter I had to tune carefully. Too high and it wouldn't fit the market. Too low and arbitrage wasn't removed.

#         total_loss = loss_fit + penalty_arbitrage
#         total_loss.backward()
#         optimizer.step()
```

One of the "aha!" moments was realizing how sensitive the training was to the `penalty_weight`. Initially, I set it too low, and the network happily produced arbitrage opportunities if it meant a slightly better MSE on the yields. Cranking it up helped, but then the fit to market yields sometimes suffered. It was a balancing act. I also found that normalizing the input market yields (e.g., to have zero mean and unit variance over the batch, or over the whole dataset) helped the network train much more stably. Without it, the gradients for some parameters would just explode.

I remember one specific week where my forward rates, especially at the very short end, kept dipping negative despite the penalty. I was checking my forward rate formula implementation against multiple textbooks. The issue wasn't the formula itself, but rather the numerical stability when `t` was very small in `(t/τ₁) * exp(-t/τ₁)`. Adding a small epsilon to `τ₁` in denominators or carefully managing the `exp` terms helped. I also found a post on a quantitative finance forum (I think it was Wilmott or Quant SE) discussing numerical stability in Svensson model implementations, and some of the tricks mentioned there for handling near-zero maturities were adaptable.

### Training, Tuning, and "Why is it still not working?"

Training these models took time. I was running this on my personal machine, which has a decent GPU, but iterating through different architectures, hyperparameters (learning rate, batch size, number of layers, `penalty_weight` for arbitrage) was a grind.

*   **Overfitting:** My initial, more complex NNs (with more layers/neurons) were very prone to overfitting. The curves would fit the training data perfectly but produce wild oscillations between known tenor points. Adding dropout and L2 regularization to the NN weights helped a bit, but simplifying the network and focusing on a robust arbitrage penalty was more effective.
*   **Convergence:** Sometimes the loss would plateau, and it wasn't clear if it was a bad local minimum or if the learning rate was too high/low. I implemented a learning rate scheduler (`torch.optim.lr_scheduler.ReduceLROnPlateau`) which helped in squeezing out a bit more performance once things started to slow down.
*   **The "Smoothness" Factor:** While the arbitrage penalty helped with positive forward rates, it didn't always guarantee *smooth* forward rates. The raw Nelson-Siegel can sometimes produce a "kink" if the parameters aren't well-behaved. My NN-driven parameters sometimes inherited this. I experimented with adding another penalty term for the second derivative of the forward curve (to encourage smoothness), but this made the loss function even more complex and harder to tune. I eventually decided to focus primarily on the arbitrage-free aspect and accept whatever smoothness the NS structure with positive forwards gave me.

A breakthrough came when I started looking at the distribution of the Nelson-Siegel parameters themselves. For some days, the optimizer in the pure `scipy` version would give very extreme values for, say, β₂, especially if the curve was flat. The NN seemed to learn to regularize these parameters implicitly, producing more stable sets of βs and τ over time, which generally led to more well-behaved curves.

### Benchmarking Against QuantLib

The true test was comparing my model against something established. QuantLib is an obvious choice for a C++ based open-source library, but I used its Python bindings (`QuantLib-Python`) for easier comparison within my Python environment.

I used QuantLib's implementation of Nelson-Siegel and Svensson as benchmarks. The metrics were:
1.  **Root Mean Squared Error (RMSE)** against observed market yields.
2.  **Number of arbitrage violations:** I checked this by deriving forward rates from the fitted curves and counting how many were negative.
3.  **Hedging Errors:** This was more involved. I simulated pricing a portfolio of simple interest rate swaps using my model's curve vs. QuantLib's curve, then looked at one-day P&L discrepancies if the market moved slightly. The idea was that a "better" curve should lead to more stable hedge ratios and smaller unexplained P&L. This part is still a bit experimental, as setting up a realistic hedging simulation is complex.
4.  **Smoothness:** Visually inspecting the forward rate curves, and also looking at the sum of squared second derivatives as a rough quantitative measure.

The results were encouraging. My NN-enhanced Nelson-Siegel generally achieved:
*   RMSE comparable to QuantLib's standard Nelson-Siegel, sometimes slightly better, especially on more "unusual" curve days.
*   Significantly fewer arbitrage violations. The penalty term, when tuned correctly, was quite effective. There were still occasional tiny negative forward rates in the very far future (e.g., 50+ years), which were likely numerical artifacts, but in the main part of the curve (up to 30 years), it was much cleaner than the unconstrained `scipy` fits.
*   Smoother *forward rate* curves compared to a basic NS fit that might have had to contort itself to fit market prices without an explicit arbitrage consideration.
*   For hedging errors, the results were more nuanced. On average, my model seemed to provide slightly more stable hedge parameters for short-dated swaps, but it wasn't a universal win. This needs more investigation.

I wouldn't say my model blew QuantLib out of the water across the board. QuantLib's solvers are highly optimized and robust. However, for the specific goal of enforcing arbitrage-free conditions more directly within the fitting process using a data-driven approach, the neural network showed real promise. It learned to produce parameters that were inherently more stable.

### The C# Interface: Thinking Ahead

One of the potential applications I had in mind was integrating this into .NET based pricing engines. Many financial institutions have a lot of infrastructure built on .NET. So, I decided to build a C# interface.

My first thought was to use something like gRPC or Thrift for inter-process communication between Python and C#. That seemed like overkill for a personal project. I also considered rewriting the inference part of the neural network in C# using a library like ML.NET or ONNX Runtime. ONNX (Open Neural Network Exchange) seemed like the most promising route.

I successfully exported my PyTorch model to the ONNX format.
```python
# dummy_input = torch.randn(1, num_market_tenors, device='cpu') # Ensure it's on CPU for export
# model.to('cpu') # Move model to CPU
# torch.onnx.export(model,
#                   dummy_input,
#                   "nelson_siegel_nn.onnx",
#                   input_names=['market_yields_input'],
#                   output_names=['raw_ns_params_output'],
#                   dynamic_axes={'market_yields_input': {0: 'batch_size'},
#                                 'raw_ns_params_output': {0: 'batch_size'}})
```
Loading and running this ONNX model in C# using `Microsoft.ML.OnnxRuntime` was surprisingly straightforward. The main challenge was ensuring the pre-processing of input yields in C# exactly matched what the Python model expected during training (e.g., normalization, order of tenors).

A simplified C# snippet to call the ONNX model might look something like this (actual implementation would be more robust):

```csharp
// using Microsoft.ML.OnnxRuntime;
// using Microsoft.ML.OnnxRuntime.Tensors;

// public class NelsonSiegelONNXPredictor
// {
//     private InferenceSession session;
//     private List<string> inputNames; // Should be just one: "market_yields_input"
//     private List<string> outputNames; // Should be just one: "raw_ns_params_output"

//     public NelsonSiegelONNXPredictor(string modelPath)
//     {
//         this.session = new InferenceSession(modelPath);
//         // Assuming single input and output based on export
//         this.inputNames = this.session.InputMetadata.Keys.ToList();
//         this.outputNames = this.session.OutputMetadata.Keys.ToList();
//     }

//     public float[] PredictRawNSParams(float[] marketYields)
//     {
//         // Ensure marketYields are preprocessed and in the correct order/shape
//         // For a model expecting [1, num_market_tenors]
//         var dimensions = new int[] { 1, marketYields.Length };
//         var inputTensor = new DenseTensor<float>(marketYields, dimensions);

//         var inputs = new List<NamedOnnxValue>
//         {
//             NamedOnnxValue.CreateFromTensor(this.inputNames, inputTensor)
//         };

//         using (var results = this.session.Run(inputs, this.outputNames)) // Run the session
//         {
//             // Assuming the output is a DenseTensor<float>
//             var outputTensor = results.FirstOrDefault(item => item.Name == this.outputNames)?.AsTensor<float>();
//             if (outputTensor != null)
//             {
//                 // The output is still raw_ns_params.
//                 // The transformation (e.g., exp for tau) and the actual NS formula application
//                 // would need to be re-implemented in C# or the ONNX graph made more complex.
//                 // For this project, I only took the raw params to C# and then reimplemented
//                 // the parameter transformation and the Nelson-Siegel formula in C#.
//                 return outputTensor.ToArray();
//             }
//             return null;
//         }
//     }
    
//     // In C#, I would then need to re-implement the logic:
//     // 1. Transform these raw params (e.g. raw_tau -> exp(raw_tau))
//     // 2. Plug them into the Nelson-Siegel yield formula
//     // 3. Implement the forward rate calculations from these C# computed yields/params
// }
```
The key was that the ONNX model just gives the *raw* parameters. The logic to transform `raw_tau` (e.g., `exp(raw_tau)`) and then plug these parameters into the Nelson-Siegel formulas to get actual yields, and then derive forward rates, had to be faithfully re-implemented in C#. This was a bit tedious but ensured the C# side could independently verify curve properties. Initially, I had slight discrepancies between Python and C# outputs due to floating-point precision differences and how `exp` was handled, but careful comparison resolved most of them.

### Final Thoughts and Lessons Learned

This project was a fantastic learning experience. Combining traditional financial models with neural networks isn't just about throwing a generic NN at a problem; it requires careful thought about how the NN can augment the existing model structure in a meaningful way.

Key takeaways for me:
*   **Domain knowledge is crucial:** Understanding the nuances of yield curves, arbitrage-free conditions, and the Nelson-Siegel model was far more important than just knowing how to build a PyTorch model.
*   **The "arbitrage-free" constraint is hard:** It’s not a simple add-on. It needs to be deeply integrated into the model's learning process. My penalty-based approach worked reasonably well, but I suspect more advanced techniques (e.g., structured outputs from the NN that inherently guarantee positive forward rates) could be even better, though likely more complex to design.
*   **Data preprocessing is half the battle:** Garbage in, garbage out. Cleaning and preparing the ECB data took a significant amount of time but was essential.
*   **Start simple, then iterate:** My first few NN attempts were overly complex. Scaling back and focusing on getting the core mechanics right (NN predicting NS params, penalty for arbitrage) was more effective.
*   **Benchmarking is essential:** Comparing against QuantLib provided a reality check and highlighted areas where the NN approach offered genuine advantages.
*   **Interoperability matters:** Exporting to ONNX and interfacing with C# showed that these Python-based ML models can indeed be integrated into different ecosystems, though it requires careful handling of data types and model input/output contracts.

There's still plenty of room for improvement. I'd like to explore more sophisticated ways to enforce the arbitrage-free condition, perhaps by having the neural network directly output parameters of a positively-constrained forward curve representation. Also, a more rigorous test of hedging performance would be valuable.

Overall, it was a challenging but rewarding project. It definitely deepened my appreciation for the complexities of fixed income modeling and the potential (and pitfalls) of applying machine learning in this space.