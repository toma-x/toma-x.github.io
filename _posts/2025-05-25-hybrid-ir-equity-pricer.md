---
layout: post
title: Hybrid IR-Equity Derivative Pricer
---

## Wrestling the Hydra: My Foray into Hybrid IR-Equity Derivative Pricing with Heston-Hull-White

This last semester has been a deep dive, to say the least. What started as a somewhat ambitious idea for a personal project – pricing exotic options with stochastic interest rates – quickly snowballed into a full-blown implementation of a Hybrid Interest Rate-Equity model. Specifically, I decided to tackle the Heston-Hull-White model, and let me tell you, it was a journey. My weapon of choice: C#, with a crucial assist from QuantLib for the heavy lifting of calibration.

### The Theoretical Mountain

Initially, the goal was to get a better handle on how interest rates and equity volatility interact. Fixed rates and Black-Scholes are fine for vanilla stuff, but the real world is messier. I stumbled upon the Heston-Hull-White (HHW) model in a few papers (Brigo & Mercurio's book was also a constant, if somewhat intimidating, companion). It seemed like a solid choice: Heston for stochastic volatility on the equity side, and Hull-White for stochastic short rates. The correlation between all the driving Brownian motions (equity, variance, and interest rates) is where the "hybrid" magic, and the complexity, really lies.

The SDEs looked something like this (from memory, so apologies for any transcription errors from my notes):

For the asset price S_t:
`dS_t = r_t S_t dt + \sqrt{v_t} S_t dW_1_t`

For the variance v_t (Heston):
`dv_t = \kappa(\theta - v_t)dt + \sigma_v \sqrt{v_t} dW_2_t`

For the short rate r_t (Hull-White):
`dr_t = (\phi(t) - a r_t)dt + \sigma_r dW_3_t`

And then `dW_1_t, dW_2_t, dW_3_t` are correlated. My first thought was just, "How on earth am I going to simulate this?"

### Setting Up the Battlefield: C# and QuantLib

I decided on C# mostly because I've used it for a couple of other university projects and find Visual Studio a pretty comfortable environment. The real challenge was getting QuantLib to play nice. I knew I didn't want to implement calibration routines from scratch – that’s a whole Ph.D. thesis in itself. QuantLib has all that, but it's C++, and I needed the C# bindings (SWIG wrappers).

Getting the `QuantLib- Обычно я скачиваю с официального сайта QuantLib, но в этот раз решил попробовать с NuGet пакета, чтобы упростить себе жизнь. (Usually I download from the official QuantLib website, but this time I decided to try the NuGet package to make my life easier). That took a bit of fiddling. I remember having some issues with the target framework for my C# project not matching what the QuantLib C# bindings expected. Lots of recompiling and checking project properties. Finally, after a few tries, I got a basic QuantLib example – like creating a `DayCounter` – to run in C#. Small victory, but crucial.

### The Heart of the Beast: Monte Carlo in C#

With QuantLib somewhat tamed for later calibration duties, I turned to the core pricer. This meant a Monte Carlo simulation for the HHW model.

**Path Generation:** This was the trickiest C# part. I opted for an Euler-Maruyama discretization for the SDEs. I know Milstein offers better convergence for some SDEs, but given the complexity of HHW already, I wanted to start with something more manageable. My time constraint was also a factor here; I needed to get a working prototype first.

The correlation part had me scratching my head for a while. I had a 3x3 correlation matrix:
`rho_S,v` (equity and variance)
`rho_S,r` (equity and rates)
`rho_v,r` (variance and rates)

To generate correlated random numbers, I had to use Cholesky decomposition. My linear algebra classes finally paid off! I found a decent C# math library (`MathNet.Numerics`, I think) that had a Cholesky function, which saved me from implementing that from scratch too.

My path generation loop for a single path started to look something like this (this is a very simplified snippet from one of my earlier, messier versions):

```csharp
// Inside my MonteCarloPricer class
// ...
private PathResult GenerateSinglePath(
    HestonHullWhiteParameters hhParams,
    double S0, double r0, double v0,
    double T, int numSteps,
    RandomNumberGenerator rng) // My wrapper for RNG
{
    double dt = T / numSteps;
    double sqrtDt = Math.Sqrt(dt);

    var pathS = new List<double> { S0 };
    var pathR = new List<double> { r0 };
    var pathV = new List<double> { v0 };

    double currentS = S0;
    double currentR = r0;
    double currentV = v0;

    // Cholesky decomposition L of correlation matrix (done elsewhere, passed in or part of hhParams)
    // Matrix L = hhParams.CholeskyMatrix;

    for (int i = 0; i < numSteps; i++)
    {
        double dZ1 = rng.NextGaussian();
        double dZ2 = rng.NextGaussian();
        double dZ3 = rng.NextGaussian();

        // Correlate the random numbers
        // dW_S = L*dZ1
        // dW_V = L*dZ1 + L*dZ2
        // dW_R = L*dZ1 + L*dZ2 + L*dZ3
        // (Actual implementation involved accessing matrix elements properly)
        double dWS = hhParams.LMatrix * dZ1;
        double dWV = hhParams.LMatrix * dZ1 + hhParams.LMatrix * dZ2;
        double dWR = hhParams.LMatrix * dZ1 + hhParams.LMatrix * dZ2 + hhParams.LMatrix * dZ3;


        // Euler discretization for Hull-White rate
        // phi(t) needs to be obtained from the input yield curve for Hull-White
        // This was a major point of confusion - how to get phi(t) correctly aligned with QuantLib's calibration.
        // For now, let's assume a simple theta_r for dr_t = (theta_r - a*currentR)*dt + sigmaR*dW_R
        // currentR += hhParams.HW_a * (hhParams.HW_theta_t_placeholder - currentR) * dt + hhParams.HW_sigmaR * sqrtDt * dWR; // Simplified
        // This theta_r placeholder was a source of bugs until I properly hooked it to the calibrated forward curve.
        // The actual Hull-White drift is (theta(t) - a*r_t)dt where theta(t) ensures the model fits the initial term structure.
        // I later learned that theta(t) = f'(t) + a*f(t) + (sigma_r^2 / 2a) * (1 - exp(-2at)) where f(t) is the instantaneous forward rate.
        // For the actual pricer, I used a discrete approximation of this or passed the calibrated QuantLib curve.

        // This is where I had to get the forward rate from the QuantLib curve.
        // Let's say I have a method GetForwardRate(time)
        double forwardRate = GetInstantaneousForwardRate(i * dt); // A helper I wrote
        double theta_t = GetDerivativeOfForwardRate(i * dt) + hhParams.HW_a * forwardRate; // Simplified, sigma_r part missing for brevity
                                                                                        // but needs to be consistent with QL's HW model.
        currentR += (theta_t - hhParams.HW_a * currentR) * dt + hhParams.HW_sigmaR * sqrtDt * dWR;
        currentR = Math.Max(currentR, 0.0001); // Ensure positive rates, crude floor

        // Euler for Heston variance (Feller condition 2*kappa*theta > sigma_v^2 is important here)
        currentV += hhParams.Heston_kappa * (hhParams.Heston_theta - currentV) * dt +
                    hhParams.Heston_sigmaV * Math.Sqrt(Math.Max(currentV,0)) * sqrtDt * dWV; // Max(currentV,0) for sqrt
        currentV = Math.Max(currentV, 0.00001); // Variance floor

        // Euler for Asset Price
        currentS += currentR * currentS * dt + Math.Sqrt(currentV) * currentS * sqrtDt * dWS;
        currentS = Math.Max(currentS, 0.0); // Asset price floor

        pathS.Add(currentS);
        pathR.Add(currentR);
        pathV.Add(currentV);
    }
    // PathResult would store these lists
    return new PathResult(pathS, pathR, pathV);
}

// Placeholder for parameters
public class HestonHullWhiteParameters {
    public double Heston_kappa, Heston_theta, Heston_sigmaV, Heston_rho_SV; // rho_SV for Heston alone
    public double HW_a, HW_sigmaR;
    public double EquityRate_rho, VolRate_rho; // Correlations for the hybrid model
    public QuantLib.Matrix LMatrix; // Cholesky matrix for dWS, dWV, dWR
    // ... plus QuantLib objects for curves etc.
}

// This is just a conceptual snippet. My actual code evolved a lot.
// The GetInstantaneousForwardRate and its derivative were particularly nasty to get right
// and ensure they matched what QuantLib was using internally during calibration.
// I spent a lot of time just printing out QuantLib's zero rates, forward rates,
// and discount factors at various points to try and match them.
```

One tricky bit was ensuring the variance `v_t` stayed positive. The "full truncation" scheme is common, where you use `max(v_t, 0)` inside the square root, but I read some papers suggesting other schemes. For simplicity, I went with `max(v_t, 0)` in the diffusion term and a floor for `v_t` itself. Not perfect, but practical for a student project.

**Exotic Option Choice: Asian Call**
I decided to price an Asian Call option. The payoff is `max(Average_S - K, 0)`, where `Average_S` is the arithmetic average of the stock price over a certain period. This path-dependency makes it a good candidate for a hybrid model, as the averaging process will be influenced by the entire path of interest rates and volatility.

### The QuantLib Calibration Maze

This was, without a doubt, the most challenging part. The idea was to use QuantLib to calibrate the Heston parameters to equity option market prices (or implied vols) and the Hull-White parameters to interest rate derivatives (like caplets or swaptions). Then, combine these with an assumed correlation structure.

My process looked something like this:
1.  **Hull-White Calibration:**
    *   Get a yield curve. I bootstrapped one from some sample market deposit and swap rates using QuantLib's `PiecewiseLogLinearDiscount` curve.
    *   Set up a `HullWhite` model in QuantLib, providing it the term structure.
    *   Create `CapFloor` pricing engines using this model.
    *   Define an objective function to minimize the difference between model CapFloor prices and some "market" CapFloor prices (I had to make these up for the project, or use example data).
    *   Use QuantLib's `LevenbergMarquardt` optimizer.

    Getting the `QuantLib.HullWhiteProcess` correctly initialized and then extracting the calibrated `a` and `sigma_r` parameters took ages. I remember being stuck on getting the `phi(t)` term (the `fittingTerm` in QuantLib Hull-White) from the model after calibration, as my C# pricer needed to be consistent. I found a QuantLib forum post (I wish I'd saved the link) that discussed accessing `model.parameters()` and how they map to `a` and `sigma`. That was a lifesaver. It returns a `QLArray`, and you have to know which index corresponds to which parameter.

2.  **Heston Calibration:**
    *   Similar process: set up a `HestonModel` in QuantLib.
    *   Use a set of European option market prices (again, mostly hypothetical or from textbook examples for my purposes).
    *   Create `AnalyticHestonEngine` instances.
    *   Calibrate `kappa`, `theta`, `sigma_v`, `rho_S,v`, and `v0` using an optimizer. This was often slow and sometimes failed to converge, forcing me to play with initial guesses or optimizer settings like `maxIterations` or `rootEpsilon`.

    One particular headache was the `HestonModelHelper` class in QuantLib. You need to feed it market option data (strike, maturity, price/vol). Ensuring the `Calendar` and `DayCounter` settings were consistent everywhere was crucial, otherwise, you'd get subtle errors that were hard to debug.

3.  **Combining and Correlation:** The HHW model requires correlations `rho_S,r` (equity-rate) and `rho_v,r` (vol-rate). These are harder to calibrate directly. For this project, I treated them as exogenous inputs I could vary to see their impact. This was a simplification, but full calibration of a HHW model is extremely complex.

Once I (thought I) had calibrated parameters, I needed to get them from QuantLib objects back into my C# `HestonHullWhiteParameters` struct. This involved a lot of `model.params()[i]` calls and careful mapping.

Here's a conceptual idea of how I tried to pull parameters from QuantLib for Hull-White (actual code was more involved with error handling and specific QuantLib object lifecycles):

```csharp
// Assuming 'calibratedQLHullWhiteModel' is a QuantLib.HullWhite object
// that has been successfully calibrated.
// And 'yieldCurveHandle' is a QuantLib.YieldTermStructureHandle for the initial curve.

// ... inside a method that bridges QuantLib to my C# params ...
// My HestonHullWhiteParameters customParams = new HestonHullWhiteParameters();

// QuantLib.HullWhite qlHWModel = (QuantLib.HullWhite)engine.getModel(); // Or however I got the model
// customParams.HW_a = qlHWModel.parameters();
// customParams.HW_sigmaR = qlHWModel.parameters();

// Storing the term structure handle is important because HW is fitted to it.
// customParams.InitialTermStructure = yieldCurveHandle; // My own class would wrap this or store necessary info

// The 'fittingTerm' or the way Hull-White ensures it matches the initial yield curve
// is handled internally by QuantLib's HullWhite model when you price with it.
// For my own C# pricer's theta(t) = f'(t) + a*f(t) + ..., I needed to be able to query
// f(t) (instantaneous forward rate) and f'(t) from this yieldCurveHandle.
// QuantLib's YieldTermStructure has methods like .forwardRate() which I used.
// Getting the derivative f'(t) was approximated numerically if a direct method wasn't obvious.
// This consistency was a major point of struggle.
```
The `phi(t)` in my SDE `dr_t = (\phi(t) - a r_t)dt + \sigma_r dW_3_t` is crucial. In QuantLib's Hull-White, this `phi(t)` (often denoted `theta(t)` in literature) is chosen such that the model perfectly fits the initial term structure of interest rates. I had to make sure my C# implementation of the Hull-White process used a `phi(t)` consistent with the term structure I calibrated against in QuantLib. This involved querying instantaneous forward rates `f(t)` from the QuantLib yield curve and calculating `phi(t) = df(t)/dt + a*f(t)`. The derivative `df(t)/dt` was another fun part – I ended up using a finite difference approximation on the forward rates from the curve.

### Putting It All Together & First Results

After (many) cycles of coding, debugging, and re-reading theory, I finally managed to get the calibrated parameters from QuantLib (or my best attempt at them) into my C# Monte Carlo engine. I ran it for an Asian Call option with, say, 100,000 paths and 252 time steps (representing daily steps for a year).

The first time I got a non-NaN price that seemed somewhat in the ballpark of what I might expect (e.g., slightly different from a Black-Scholes price for a similar European option), it was a huge relief. I spent a lot of time comparing prices with and without stochastic rates, or with different correlation parameters, to see if the model behaved intuitively. For example, for a payer option, a positive correlation between rates and the underlying might increase the option price, and my model seemed to show such sensitivities, which was reassuring.

### Reflections and What's Next

This project was an order of magnitude more complex than anything I'd attempted before.
Key takeaways:
*   **The Devil is in the Details:** Tiny inconsistencies between the QuantLib model assumptions (e.g., how `phi(t)` is derived for Hull-White) and my C# implementation could lead to wildly incorrect results. Hours were spent just trying to align these.
*   **Calibration is an Art:** Getting optimizers to converge is tricky. Initial guesses matter. Market data quality (even my hypothetical data) matters.
*   **C# and QuantLib Can Work:** It's not seamless, and the documentation for the C# bindings can be sparse compared to C++ or Python, but it's doable. StackOverflow and the QuantLib mailing list archives were invaluable, even if I often had to "translate" Python or C++ examples in my head.
*   **Numerical Stability:** With SDEs, especially for variance, ensuring paths don't blow up or go negative requires careful implementation of the discretization scheme.

There's still so much more that could be done.
*   **Performance:** My C# Monte Carlo is slow. Parallelizing the path generation loop would be the next logical step.
*   **More Rigorous Calibration:** Calibrating the full HHW model simultaneously, including the equity-rate and vol-rate correlations, is the "proper" way but significantly harder.
*   **Different Exotic Options:** Trying other path-dependent options like barrier or cliquet options.
*   **Advanced Discretization:** Implementing a Milstein scheme or other higher-order schemes for the SDEs.

Overall, while incredibly challenging, this project taught me a massive amount about financial modeling, numerical methods, and the practicalities of integrating complex libraries. There were many late nights staring at code, wondering why my variance was exploding or my calibrated parameters made no sense. But finally seeing plausible numbers emerge was incredibly rewarding. It's definitely given me a much deeper appreciation for the models used in the industry.