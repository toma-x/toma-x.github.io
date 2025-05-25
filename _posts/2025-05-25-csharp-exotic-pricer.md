---
layout: post
title: C\# Exotic Derivative Pricer
---

## Bermudan Swaptions in C#: A Monte Carlo Pricing Journey with AAD

This semester, I dived headfirst into the deep end of quantitative finance for my personal project: building a Monte Carlo pricer for Bermudan swaptions in C# (.NET). The goal wasn't just to get a price, but to also tackle the efficient calculation of Greeks using Adjoint Algorithmic Differentiation (AAD). It was a challenging experience, filled with moments of complete confusion followed by satisfying breakthroughs.

### Getting Started: The Core Monte Carlo Engine and Hull-White

The initial step was setting up the basic Monte Carlo framework. I chose C# primarily because of my familiarity with the .NET ecosystem and its strong numerical libraries, although I mostly ended up building things from scratch to really understand them.

Pricing a Bermudan swaption requires simulating interest rate paths. I opted for the Hull-White one-factor model, dr_t = (θ(t) - a * r_t)dt + σdW_t. My reasoning was that it's relatively tractable, captures mean reversion, and there are analytical formulas for zero-coupon bonds which are useful for calibration and sanity checks. I probably spent a good week just trying to get my head around calibrating θ(t) to the initial yield curve. The QuantLib documentation and various forum discussions were invaluable here, though often they'd discuss the C++ implementation, so translating concepts to C# was sometimes a bit of a puzzle.

Path generation itself was an early hurdle. I implemented the standard Euler-Maruyama scheme for discretization:

```csharp
public static double[,] GenerateHullWhitePaths(
    double r0, 
    Func<double, double> theta, 
    double a, 
    double sigma, 
    double T, 
    int numSteps, 
    int numPaths,
    Random rand) // rand is a pre-seeded random number generator
{
    double dt = T / numSteps;
    double sqrtDt = Math.Sqrt(dt);
    var paths = new double[numPaths, numSteps + 1];

    for (int i = 0; i < numPaths; i++)
    {
        paths[i, 0] = r0;
        for (int j = 0; j < numSteps; j++)
        {
            double dW = rand.NextGaussian() * sqrtDt; // NextGaussian() gives N(0,1)
            double drift = (theta(j * dt) - a * paths[i, j]) * dt;
            paths[i, j + 1] = paths[i, j] + drift + sigma * dW;
        }
    }
    return paths;
}
```
My `NextGaussian` method uses the Box-Muller transform. Initially, I had a bug where I was using the same random numbers for `dW` across different paths for the same time step – that led to some very correlated (and very wrong) results until a late-night debugging session revealed the silly mistake.

Before tackling the Bermudan's early exercise feature, I made sure I could price a European swaption correctly. This felt like a crucial checkpoint. If the European version wasn't working, the Bermudan certainly wouldn't.

### The Bermudan Beast: Longstaff-Schwartz

The early exercise feature of Bermudan swaptions is what makes them tricky. The holder can exercise the swaption at a set of pre-defined dates. This means at each exercise date, one needs to decide if exercising immediately is better than holding onto the option. This is where the Longstaff-Schwartz algorithm comes in.

The core idea is to work backwards from the last exercise date. At each exercise date `t_i`:
1.  Calculate the intrinsic value (the value if exercised immediately).
2.  Estimate the continuation value (the expected value of holding the option, discounted back to `t_i`). This is done by regressing the realized future payoffs from `t_{i+1}` onwards against basis functions of the current state (in my case, the short rate `r_t`).
3.  The optimal decision is to exercise if the intrinsic value is greater than the continuation value.

Choosing the basis functions for the regression was a bit of an art. I started with simple polynomials of the short rate: 1, r, r^2. I remember reading a few papers, and some forum posts on Wilmott, where people discussed using Laguerre polynomials or other more exotic choices, but I decided to stick with simple polynomials for now to manage complexity. My initial implementation of the regression was clunky, relying on a basic matrix library I found, and getting the indices right for the backward induction loop was a nightmare.

Here's a conceptual snippet of the backward iteration step (the actual code is much more involved with instrument details):

```csharp
// Simplified logic for a single exercise date backward step
// cashFlowsAtT contains discounted future cashflows if not exercised at current_t
// stateVariableAtT contains the short rate r(t) for paths in the money
double[] intrinsicValues = CalculateIntrinsicValue(currentExerciseDate, paths, K_strike); // K_strike is the fixed rate
double[] continuationValues = new double[numPathsInMoney];

// Perform regression for paths in the money
if (pathsInTheMoney.Count > 0) 
{
    var xValues = pathsInTheMoney.Select(p => GetStateVariable(p, currentExerciseDate)).ToArray();
    var yValues = pathsInTheMoney.Select(p => GetDiscountedFuturePayoff(p, currentExerciseDate, nextDecisionPayoffs)).ToArray();

    // Using a simple polynomial regression (e.g., y = beta0 + beta1*x + beta2*x^2)
    // coefficients = DoRegression(xValues, yValues, degree: 2); 
    // For each path in the money:
    // continuationValues[idx] = EvaluatePolynomial(coefficients, GetStateVariable(path, currentExerciseDate));
}

// Decision logic
for (int i = 0; i < numPaths; i++)
{
    if (IsExercisable(path_i, currentExerciseDate)) // Check if option is alive for this path
    {
        double intrinsic = intrinsicValues[i];
        double continuation = (IsPathInTheMoney(path_i, currentExerciseDate)) ? 
                                GetContinuationValueForPath(path_i, coefficients) : 0.0;
                                
        if (intrinsic > continuation)
        {
            currentPayoffs[i] = intrinsic; 
            // Mark option as exercised for this path, no further cashflows from this option
        }
        else
        {
            currentPayoffs[i] = GetDiscountedFuturePayoff(path_i, currentExerciseDate, nextDecisionPayoffs); // Discounted payoff from next decision
        }
    }
}
```

Debugging this was tough. My initial prices were way off from benchmarks I tried to find online (which are scarce for specific Bermudan swaptions). One major issue was handling paths that were out-of-the-money for the regression; including them inappropriately skewed the continuation value. I spent hours printing out path values and regression coefficients trying to trace where things went wrong. The breakthrough came when I meticulously stepped through the Longstaff-Schwartz paper again (the original 2001 one) and realized I was misspecifying my dependent variable in the regression for some scenarios.

### Greeks via AAD: The Real Challenge

Calculating Greeks (Delta, Vega, Rho) by "bumping" – re-pricing the swaption after a small change in an input parameter – is computationally brutal for Monte Carlo methods, especially with path-dependent options like Bermudans. This led me to explore Adjoint Algorithmic Differentiation (AAD).

The idea behind AAD is to compute all derivatives with respect to inputs in a single backward pass, roughly proportional to the cost of the original function evaluation, regardless of the number of inputs. This sounded almost too good to be true. I mostly relied on the seminal work by Giles and Glasserman ("Smoking Adjoints: Fast Monte Carlo Greeks") and some more recent papers to understand the theory.

Implementing AAD from scratch for the entire pricer was beyond the scope of what I felt I could achieve in the given time. My approach was to implement a basic "AAD-aware" numerical type, let's call it `DualNumberAAD`, which stores its primal value and an adjoint (or derivative contribution). Then, I overloaded basic arithmetic operations for this type.

```csharp
public struct DualNumberAAD
{
    public double Value;
    public double Adjoint; // Stores dOutput/dValue, to be accumulated

    public DualNumberAAD(double value)
    {
        Value = value;
        Adjoint = 0.0; // Initialized to zero
    }

    public static DualNumberAAD operator +(DualNumberAAD a, DualNumberAAD b)
    {
        return new DualNumberAAD(a.Value + b.Value);
        // Adjoint propagation happens in the backward pass
    }

    public static DualNumberAAD operator *(DualNumberAAD a, DualNumberAAD b)
    {
        return new DualNumberAAD(a.Value * b.Value);
    }
    // ... other operators (-, /, Math.Exp, etc.)
}
```

The tricky part is managing the "tape" or computation graph and the backward propagation. For every operation `c = f(a, b)`, if `a` and `b` are `DualNumberAAD`, then during the forward pass, `c.Value` is computed. During the backward pass, `a.Adjoint += c.Adjoint * df/da` and `b.Adjoint += c.Adjoint * df/db`.

Rewriting parts of the Hull-White path generation and the swaption payoff logic to use `DualNumberAAD` was painstaking. Every function that could affect the final price and depended on an input parameter for which I needed a Greek had to be modified. For example, discounting a cashflow `CF` at time `t` using short rate `r` path: `PV = CF * exp(-Integral(r(s)ds))`. If `r0` (initial short rate) is an input for Delta, then the `r` path generation needs to handle `DualNumberAAD`s.

A significant challenge with AAD in the context of Longstaff-Schwartz is the regression step and the exercise decision. The exercise boundary is non-smooth, and the regression coefficients themselves depend on the input parameters. This is where most "off-the-shelf" AAD explanations fall short for path-dependent American-style options. I had to make some simplifications and accept that the Greeks around the exercise boundary might be less accurate. I primarily focused on getting AAD working for the parts of the calculation that were smoother, like the discounting and the payoff calculations assuming a fixed exercise strategy. Getting full AAD through the optimal stopping decision itself is an advanced research topic. My practical implementation involved "freezing" the exercise decision from a primal pass and then applying AAD to the resulting cashflows, or calculating derivatives of the continuation value approximator.

For instance, if my rate path generation was modified to handle `DualNumberAAD` inputs (e.g., `r0_dual`), the output paths `paths_dual[i, j]` would be `DualNumberAAD` types. Then, calculating a swap leg value:

```csharp
// Simplified fixed leg payment
// Assume r_path_segment_dual elements are DualNumberAAD
// Assume discountFactors_dual are computed from r_path_segment_dual
DualNumberAAD fixedPaymentPV_dual = new DualNumberAAD(0.0);
for(int k=0; k < paymentTimes.Length; ++k)
{
    // nominal_dual could also be a DualNumberAAD if we need dV/dNominal
    DualNumberAAD payment_dual = nominal_dual * K_strike_dual * yearFraction_dual;
    fixedPaymentPV_dual += payment_dual * discountFactors_dual[k];
}
```
Then, in the backward pass, after `finalPrice_dual.Adjoint` is seeded with 1.0, the adjoints would propagate back through these additions and multiplications. For `c = a * b`, `a.Adjoint += c.Adjoint * b.Value` and `b.Adjoint += c.Adjoint * a.Value`. For `c = a + b`, `a.Adjoint += c.Adjoint` and `b.Adjoint += c.Adjoint`. This had to be manually coded for each operation in the backward sweep.

### Benchmarking AAD vs. Bumping

After countless hours of debugging the AAD logic (mostly null reference exceptions from improperly handled `DualNumberAAD` objects or incorrect adjoint accumulation), I finally got to the benchmarking stage. I set up tests to calculate Delta (sensitivity to initial short rate `r0`) and Rho (sensitivity to parallel shift in yield curve) for a sample Bermudan swaption.

The results were quite satisfying. For a reasonable number of paths (e.g., 50,000) and steps (e.g., 200 for a 10-year swaption with annual exercise):
*   **Bumping:** Calculating Delta required two full Monte Carlo valuations (one for `r0` and one for `r0 + dr`). For Rho, it could be even more if I wanted sensitivities to multiple points on the curve.
*   **AAD:** A single forward pass (slightly more expensive due to `DualNumberAAD` operations) and a single backward pass yielded all derivatives simultaneously.

The speed-up for AAD was significant, easily an order of magnitude faster for calculating multiple Greeks compared to running multiple full re-pricings via bumping. The accuracy was also comparable to finite differences, provided the bump size for finite differences was chosen carefully (too small gives noise, too large gives truncation error). I did notice more noise in the AAD Greeks when the exercise decision was very sensitive, which aligns with the theoretical difficulties.

### Reflections and Next Steps

This project was by far the most complex piece of coding I've undertaken. Implementing the Hull-White model, the Longstaff-Schwartz algorithm, and then layering AAD on top of it pushed my understanding of both finance and software development. The main lesson was the importance of breaking down complex problems and verifying each component meticulously. Those sanity checks with European swaptions saved me a lot of pain later on.

There are many limitations, of course. The Hull-White model is a one-factor model, and a multi-factor model would be more realistic. The AAD implementation is specific to this pricer and not a general-purpose library; it also makes simplifying assumptions regarding the non-smoothness of the exercise decision.

Future work could involve:
*   Implementing a more advanced interest rate model (e.g., HJM or LMM).
*   Refining the AAD implementation, perhaps exploring ways to better handle the regression and exercise decisions.
*   Building a small UI to interact with the pricer.

Overall, it was a fantastic learning experience. Wrestling with the intricacies of these models and techniques has given me a much deeper appreciation for the challenges in computational finance.