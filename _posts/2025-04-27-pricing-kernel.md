---
layout: post
title: Optimized Pricing Kernel Implementation
---

I recently wrapped up a personal project I've been chipping away at: building an optimized kernel for option pricing. Coming from a background where most of my initial financial modeling was done purely in Python, I started to feel the performance bottlenecks, especially when dealing with large datasets or complex models. My earlier pure-Python pricing scripts, while functional, were just too slow for what I envisioned for a more robust analysis tool. The goal for this project was fairly straightforward: develop a high-performance pricing engine in a compiled language and then figure out how to expose that functionality back to Python, where I'm more comfortable with data manipulation and analysis.

The decision to go with C++ for the core kernel wasn't immediate. I briefly looked at Cython, thinking it might be an easier transition from Python, but the syntax felt a bit clunky for what I wanted to build from the ground up, and I wasn't sure how much "true" performance gain I'd get compared to a heavily optimized C++ implementation. Rust was another thought, given its reputation for safety and performance, but my C++ knowledge was already significantly more developed from coursework. Given the time I had allocated (about a month of evenings and weekends), diving deep into Rust felt like a side quest I couldn't afford. So, C++ it was.

Building the core C++ kernel took up the bulk of the first two weeks. I started with a simple Black-Scholes model implementation. My first draft was pretty naive, lots of explicit loops and standard library math functions. Getting everything to compile cleanly with GCC was an initial hurdle – lots of `-std=c++17` flags I kept forgetting, and header include paths that weren't quite right for external libraries like Boost (which I was considering for special functions, though I ended up mostly relying on `<cmath>`). I remember spending maybe two hours just fixing cryptic linker errors on the first day, realizing I wasn't linking against the math library correctly.

A specific point of confusion early on was handling input data. In Python, pandas DataFrames make passing around arrays of parameters (like strikes, volatilities, etc.) trivial. In C++, I had to decide on containers. My first pass used `std::vector<double>`. It worked, but I had a hunch that for pure numerical loops, raw arrays might be faster due to potentially better cache locality and simpler memory layout. After coding up the vector version, I wrote a quick benchmark locally, then spent another evening (maybe 3-4 hours) refactoring the core loops to use raw `double*` pointers passed around. I had to be much more careful with memory management, obviously, but initial microbenchmarks suggested it was slightly faster for the core calculation loops. I decided to stick with raw pointers for the critical inner loops where performance was paramount, and use `std::vector` for parameter storage outside the hot path. This felt like a reasonable compromise given I didn't want to write a full-blown memory manager.

Here's a simplified snippet of what the core calculation function ended up looking like (this is *after* moving to pointers):

```cpp
// price_kernel.cpp
#include <cmath>
#include <vector> // Still used for input vectors before copying to raw pointers

// ... (other includes and helper functions like normal_cdf)

extern "C" { // Essential for C linkage if called from other languages easily

    void black_scholes_batch(
        int num_options,
        const double* S,        // Spot price array
        const double* K,        // Strike array
        const double* T,        // Time to maturity array
        const double* R,        // Risk-free rate array
        const double* Sigma,    // Volatility array
        const int* option_type, // 0 for call, 1 for put array
        double* prices          // Output price array
    ) {
        // This is the optimized version using raw pointers
        // Initial approach used std::vector<double> inputs and outputs,
        // which worked but had a slight overhead in my early tests.
        // Decided on pointers for direct memory access in the inner loop.

        for (int i = 0; i < num_options; ++i) {
            double d1 = (std::log(S[i] / K[i]) + (R[i] + 0.5 * Sigma[i] * Sigma[i]) * T[i]) / (Sigma[i] * std::sqrt(T[i]));
            double d2 = d1 - Sigma[i] * std::sqrt(T[i]);

            if (option_type[i] == 0) { // Call
                prices[i] = S[i] * normal_cdf(d1) - K[i] * std::exp(-R[i] * T[i]) * normal_cdf(d2);
            } else { // Put
                prices[i] = K[i] * std::exp(-R[i] * T[i]) * normal_cdf(-d2) - S[i] * normal_cdf(-d1);
            }
            // Need to handle edge cases like T=0 or Sigma=0,
            // my first version didn't do this leading to NaNs.
            // Added checks for T > 0 and Sigma > 0 after debugging.
        }
    }

    // ... (maybe other pricing models later)
}

// My initial normal_cdf implementation used a simple Taylor series - BAD IDEA.
// It was inaccurate for extreme values. Switched to using the Boost math
// library's `boost::math::cdf::normal(0, 1, x)` or, as shown above (simpler),
// a more numerically stable custom implementation found on a numerical recipes forum
// after spending maybe 5 hours trying to debug why my calculated prices
// differed slightly from online calculators. Boost felt like overkill just for CDF,
// and linking it added build complexity, so I went with a public domain algorithm.
```

With the C++ core reasonably stable, the next challenge was the Python binding. I chose `pybind11`. It seemed relatively modern and less boilerplate-heavy than SWIG, and specifically designed for C++11 and later, which aligned with my C++ code. Getting `pybind11` set up with CMake was another time sink. The documentation is good, but integrating it into my existing, albeit simple, C++ build structure took trial and error. My CMakeLists.txt file grew from about 10 lines to 30 as I figured out how to add `pybind11` as a submodule or external dependency, set up the module, and link everything correctly. I hit a persistent error for about an hour related to Python library paths that CMake wasn't finding until I explicitly set `PYTHON_EXECUTABLE` and `PYTHON_INCLUDE_DIR`.

Mapping the C++ functions and data types to Python was mostly straightforward with `pybind11`. Defining the module and exposing the `black_scholes_batch` function looked something like this:

```cpp
// pybind11_kernel.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed for vector conversions if I hadn't switched to raw pointers
#include <pybind11/numpy.h> // Essential for handling NumPy arrays efficiently

#include "price_kernel.h" // My C++ pricing functions header

namespace py = pybind11;

PYBIND11_MODULE(optimized_pricing_kernel, m) {
    m.doc() = "Optimized option pricing kernel"; // Optional module docstring

    m.def("black_scholes_batch", &black_scholes_batch,
          "Compute Black-Scholes prices for batches of options",
          py::arg("num_options"),
          py::arg("S").noconvert(), // Use .noconvert() to ensure we get raw NumPy arrays
          py::arg("K").noconvert(),
          py::arg("T").noconvert(),
          py::arg("R").noconvert(),
          py::arg("Sigma").noconvert(),
          py::arg("option_type").noconvert(),
          py::arg("prices").noconvert() // Output array pre-allocated in Python
    );

    // Initially I tried passing std::vector<double> back and forth,
    // but pybind11 + NumPy interop is much faster by directly
    // accessing the NumPy array buffer using .noconvert() and
    // pybind11::array_t. This was a key optimization I found
    // trawling through pybind11 examples and StackOverflow posts
    // on NumPy integration - saved me hours of potential manual
    // data copying. Realized this maybe 2.5 weeks in.
}
```

The `noconvert()` flag and handling NumPy arrays directly via `pybind11::array_t` (which casts nicely to `double*` for simple cases like mine) was a crucial discovery. My initial attempt involved `std::vector` which `pybind11` can convert, but it involved extra copying. Switching to direct NumPy buffer access after reading a few examples significantly improved the binding overhead.

The final phase was integration and performance testing. I wrote Python scripts to generate large sets of random option parameters, price them using my old pure-Python code, and then price them using the new `optimized_pricing_kernel` module. I used Python's `timeit` module and also the `cProfile`/`line_profiler` tools to identify bottlenecks.

The very first performance test results were confusing. The C++ version was faster, but not by the 20% I was hoping for based on simple C++ benchmarks. Profiling showed that a lot of time was still spent in Python setting up the data arrays and calling the C++ function repeatedly for smaller batches. The "batch" function was designed to take *all* parameters for *all* options in one go. When I switched my Python test script to prepare all input arrays upfront in NumPy and make a single call to the C++ `black_scholes_batch` function, the performance jumped significantly. This demonstrated that the overhead of crossing the Python/C++ boundary was non-trivial, and batching calls was essential. This optimization in the *usage pattern* within Python added maybe another day (6 hours) to refine.

After ensuring the Python usage was properly batched and the C++ core was using raw pointers on NumPy data buffers, I consistently observed around a 20% speedup compared to my original vectorized pure-Python implementation for batches of 100,000 options. It wasn't the 100x speedup you sometimes hear about when moving to C++, but for this specific type of calculation (heavy on floating-point math, light on complex data structures), 20% felt like a solid and achievable gain given the project scope and time constraints.

Looking back, the project took roughly three weeks of focused effort. The biggest lessons weren't just about C++ or Python bindings in isolation, but about the *interaction* between them – how data is passed, the overhead of function calls across the boundary, and the importance of profiling the *integrated* system, not just the individual components. There are still areas for improvement, like adding more sophisticated models, better error handling from the C++ side, and perhaps a more robust build system, but for now, having a faster core pricing engine accessible from my familiar Python environment is a significant step forward for my personal analytical tools.