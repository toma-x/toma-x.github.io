---
layout: post
title: Optimizing \textbf{GPU-Accelerated} Monte Carlo
---

## Tackling Exotic Option Pricing with CUDA: My Journey to a 50x Speedup

For a while now, I've been fascinated by the intersection of finance and high-performance computing. Monte Carlo methods are a cornerstone in quantitative finance for pricing derivatives, especially the more complex, exotic ones. However, their computational cost can be a significant bottleneck. After slogging through some painfully slow CPU-based simulations for a previous assignment, I decided to take on a more ambitious project: accelerating Monte Carlo simulations for exotic option pricing using GPU parallelism with CUDA. My goal was to write CUDA kernels in C++ and see if I could get a substantial speedup over a traditional CPU approach. Spoiler: it worked, and the results were pretty exciting.

My starting point was a fairly standard C++ implementation for pricing, let's say, an Asian call option. An Asian option's payoff depends on the average price of the underlying asset over a certain period, making it path-dependent and a bit more computationally intensive than a plain vanilla European option. My CPU code was single-threaded, using the Box-Muller transform for generating normally distributed random numbers to simulate stock price paths according to Geometric Brownian Motion. It worked, but for a decent number of paths (say, a million) and a good number of time steps (e.g., 252, for daily steps in a year), it would take an uncomfortably long time to run. We're talking minutes, which isn't ideal when you want to iterate quickly or test different parameters.

The jump to CUDA C++ felt like diving into the deep end. I'd read about GPU architecture – all those cores! – and how CUDA allows you to harness them, but theory is one thing and practice is another. My main resource initially was the NVIDIA CUDA C Programming Guide, along with countless forum posts and some online lectures. My university has a few machines with NVIDIA RTX 3070 cards, which became my development playground.

The first challenge was just getting a basic CUDA kernel to compile and run. My "hello world" equivalent was a simple vector addition kernel. Even that took a bit of fiddling with `nvcc`, understanding `__global__` functions, and the whole `<<<gridDim, blockDim>>>` syntax for kernel launches. I remember staring at an early, very uncooperative kernel that just wasn't producing the right output, only to realize I was making a silly mistake in how I was calculating global thread IDs. A classic `int tid = blockIdx.x * blockDim.x + threadIdx.x;` was my savior, but figuring out why my initial, more convoluted attempt failed took longer than I'd like to admit.

Once I had a basic grasp, the real task began: porting the Monte Carlo simulation. The core of Monte Carlo is generating many independent random paths. This immediately brought up the question of random number generation on the GPU. My CPU code used Box-Muller, but implementing that efficiently and correctly for thousands of parallel threads, ensuring independence and good statistical properties, seemed like a recipe for disaster. I briefly considered trying to adapt some existing CPU-based PRNGs, but that felt like reinventing the wheel, and probably a very wobbly wheel at that. Thankfully, NVIDIA provides the cuRAND library. This was a huge help. Setting up cuRAND generators for each thread, or rather, managing states and sequences, was the next hurdle. I spent a good evening trying to understand how `curand_init` and `curand_normal` (or `curand_uniform` then Box-Muller on device) should be used within a kernel. My first attempt with `cuRAND` gave me correlated results across threads, which completely skewed the option prices. It turned out I wasn't seeding the states correctly for each thread to ensure unique random number sequences. A lot of print-debugging (or what passes for it in CUDA by copying intermediate results to CPU) and head-scratching later, I landed on using a unique seed per thread, often derived from the thread ID and a global seed, along with a sequence number.

Let's get into the specifics of the Asian option pricing kernel. For an Asian call option, the payoff is `max(AveragePrice - StrikePrice, 0)`. So, for each simulated path, I needed to:
1.  Simulate the stock price path step-by-step using Geometric Brownian Motion: `S_t = S_{t-1} * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)`, where `Z` is a standard normal random variable.
2.  Calculate the average stock price along that path.
3.  Determine the payoff for that path.
Finally, the option price is the average of all these path payoffs, discounted back to the present.

My strategy was to assign each CUDA thread the task of simulating one full price path.

Here's a simplified look at what the main device function for a single path simulation started to look like. This is a `__device__` function called by my global kernel:

```cpp
__device__ float simulatePathAndGetPayoff(
    float S0, float K, float r, float v, float T, int numSteps,
    curandStatePhilox4_32_10_t* localRandState, float* pathPrices) {

    float dt = T / numSteps;
    float currentS = S0;
    float priceSum = 0.0f;

    // Store S0 for averaging if needed for the specific Asian option type
    // pathPrices = S0; // Depends if S0 is part of the average

    for (int i = 0; i < numSteps; ++i) {
        // Generate standard normal random number
        float Z = curand_normal(localRandState);
        currentS *= expf((r - 0.5f * v * v) * dt + v * sqrtf(dt) * Z);
        priceSum += currentS;
        // if (pathPrices != nullptr) pathPrices[i+1] = currentS; // For debugging path
    }

    float avgPrice = priceSum / numSteps;
    return fmaxf(avgPrice - K, 0.0f);
}
```
The `pathPrices` array was something I added for debugging initially, to be able to copy a few full paths back to the CPU and inspect them. For the final version focused on just getting the price, I wouldn't necessarily need to store every single price point for every path on the GPU if memory became an issue, just the running sum for the average.

The global kernel then would look something like this:

```cpp
__global__ void asianOptionPricingKernel(
    float* d_payoffs, unsigned int numSims,
    float S0, float K, float r, float v, float T, int numSteps,
    unsigned long long seed) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numSims) {
        curandStatePhilox4_32_10_t localRandState;
        // Initialize cuRAND state for each thread
        // Critical to get different seeds/sequences per thread
        curand_init(seed, tid, 0, &localRandState);

        // Simulate one path and calculate its payoff
        // No pathPrices array passed here for the optimized version
        float payoff = simulatePathAndGetPayoff(S0, K, r, v, T, numSteps, &localRandState, nullptr);
        d_payoffs[tid] = payoff;
    }
}
```
In the host code, I'd allocate memory on the GPU for `d_payoffs` using `cudaMalloc`, launch the kernel, copy the results back using `cudaMemcpy(..., cudaMemcpyDeviceToHost)`, and then average these payoffs on the CPU. Discounting would also typically happen on the CPU after averaging.

One tricky part was deciding on the grid and block dimensions. `blockDim.x` is limited (e.g., to 1024 threads per block on many architectures). I played around with block sizes like 128, 256, and 512. A larger block size can sometimes help with hardware utilization, but it depends on the kernel's resource usage (registers, shared memory). I mostly used 256 threads per block as it seemed like a good starting point. Then `gridDim.x` would be `(numSims + blockDim.x - 1) / blockDim.x` to ensure enough blocks to cover all simulations.

Data transfer between CPU and GPU (`cudaMemcpy`) can be a bottleneck. For this project, I was sending the scalar parameters (S0, K, r, etc.) to the kernel directly (or via constant memory, which I experimented with but found minimal difference for this particular kernel size) and getting back an array of payoffs. The bulk of the work was done on the GPU, so the data transfer overhead was relatively small compared to the computation, which is ideal.

Debugging was, as expected, painful at times. `printf` inside a CUDA kernel is possible but can be slow and cumbersome. My go-to was often to write intermediate results to global memory and copy them back to the CPU for inspection. This is how I caught the `cuRAND` seeding issue. I noticed my option prices were way off, and when I outputted the first few random numbers generated by each thread, many were identical. A facepalm moment followed by a dive back into the `cuRAND` documentation and some StackOverflow threads discussing proper state initialization. One particular thread I remember had a user with a very similar problem, and the solution involved carefully offsetting the sequence number or using a more robust seeding scheme based on global thread ID.

The performance tuning phase was iterative. My first GPU version was faster than the CPU, but not dramatically so. I used NVIDIA's `nvprof` (and later Nsight Systems as `nvprof` is getting older) to look at kernel execution time and occupancy. Occupancy tells you how many warps (groups of 32 threads) are active on a multiprocessor at a given time. Low occupancy can mean your kernel isn't using the GPU efficiently. For my kernel, register usage per thread was a factor. Too many registers, and fewer threads can run concurrently on a multiprocessor. I had to be mindful of the complexity within `simulatePathAndGetPayoff`. I tried a version where `pathPrices` was stored in shared memory for each block to do some intermediate block-level averaging, but for simple path averaging per thread, it didn't give a significant boost and added complexity. The main win was simply having so many threads run in parallel.

My CPU baseline was a single-threaded C++ program compiled with g++ using `-O3` optimization. For, say, 1,000,000 paths and 252 steps, the CPU version might take around 120 seconds on my test machine. The GPU version, running on the RTX 3070, once I ironed out the kinks, brought this down to about 2.4 seconds for the same parameters. That’s a 50x speedup! This was a huge moment of satisfaction. The GPU has thousands of cores (the RTX 3070 has 5888 CUDA cores), and even though each core is simpler than a CPU core, the sheer parallelism for a task like Monte Carlo, where each path simulation is independent, is a game-changer.

Of course, it wasn't all smooth sailing. There were moments of intense frustration, especially when a kernel would silently produce wrong results or, worse, crash with an unspecified launch failure. One such instance was due to trying to allocate too much shared memory per block – a rookie mistake. Another time, an out-of-bounds write in the `d_payoffs` array (due to a miscalculation in `tid` before I put the `if (tid < numSims)` guard) caused sporadic errors that were a nightmare to track down.

Looking back, this project taught me an incredible amount about parallel programming paradigms, GPU architecture, and the practicalities of CUDA C++. While the 50x speedup was fantastic, the deeper understanding of how to approach and debug massively parallel problems was the real takeaway. There's still more that could be done – exploring different exotic options, perhaps looking into reducing the payoffs on the GPU instead of copying all of them back, or even using multiple GPUs. But for now, seeing those simulation times plummet was incredibly rewarding. It definitely solidified my interest in HPC.