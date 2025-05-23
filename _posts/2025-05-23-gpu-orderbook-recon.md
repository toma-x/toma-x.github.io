---
layout: post
title: Latency Optimized Order Book Reconstruction via GPU
---

## Pushing the Boundaries: Sub-Microsecond Order Book Reconstruction with C++ CUDA

For a while now, I've been pretty deep into the world of high-frequency trading concepts, not for actual trading, mind you, but the computational challenges are just fascinating. One area that particularly caught my attention was the reconstruction of limit order books (LOBs). The sheer volume of data and the need for ultra-low latency processing make it a perfect candidate for some serious optimization. So, I decided to take a crack at it for a personal project, aiming to leverage GPU parallelism with C++ and CUDA. My goal was ambitious: sub-microsecond update speeds.

The core problem with LOB reconstruction is managing a flood of events – new orders, cancellations, modifications, and trades – each needing to be processed in strict sequence to maintain an accurate view of the market. On a CPU, even highly optimized C++ code can hit a wall when dealing with millions of updates per second. Each event potentially modifies the sorted price levels of the book, which can be computationally intensive.

My initial thought was, "Can I just throw more CPU cores at it?" But LOB updates have strong sequential dependencies, especially within a single instrument. Simply parallelizing across events naively would lead to race conditions and an incorrect book state. I read a few papers on FPGA implementations, which are incredible but way beyond my current resources and expertise. Then I started looking more seriously at GPUs. The SIMT (Single Instruction, Multiple Threads) architecture of CUDA seemed promising if I could structure the problem correctly. The idea wasn't to parallelize individual event processing across many events (due to dependencies), but perhaps to parallelize the *internal* operations of updating the book for *each* event, or to find ways to batch independent updates.

Setting up the CUDA environment on my Linux machine with an NVIDIA GeForce RTX 3070 wasn't too bad, thankfully. I’d had some experience with CUDA from a university course, but mostly basic vector additions. This was a different beast. The NVIDIA documentation and some StackOverflow threads were my best friends for figuring out the right toolkit version (CUDA 11.6 at the time) and compiler flags for nvcc.

The first major hurdle was data representation. How do you represent an order book efficiently on a GPU? A typical LOB is a sorted list of price levels, and each level contains a queue of orders. Linked lists are generally a no-go on GPUs due to pointer chasing, which kills memory coalescing. I considered using arrays for price levels, perhaps pre-allocating a large number of potential price ticks. For the orders at each price level, I briefly thought about dynamic arrays but worried about reallocation overhead and memory fragmentation on the GPU.

I settled on a fixed-size array for price levels for both bids and asks. Each entry in this array would then point to (or rather, in a GPU context, perhaps manage an index into) a block of orders. To simplify, I initially decided to just store the aggregate volume at each price level, not individual orders. This simplified the problem to updating quantities at price levels. I knew this was a shortcut, but I needed a starting point.

My first attempt at a kernel was to process a batch of incoming tick data. Each tick would be an event (e.g., 'N' for new order, 'C' for cancel).

```cpp
// A very simplified representation of a book level
struct PriceLevel {
    double price;
    int quantity;
};

// Device array for bids and asks
__device__ PriceLevel d_bids[MAX_LEVELS];
__device__ PriceLevel d_asks[MAX_LEVELS];
__device__ int d_num_bid_levels;
__device__ int d_num_ask_levels;

// Simplified event structure
struct Event {
    char type; // 'A' (Add/New), 'M' (Modify), 'X' (Cancel)
    char side; // 'B' (Bid), 'S' (Ask)
    double price;
    int quantity;
    long long order_id; // Important for cancels/modifies
};

__global__ void process_event_kernel(Event* event) {
    // This kernel is overly simplistic and processes only one event.
    // It's more of a conceptual starting point.
    // A real implementation would handle a batch of events or use a different parallelism strategy.

    int tid = threadIdx.x; // Not really using this effectively here for a single event

    if (tid == 0) { // Let only one thread process this single event for now
        if (event->type == 'A') {
            if (event->side == 'B') {
                // Naive insertion - THIS IS WRONG for maintaining sorted order without more logic
                // and doesn't handle existing levels properly.
                // This is where the real complexity begins.
                // For a real LOB, you'd search for the price level, update if exists,
                // or insert and shift if new and maintaining a dense sorted array.
                for (int i = 0; i < d_num_bid_levels; ++i) {
                    if (d_bids[i].price == event->price) {
                        d_bids[i].quantity += event->quantity;
                        return; // Found and updated
                    }
                    // More complex logic needed here for sorted insertion
                }
                // If not found, and if there's space, add it (simplified)
                // This part is tricky because you need to maintain sorted order.
                // A real implementation would be far more complex, involving shifting elements
                // or using a different data structure that supports efficient sorted insertion.
            } else { // Ask side
                // Similar logic for asks
            }
        }
        // ... handle 'M' and 'X' types
    }
}
```

The snippet above is a gross oversimplification of what's needed. My actual first "working" kernel was processing a *single* event with one thread block, trying to parallelize the search within that price level array. For instance, if adding a new bid, I'd launch threads to find the correct insertion point in the `d_bids` array. This immediately brought up issues: how do you efficiently insert into a sorted array in parallel and then shift elements? This seemed like I was trying to force a CPU-style algorithm onto the GPU.

I quickly realized that processing one event at a time with a kernel launch was incredibly inefficient due to kernel launch overhead. The key had to be processing a *batch* of events, or having a persistent kernel that pulls events from a queue. I leaned towards batch processing first.

A major confusion point was managing the state of the LOB between kernel calls. Each batch of events modifies the book. The modified book from `batch_N` must be the input to `batch_N+1`. This meant copying the LOB data back and forth between CPU and GPU if I wasn't careful, or keeping it resident on the GPU and passing pointers. I opted for keeping the LOB resident on the GPU.

One of the first "Aha!" moments came when I started thinking about how to update the price levels. Instead of a dense sorted array that requires shifting, I considered using a fixed array representing all possible price ticks in a certain range. For example, if prices move in 0.01 increments, I could have an array where `book[price_as_int_offset]` holds the quantity. This avoids shifting but can lead to huge, sparse arrays if the price range is wide. This is a common trick, sometimes called a "direct mapped" or "array LOB." I decided to try this for a limited price range. This simplifies updates to `O(1)` once you have the price, but the memory footprint was a concern.

```cpp
// Using a direct-mapped array for quantities
// Assuming prices are converted to integer indices
__device__ int d_bid_quantities[MAX_PRICE_TICKS];
__device__ int d_ask_quantities[MAX_PRICE_TICKS];
// Need to also store the actual best bid/ask prices and manage the top of book

// Helper to convert price to an index (very dependent on tick size and min price)
__device__ int price_to_index(double price, double min_price, double tick_size) {
    return static_cast<int>((price - min_price) / tick_size);
}

__global__ void process_event_batch_direct_map_kernel(Event* events, int num_events, double min_price, double tick_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_events) {
        Event current_event = events[idx];
        int price_idx = price_to_index(current_event.price, min_price, tick_size);

        // This is still problematic for concurrency if multiple events in the batch
        // affect the SAME price_idx. Needs atomic operations.
        if (current_event.type == 'A') {
            if (current_event.side == 'B') {
                // d_bid_quantities[price_idx] += current_event.quantity; // Needs to be atomic
                atomicAdd(&d_bid_quantities[price_idx], current_event.quantity);
            } else { // Ask
                atomicAdd(&d_ask_quantities[price_idx], current_event.quantity);
            }
        } else if (current_event.type == 'X') { // Cancel
            if (current_event.side == 'B') {
                atomicSub(&d_bid_quantities[price_idx], current_event.quantity); // Ensure it doesn't go negative
            } else { // Ask
                atomicSub(&d_ask_quantities[price_idx], current_event.quantity);
            }
        }
        // 'M' (Modify) would be like a cancel then an add, or more complex atomic update
    }
}
```
This `process_event_batch_direct_map_kernel` started to look more promising. Each thread in a grid could process one event from a batch. The use of `atomicAdd` was crucial here. I remember spending a good day or two wrestling with race conditions before I properly implemented atomic operations for quantity updates. My initial non-atomic attempts led to wildly incorrect book depths when processing concurrent updates to the same price level from synthetic data. `cuda-memcheck` was my constant companion, screaming about race conditions until I got the atomics right.

The biggest challenge with the direct-mapped approach was finding the best bid and offer (BBO). If you just update quantities in a giant array, you then need to scan that array to find the highest price with a non-zero bid quantity, and the lowest price with a non-zero ask quantity. This scan can be slow if the array is large and sparse. I experimented with having a separate small kernel update the BBO after each batch, or even trying to maintain BBO in parallel, which got complicated fast.

I looked into some parallel reduction techniques for finding the max/min price index with quantity. NVIDIA's own CUDA samples have good examples of parallel reductions. I tried to adapt one for finding the top of the book. It involved multiple kernel launches or a more complex single kernel with multiple synchronization points (`__syncthreads()`). This was where I really started to appreciate the subtleties of GPU programming – it's not just about launching many threads, but how they communicate and synchronize.

For generating synthetic tick data, I wrote a simple Python script that produced a stream of events – new orders clustered around a central price, and corresponding cancels. I made sure the data had some bursts to simulate volatile periods. Feeding this data to the GPU involved `cudaMemcpyHostToDevice`. I learned about pinned memory (`cudaHostAlloc`) for faster H2D transfers, which gave a noticeable, though not game-changing, improvement for my batch sizes.

One specific breakthrough happened when I was struggling with the BBO scan after using the direct-mapped array for quantities. The scan was becoming the bottleneck. I remember reading a forum post (I think it was on the NVIDIA developer forums) discussing maintaining a separate, smaller, sorted list of *active* price levels on the GPU, in addition to the direct-mapped array for quantities. The direct-mapped array handles the quick updates, and a more complex kernel (or a series of smaller ones) would manage this smaller sorted list of levels that actually have volume. This seemed like a hybrid approach. Modifying this sorted list in parallel was still tricky.

I then pivoted to a more structured approach for the price levels themselves, limiting the book depth to, say, 100 levels on each side. Each event would be processed by a dedicated thread (or a warp of threads if the event processing itself was complex). For an incoming order, the thread would:
1.  Identify if it's a bid or ask.
2.  Search the existing sorted price levels (a small, dense array representing the visible book). This search could be parallelized or done efficiently by one thread if the number of levels is small (e.g., binary search).
3.  If the price level exists, atomically update the quantity.
4.  If the price level does not exist and the order is aggressive enough to be in the visible book:
    *   This was the hardest part: inserting a new price level and shifting existing ones in a way that's GPU-friendly. I ended up using a temporary buffer in shared memory for a block of threads to cooperatively rebuild the affected portion of the book. This involved careful use of `__syncthreads()`. An order that fell outside the visible depth was simply ignored or aggregated into a "beyond book" counter.

Here's a conceptual snippet for updating a fixed-depth book, focusing on adding an order. This is still very simplified and omits many details like order IDs, modifications, and precise matching logic.

```cpp
// Fixed depth LOB on GPU
__device__ PriceLevel d_gpu_bids[VISIBLE_DEPTH]; // Sorted descending by price
__device__ PriceLevel d_gpu_asks[VISIBLE_DEPTH]; // Sorted ascending by price
__device__ int d_gpu_num_bids;
__device__ int d_gpu_num_asks;

// Kernel to process one new order - simplified for illustration
__global__ void process_new_order_fixed_depth(Event new_order) {
    // This processes ONE order. In reality, you'd batch them.
    // Assume this kernel is launched with enough threads, but logic below is mostly sequential for one order.
    // Parallelism would come from processing MANY such orders simultaneously, each by one or more threads.

    if (new_order.side == 'B') {
        // Attempt to insert into d_gpu_bids
        // 1. Find insertion point or matching level
        int match_idx = -1;
        int insert_pos = 0;
        for (int i = 0; i < d_gpu_num_bids; ++i) {
            if (d_gpu_bids[i].price == new_order.price) {
                match_idx = i;
                break;
            }
            if (d_gpu_bids[i].price < new_order.price) { // Bids sorted high to low
                insert_pos = i;
                break;
            }
            insert_pos = i + 1;
        }

        if (match_idx != -1) {
            atomicAdd(&d_gpu_bids[match_idx].quantity, new_order.quantity);
        } else {
            // Insert new price level if it makes the book and there's space
            // This requires shifting elements, which is tricky in parallel.
            // This naive shift is not good for performance on GPU if done by single thread for many levels.
            if (insert_pos < VISIBLE_DEPTH) {
                // This is a critical section if multiple threads are trying to insert!
                // For a real system, this needs a lock or a lock-free parallel shift.
                // For now, assuming a single thread context for this part of the logic for simplicity of example.
                for (int k = d_gpu_num_bids -1; k >= insert_pos; --k) {
                    if (k + 1 < VISIBLE_DEPTH) {
                        d_gpu_bids[k+1] = d_gpu_bids[k];
                    }
                }
                d_gpu_bids[insert_pos].price = new_order.price;
                d_gpu_bids[insert_pos].quantity = new_order.quantity;
                if (d_gpu_num_bids < VISIBLE_DEPTH) {
                    atomicAdd(&d_gpu_num_bids, 1); // Or manage d_gpu_num_bids carefully
                }
            }
        }
    } else { // Ask side
        // Similar logic for asks, sorted low to high
    }
}
```
The `process_new_order_fixed_depth` kernel above is still conceptual for a *single* event. The real trick was designing how a block of threads would handle a *batch* of events, with each thread potentially taking an event and then cooperatively updating the shared LOB structure. The shifting part was a major pain. I tried having threads in a block each responsible for one level of the book during a shift operation, using shared memory to stage the data. It got complicated, and `__syncthreads()` became my best friend and worst enemy due to subtle bugs if not used perfectly.

Benchmarking was done by generating a large batch of synthetic events (e.g., 1 million), transferring them to the GPU, launching the kernel, and then synchronizing with `cudaDeviceSynchronize()`. I used `std::chrono` on the CPU side to measure the time taken just for the kernel execution (excluding H2D/D2H transfers initially, then including them for a more holistic view). My target was the update speed *per event* within the kernel.

After many iterations, focusing on the fixed-depth LOB with careful parallel processing of event batches (where each thread handled one event and contended for LOB access using atomics or carefully staged updates to the sorted arrays), I started seeing promising numbers. The key was to minimize global memory contention and maximize coalesced memory access when reading the LOB state or writing updates. Using shared memory to cache the top few levels of the book for threads within a block helped somewhat, but the access patterns were still complex.

Eventually, by processing batches of around 1024 to 4096 events, with each event handled by a single thread that performed a binary search on the small (e.g., 100-level) fixed-depth book and then used atomic operations for quantity updates or a carefully managed insertion/deletion for price levels, I achieved update times that, when averaged per event, dipped below one microsecond for the kernel execution phase. This was on synthetic data where events were somewhat uniformly distributed across the top levels of the book. Worst-case scenarios (like many updates needing to insert at the very top, causing many shifts) were slower but still significantly faster than my initial CPU attempts. The RTX 3070 has a decent number of CUDA cores and good memory bandwidth, which certainly helped.

The sub-microsecond figure refers to the average time to process a single event update *within the GPU kernel*, after the data is already on the GPU and before results are copied back. For a batch of 1000 events, if the kernel took 500 microseconds, that's 0.5 microseconds per event.

Lessons learned? CUDA programming has a steep learning curve, especially for problems that aren't embarrassingly parallel. Debugging is tough. Thinking in terms of warps, shared memory, and memory coalescing is crucial. My initial C++ habits of complex objects and pointer-heavy structures had to be completely rethought for the GPU. I also learned that "good enough" can be a valid endpoint for a student project; a perfectly generic, always-fast solution for all LOB scenarios is a massive undertaking.

Future work? I'd love to explore handling individual orders instead of just aggregated price levels, which means managing queues of orders per price level on the GPU. That's another level of complexity, especially for cancellations and modifications that need to find specific orders. Also, integrating market data directly via something like RDMA, bypassing the CPU for data ingress, would be the next step for a truly low-latency system, but that's way beyond a home project for now.

This project was a huge learning experience, pushing my C++ and parallel programming skills significantly. It's one thing to read about GPU architecture and another to wrestle with it to make your specific, complex algorithm fly.