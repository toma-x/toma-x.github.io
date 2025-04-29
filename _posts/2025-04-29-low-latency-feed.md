---
layout: post
title: Low-Latency Market Data Feed
---

Low-Latency Market Data Feed Parser: A Deep Dive

Building a market data feed handler has been a significant project, driven by the need for speed and reliability in processing real-time financial data. The goal was ambitious: construct a C++ application capable of receiving, parsing, and distributing market data from a high-volume UDP multicast feed with minimal latency, ideally below 10 microseconds from wire to output. This isn't just about getting the data; it's about getting it *fast* enough to be useful for latency-sensitive applications.

The feed uses UDP multicast, which is standard for distributing data to multiple consumers efficiently without overwhelming the source. C++ was the natural choice for performance. I decided to leverage Boost libraries where possible, specifically Boost.ASIO for network I/O, as it provides a robust framework for asynchronous operations.

Setting up the UDP receiver seemed straightforward initially. Boost.ASIO's `ip::udp::socket` and `async_receive_from` functions are well-documented.

```cpp
// Simplified excerpt from receiver setup
boost::asio::io_context io_context;
boost::asio::ip::udp::socket socket_(io_context);
boost::asio::ip::udp::endpoint listen_endpoint(
    boost::asio::ip::address::from_string("0.0.0.0"), // Listen on all interfaces
    data_port);

socket_.open(listen_endpoint.protocol());
socket_.set_option(boost::asio::ip::udp::socket::reuse_address(true));
socket_.bind(listen_endpoint);

// Join the multicast group
socket_.set_option(
    boost::asio::ip::multicast::join_group(
        boost::asio::ip::address::from_string(multicast_group_ip)));

// Prepare buffer for incoming data
std::vector<char> recv_buffer_(65536); // Max UDP packet size

void do_receive() {
    socket_.async_receive_from(
        boost::asio::buffer(recv_buffer_), remote_endpoint_,
        [&](boost::system::error_code ec, std::size_t bytes_recvd) {
            if (!ec) {
                // Packet received! Now what?
                // Initial thought: process here or copy and process elsewhere?
                // Copying seems safe but slow. Processing here blocks receiver.
                handle_packet(recv_buffer_.data(), bytes_recvd); // <--- This was a bottleneck initially
                do_receive(); // Continue receiving
            } else {
                // Handle error
                std::cerr << "Receive error: " << ec.message() << std::endl;
            }
        });
}

// Start receiving loop
// io_context.run();
```

The first challenge was handling the raw packet data. The feed uses a custom binary protocol. This meant no standard parsing libraries would work directly; I had to manually decode the bytes. This involved reading specific offsets, understanding integer sizes (little-endian vs. big-endian), and extracting fields. My initial approach involved reading bytes and casting pointers or using `memcpy` into C++ structs designed to match the protocol layout.

```cpp
// Example: Attempting to map a packet header struct
#pragma pack(push, 1) // Ensure no padding
struct PacketHeader {
    uint16_t message_type;
    uint16_t message_length; // Length includes header
    uint32_t sequence_number;
    uint64_t timestamp_ns;
    // ... more fields
};
#pragma pack(pop)

void handle_packet(const char* data, size_t size) {
    if (size < sizeof(PacketHeader)) {
        // Too small, drop or log error
        return;
    }
    const PacketHeader* header = reinterpret_cast<const PacketHeader*>(data);
    // Now process header->message_type, header->message_length, etc.
    // Need to handle endianness! My first attempts forgot about this entirely.
    // Fixed: Add functions like boost::endian::load_little_endian() or similar manual shifts.

    const char* message_body_ptr = data + sizeof(PacketHeader);
    size_t body_size = size - sizeof(PacketHeader);

    // Need to parse the body based on header->message_type
    // This part got complex quickly with many message types.
    // How to do this without lots of copies or branching overhead?
    // Decided to use a jump table or map of function pointers later for dispatch.
}
```

Serialization wasn't needed for receiving, but for pushing data out. Standard serialization frameworks felt too heavyweight for the low-latency requirement. A custom binary format for output was decided upon, designed for speed and minimal size, directly reflecting the parsed market data structures. This involved writing bytes directly into an output buffer.

Throughput quickly became an issue. A single thread receiving and parsing couldn't keep up with the high volume of packets (>5 Gbps means millions of packets per second). The `handle_packet` function was blocking the ASIO event loop, causing packet drops and increased latency as the socket buffer overflowed.

The solution was multithreading. The receiver thread should do *only* receiving and immediately pass the raw packet data to one or more worker threads for parsing and processing. How to pass data between threads? A queue.

My initial thought was a simple `std::queue<std::vector<char>>` protected by a `std::mutex` and using a `std::condition_variable` to signal workers.

```cpp
// Early multi-threading attempt (simplified, mutex/condition_variable not shown)
std::queue<std::vector<char>> packet_queue;
std::mutex queue_mutex;
// std::condition_variable queue_cond;

void do_receive() {
    socket_.async_receive_from(
        boost::asio::buffer(recv_buffer_), remote_endpoint_,
        [&](boost::system::error_code ec, std::size_t bytes_recvd) {
            if (!ec) {
                // Problem: Copying the packet here creates latency and allocation pressure
                // std::vector<char> packet_copy(recv_buffer_.begin(), recv_buffer_.begin() + bytes_recvd);
                // {
                //     std::lock_guard<std::mutex> lock(queue_mutex); // Lock!
                //     packet_queue.push(packet_copy);
                // }
                // queue_cond.notify_one();

                // Need zero-copy or minimal copy.
                // Maybe pass a shared_ptr to a buffer pool object?
                // Or queue raw pointers and manage lifetimes carefully? That felt risky.

                do_receive();
            } // ... error handling
        });
}

// Worker thread function (simplified)
void process_packets() {
    while(running_) {
        std::vector<char> packet;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // queue_cond.wait(lock, [&]{ return !packet_queue.empty() || !running_; });
            if (!packet_queue.empty()) {
                packet = std::move(packet_queue.front());
                packet_queue.pop();
            } else if (!running_) {
                break;
            }
        }
        if (!packet.empty()) {
            // handle_packet(packet.data(), packet.size()); // Process the packet
        }
    }
}
```

Profiling this setup under heavy load revealed significant contention on the mutex protecting the queue. The receiver was trying to push packets onto the queue much faster than workers could pull them off, leading to lock waits and latency spikes. For a low-latency system, predictable performance is key, and mutexes introduce unpredictable delays.

This led me to research lock-free data structures. Boost.Lockfree provided `boost::lockfree::queue`. This seemed ideal for a single producer (receiver) and multiple consumers (workers) scenario. It allowed pushing from the receiver and popping from workers without explicit locks, relying on atomic operations.

```cpp
// Refactored: Using Boost.Lockfree queue
#include <boost/lockfree/queue.hpp>

// Need to queue data, perhaps pointers to buffers?
// Let's use a queue of raw pointers for zero-copy potential, managing buffer lifetimes carefully.
// Need a buffer pool or similar.
// boost::lockfree::queue<PacketBuffer*, boost::lockfree::fixed_sized<false>> packet_lf_queue(1024); // Not fixed_sized initially?
// Or better, fixed size for performance predictability?

// Assuming a BufferManager provides PacketBuffer* and reclaims them
// In receiver:
// PacketBuffer* buf = buffer_manager.acquire_buffer();
// memcpy(buf->data, recv_buffer_.data(), bytes_recvd);
// buf->size = bytes_recvd;
// while (!packet_lf_queue.push(buf)) { /* queue full? wait or drop? Decide policy */ }

// In worker:
// PacketBuffer* buf = nullptr;
// if (packet_lf_queue.pop(buf)) {
//     handle_packet(buf->data, buf->size);
//     buffer_manager.release_buffer(buf); // Return buffer to pool
// }

// This structure removed the mutex bottleneck, significantly improving throughput and reducing latency jitter.
```

Implementing the zero-copy packet handling with a buffer pool and the Boost.Lockfree queue was a major breakthrough. The receiver could quickly copy the incoming packet data into a pre-allocated buffer from the pool and push the buffer's pointer onto the lock-free queue. Worker threads would pop pointers from the queue, process the data *in place* within that buffer, and then return the buffer to the pool. This avoided expensive `std::vector` copies and allocations on the hot path.

Achieving sub-10 microsecond latency wasn't just about the queue. It also involved optimizing the parsing logic itself. Accessing memory efficiently became critical. Struct layouts were reviewed to ensure frequently accessed fields were close together for better CPU cache locality. Parsing loops were profiled intensely. Manual byte manipulation was often faster than using streams or more abstract methods for this specific binary format.

The parsed data needed to be disseminated. ZeroMQ was chosen for its message patterns (specifically PUB/SUB) which are suitable for distributing the processed market data to multiple subscribers (e.g., trading algorithms). Integrating ZeroMQ involved setting up a context and a publisher socket in a thread and pushing the serialized output data onto it.

```cpp
// ZeroMQ publisher setup (simplified)
#include <zmq.hpp>

// zmq::context_t context(1);
// zmq::socket_t publisher(context, zmq::socket_type::pub);
// publisher.bind("tcp://*:5555"); // Or ipc:// or epgm:// for multicast output

// In a thread, perhaps the worker after parsing:
// void publish_market_data(const ParsedTrade& trade) {
//     // Serialize the parsed data into a buffer
//     std::vector<char> serialized_data = serialize_trade(trade); // Custom serialization

//     zmq::message_t message(serialized_data.data(), serialized_data.size());
//     // How to handle slow subscribers? ZeroMQ HWM (High Water Mark) options.
//     // publisher.send(message, zmq::send_flags::dontwait); // Non-blocking send
//     // Need to handle EWOULDBLOCK if queue is full. Maybe drop messages?
// }
```

The entire process involved continuous profiling and tuning. Tools like `perf` on Linux were invaluable for identifying CPU hotspots. Memory allocators (`tcmalloc`, `jemalloc`) were considered, though the buffer pool approach minimized dynamic allocations during runtime. Socket options for Boost.ASIO also needed tuning (e.g., buffer sizes).

Achieving >5 Gbps throughput and consistent sub-10 microsecond latency required iterating on the design multiple times, particularly the handoff between the receiver and workers, and the in-place processing in the worker threads. The combination of Boost.ASIO for efficient network handling, a lock-free queue with a buffer pool for fast inter-thread communication, careful zero-copy binary parsing, and ZeroMQ for output distribution proved effective.

This project was a deep dive into the complexities of low-level performance optimization in C++. It reinforced the importance of understanding memory access patterns, the overhead of locking in concurrent systems, and the need for precise profiling to identify bottlenecks. It was challenging, requiring careful resource management and attention to detail, but the result is a robust and high-performance data feed handler.