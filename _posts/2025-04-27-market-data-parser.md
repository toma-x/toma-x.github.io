---
layout: post
title: Low-Latency Market Data Parser
---

Low-Latency Market Data Parser Project

Okay, just wrapped up this C++20 project focusing on building a low-latency parser for simulated market data. The main goal was pretty straightforward: process incoming data from a network source as fast as possible, aiming for parsing latency well under a microsecond per message if achievable. This was a significant step up in complexity and performance requirements compared to the Python scripting I've done before.

The core challenge was handling binary data arriving over a network connection and transforming it into structured C++ objects with minimal delay. The simulated exchange protocol had variable-length messages defined by a header indicating message type and size, followed by the payload. Speed was paramount, so dynamic allocation and frequent copying were out.

Choosing C++20 was a clear decision early on. For this kind of performance requirement, compiled code is necessary, and C++ gives the low-level control needed for memory and network interactions. C++20 features weren't heavily used for core logic, but concepts like `std::span` for buffer views or ranges for potential future processing steps were nice to have available. The bulk of the work relied on standard library containers, pointers, and external libraries.

For network communication, I went with **Boost.Asio**. Handling asynchronous network I/O efficiently is tricky, and Asio is pretty much the standard C++ library for this. Initially, I struggled quite a bit with Asio's asynchronous model. Understanding how `io_context`, handlers, and buffers interact took time. My first attempts involved simple synchronous reads, which blocked and killed performance. Switching to `async_read` or `async_read_some` with completion handlers was the right path, but managing the read buffer state across multiple asynchronous operations wasn't intuitive immediately.

Here's a simplified snippet showing the async read loop pattern I settled on after some trial and error. The key was having a persistent buffer and logic in the handler to check if a full message was received.

```cpp
// Inside my connection handler class

std::vector<char> m_read_buffer; // A buffer to accumulate incoming data
size_t m_buffer_offset = 0;     // Where the next incoming data should be written

void start_read() {
    // Resize buffer if needed, maybe use a fixed pool later?
    if (m_read_buffer.size() < 4096) { // Arbitrary initial size
         m_read_buffer.resize(4096);
    }

    // Use async_read_some into the *available* part of the buffer
    // This was a point of confusion - making sure the buffer points to the correct
    // spot after previous partial reads.
    auto buffer_span = asio::buffer(m_read_buffer.data() + m_buffer_offset,
                                    m_read_buffer.size() - m_buffer_offset);

    m_socket.async_read_some(buffer_span,
        std::bind(&ConnectionHandler::handle_read, this,
                  std::placeholders::_1, std::placeholders::_2));
}

void handle_read(const boost::system::error_code& error, size_t bytes_transferred) {
    if (!error) {
        m_buffer_offset += bytes_transferred; // Advance offset

        // Now, process the accumulated data in m_read_buffer up to m_buffer_offset
        // This involves checking for full messages based on the protocol header.
        // My parsing logic sits here.
        size_t bytes_processed = parse_buffer(m_read_buffer.data(), m_buffer_offset);

        // Shift remaining data to the beginning of the buffer
        if (bytes_processed > 0) {
             // This memmove/copy was something I initially forgot, leading to data loss.
             // Or I tried resizing/erasing the vector which was too slow.
             // memmove is the way for overlapping copies.
            std::memmove(m_read_buffer.data(),
                         m_read_buffer.data() + bytes_processed,
                         m_buffer_offset - bytes_processed);
            m_buffer_offset -= bytes_processed;
        }

        // Continue reading
        start_read();

    } else {
        // Handle error... socket closed, etc.
        std::cerr << "Read error: " << error.message() << std::endl;
    }
}

// parse_buffer function signature (implementation details below)
// size_t parse_buffer(const char* data, size_t size);
```

The `parse_buffer` function was the core logic for deserializing the binary data. The simulated protocol header had a fixed size (e.g., 8 bytes) containing message type and total message length. The parsing function needed to:
1.  Check if at least the header size is available in the buffer.
2.  Read the header (careful about endianness!).
3.  Determine the full message length from the header.
4.  Check if the full message is available in the buffer.
5.  If yes, deserialize the fields from the buffer into a C++ message struct/object. This involved casting pointers or using bit shifts and masks for packed fields. Again, endianness was a common source of bugs. I spent frustrating time debugging why values were incorrect, only to realize I'd assumed host endianness instead of handling network byte order (`ntohl`, `ntohs` were essential).
6.  If a message is successfully parsed, return the number of bytes consumed so the read buffer can be shifted.
7.  If not enough data for a full message, return 0 and wait for more data.

```cpp
// Simplified parsing function structure
size_t ConnectionHandler::parse_buffer(const char* data, size_t size) {
    size_t total_processed = 0;
    while (size - total_processed >= sizeof(MessageHeader)) {
        const auto* header = reinterpret_cast<const MessageHeader*>(data + total_processed);

        // Check for minimum header size first
        if (size - total_processed < sizeof(MessageHeader)) {
             break; // Not even enough for a header
        }

        // Need to handle endianness! Assuming network byte order (big-endian)
        uint16_t msg_type = ntohs(header->message_type);
        uint16_t msg_len = ntohs(header->message_length); // This is total length including header

        // Sanity check the reported length
        if (msg_len < sizeof(MessageHeader)) {
            // Protocol error: length too small. How to handle? Skip or disconnect?
            // Initially I didn't handle this and crashed. Added robust checks later.
            std::cerr << "Protocol error: Invalid message length" << std::endl;
            // For now, maybe skip this header and try the next bytes? Or disconnect.
            total_processed++; // Skip a byte and re-evaluate. Crude but prevents infinite loops.
            continue;
        }


        // Check if the full message is in the buffer
        if (size - total_processed < msg_len) {
            break; // Not enough data for the full message yet
        }

        // Okay, full message is here. Parse based on type.
        // This required a big switch statement or map based on msg_type.
        // Deserialization logic here involved reading bytes from (data + total_processed + sizeof(MessageHeader))
        // and populating a C++ struct. Pointers and `memcpy` were useful here, avoiding stream overhead.
        // E.g., for a trade message:
        // if (msg_type == MSG_TYPE_TRADE) {
        //     if (msg_len != sizeof(TradeMessageStruct)) { // Fixed size message example
        //         std::cerr << "Protocol error: Trade message wrong size" << std::endl;
        //         total_processed += msg_len; // Skip malformed message
        //         continue;
        //     }
        //     auto* trade_msg = reinterpret_cast<const TradeMessageStruct*>(data + total_processed);
        //     // Process trade_msg... maybe pass it to a consumer thread/queue
        //     process_trade_message(trade_msg); // This needs to be fast or offloaded!
        // } else if ... // Other message types

        total_processed += msg_len; // Advance past the consumed message
    }
    return total_processed; // Return total bytes successfully parsed
}
```

A major constraint for achieving sub-microsecond latency was avoiding standard dynamic memory allocation (`new`/`delete` or `std::vector` growth) within the critical path of parsing and initial processing. These operations can involve locks or complex internal logic that adds unpredictable latency. This led me down the path of **custom allocators**.

I decided on a simple **memory pool** allocator. The idea is to pre-allocate a large block of memory and then dole out fixed-size chunks from it very quickly. When a message object is "freed", its chunk is returned to a list of available chunks rather than being returned to the system heap. This bypasses the overhead of `new`/`delete`.

Implementing a thread-safe memory pool that works correctly with varying message sizes (even though the *chunks* might be fixed, the data *within* them varies) was challenging. My initial pool only handled one size, which wasn't practical for different message types. I considered multiple pools for different sizes or a more complex arena allocator, but settled on a fixed-size pool where the size was large enough for the biggest expected message, accepting some internal fragmentation for smaller messages, as allocation speed was the priority. Debugging allocator issues – like double frees or using freed memory – was particularly painful, often leading to crashes far removed from the actual error site. Stack traces in C++ can be illuminating but also daunting when dealing with memory corruption. Reading articles and StackOverflow threads on C++ memory pool implementations was crucial here.

Integrating the custom allocator with the Asio read buffer was also tricky. Asio likes its own buffers (`asio::buffer`). I couldn't easily tell Asio to read directly into a chunk from my custom pool without writing a custom `MutableBufferSequence`. For this version, I compromised: Asio reads into a standard `std::vector` buffer, and then my parser copies (or ideally, processes in-place using pointer manipulation) the data from this temporary buffer into objects allocated from my pool. This copy adds a tiny bit of overhead but simplifies the Asio interaction significantly compared to writing a custom buffer sequence. For the target latency, this compromise seemed acceptable after measurement.

Measuring latency was critical. I used `std::chrono::high_resolution_clock` to timestamp packets immediately upon receiving them and again after they were fully parsed and validated. Calculating the difference gave the processing latency. Averaging this over thousands or millions of messages gave a clearer picture than single-message timings. This revealed bottlenecks I hadn't anticipated, like excessive data copying or inefficient checks within the parsing loop. Optimizing involved profiling (simple timing points or more advanced tools if available) and refining the parsing logic to minimize branches and memory access.

One specific headache was correctly handling partial messages and messages spanning multiple network packets. My `handle_read` function had to correctly append data, check if a full message was present *at the beginning* of the accumulated buffer, parse all available full messages sequentially, and then shift only the remaining partial data. Getting the index tracking (`m_buffer_offset` and `bytes_processed`) right took several iterations and careful walkthroughs with example data.

Overall, this project reinforced the importance of low-level details when performance is critical. Choosing the right tools (Boost.Asio, custom allocation) is just the start; the real work is in the careful implementation, understanding data representation (binary, endianness), and relentless measurement and optimization. It was a challenging but incredibly rewarding dive into performance-oriented C++ development.