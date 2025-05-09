---
layout: post
title: Low Latency Data Stream Processor
---

The idea started simple enough: build something that could take data from one place, do a quick check or transformation, and send it somewhere else, really, *really* fast. We'd been learning about stream processing in a few courses, and the concept of low latency had stuck with me. Most examples felt a bit abstract, dealing with theoretical message rates and processing times. I wanted to build something tangible, something where I could actually *measure* the delay end-to-end. That led to the "Low Latency Data Stream Processor" project.

The core requirement was processing messages arriving at potentially high rates with sub-millisecond latency. This immediately ruled out anything synchronous that might involve waiting on I/O operations like database lookups or network calls for too long within the main processing path. Asynchronous programming seemed like the obvious fit, and having used Python a fair bit, `asyncio` was the natural choice.

For messaging, Kafka was something I’d encountered in course material and seemed like the industry standard for high-throughput, fault-tolerant message queues. It felt like a solid, realistic choice compared to simpler alternatives. Plus, setting up a Kafka cluster locally (even a simple one with Docker) felt like a good learning exercise.

The processing itself needed to be fast. The typical use case I envisioned involved receiving a message, looking up some related information, doing a quick calculation based on the message content and the looked-up data, and then maybe forwarding a result. That lookup part was the potential killer for latency. A traditional database query, even a fast one, adds milliseconds. We needed something in-memory. Redis fit perfectly here. It's fast, supports simple key-value lookups, and has Python clients that support asyncio.

So, the stack coalesced: Kafka for ingestion and distribution, `asyncio` for concurrent, non-blocking processing, and Redis for low-latency data lookups.

Getting started with `asyncio` and Kafka was the first hurdle. I initially looked at the official `kafka-python` library, but it's primarily synchronous. Running its blocking methods (like `consumer.poll()`) directly in an `asyncio` loop would defeat the purpose of async. This led me to `aiokafka`, an asyncio-compatible client. The documentation was decent, but integrating it correctly into an event loop structure took some trial and error.

My initial thought process for the consumer loop in asyncio was something like this:

```python
import asyncio
from aiokafka import AIOKafkaConsumer

async def consume():
    # Initial rough idea - probably wrong
    consumer = AIOKafkaConsumer(
        'my_topic',
        bootstrap_servers='localhost:9092',
        group_id='my_processor_group'
    )
    await consumer.start()
    try:
        while True:
            # This felt too simple... is poll() async?
            msg = await consumer.getone() # Ah, getone() exists!
            print(f"Received message: {msg.value}")
            # Process the message here... this needs to be fast!
            processed_data = process_message(msg.value) # Need to make this async too?
            # Maybe send to another topic?
            # await producer.send_and_wait('output_topic', processed_data)
    finally:
        await consumer.stop()

# asyncio.run(consume()) # How to run multiple things?
```

This initial sketch immediately brought up questions. How do I run the consumer *and* potentially other tasks (like a health check server or metric reporting) in the same loop? How do I handle batches of messages instead of one at a time, which `aiokafka` supports with `getmany()`? And crucially, how do I integrate the processing logic and the Redis lookups into this async flow without blocking?

The `getmany()` method seemed more efficient for throughput, but I was hyper-focused on *latency*. Processing messages one by one as they arrived felt more aligned with the low-latency goal, minimizing the time a message spends waiting in a buffer. So, I stuck with `getone()` for the core processing loop, at least initially, reasoning that processing smaller units immediately was better for minimizing tail latency, even if it meant slightly higher per-message overhead compared to batching. This was a specific design trade-off I made based on the primary objective being *latency* over pure *throughput*.

The processing logic involved a lookup in Redis. This meant I needed an asyncio-compatible Redis client. `redis-py` has an experimental asyncio interface, but `aioredis` (now merged back into `redis-py`'s async part) was a more mature option at the time I started.

Integrating Redis looked something like this:

```python
import asyncio
from aiokafka import AIOKafkaConsumer
import redis.asyncio as redis # Using the modern redis-py async

async def process_message(msg_value, redis_client):
    # Assume msg_value is JSON bytes like { "key": "some_id", "value": 123 }
    import json
    try:
        data = json.loads(msg_value)
        lookup_key = data.get("key")
        if not lookup_key:
            print(f"Message missing 'key': {msg_value}")
            return None

        # This is the critical async lookup
        cached_data = await redis_client.get(lookup_key)

        if cached_data:
            print(f"Cache hit for key {lookup_key}")
            # Do something with cached_data and data['value']
            # ... calculation ...
            result = f"processed_{lookup_key}_{data['value']}_{cached_data.decode()}"
            return result.encode('utf-8') # Need to send bytes back?

        else:
            print(f"Cache miss for key {lookup_key}. Requires external lookup? (Skipping for now)")
            # In a real scenario, might fetch from DB asynchronously here
            return None # Or some indicator it needs further processing

    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {msg_value}")
        return None
    except Exception as e:
        print(f"Error processing message {msg_value}: {e}")
        return None


async def consume_and_process():
    consumer = AIOKafkaConsumer(
        'input_topic',
        bootstrap_servers='localhost:9092',
        group_id='my_processor_group',
        auto_offset_reset='latest', # Start from the latest message
        enable_auto_commit=True # Let Kafka handle commits for simplicity initially
    )
    # Need a Redis connection pool? Or just a single client?
    # Let's start simple with a single client connection for now.
    # This might need revisiting for higher load.
    redis_client = redis.Redis(host='localhost', port=6379, db=0)

    await consumer.start()
    print("Consumer started")

    try:
        while True:
            try:
                # Adjusted timeout - don't wait forever if no messages
                msg = await asyncio.wait_for(consumer.getone(), timeout=1.0)
                # print(f"Received message: {msg.value}") # Too noisy?
                processed_result = await process_message(msg.value, redis_client)

                if processed_result:
                    # Assuming we need to send it somewhere, e.g., another Kafka topic
                    # Need a producer instance too!
                    pass # Placeholder for sending results

            except asyncio.TimeoutError:
                # No messages in the last 1 second, loop continues
                # print("No messages received in 1 second") # Maybe log less often
                pass
            except Exception as e:
                print(f"Error in main consumer loop: {e}")
                # Decide on error handling: skip message? log and retry?

    finally:
        print("Stopping consumer")
        await consumer.stop()
        # Close redis connection? aioredis handles connections in its pool normally.
        # Single client might not need explicit close in this simple case, but good practice.
        await redis_client.close()


# How to wire this up with a producer and handle gracefully shutting down?
# Need a main async function to run everything.
# async def main():
#     producer = AIOKafkaProducer(...)
#     await producer.start()
#     consumer_task = asyncio.create_task(consume_and_process(producer))
#     # Add other tasks?
#     await consumer_task # Wait for the consumer to finish (which it won't in this loop)
#     # Need proper signal handling to stop the loop and tasks.
#
# if __name__ == "__main__":
#     asyncio.run(main())
```

This version felt more structured. I had separated the processing logic into its own async function, taking the Redis client as an argument. I used `asyncio.wait_for` with a timeout on `getone()` so the loop wouldn't just block indefinitely if the input topic was empty; it could yield control back to the event loop.

The major challenge was measuring latency accurately. "Sub-millisecond" is a tight target. Where do you measure from and to? I defined it as the time from when a message is *sent* to the input Kafka topic to the moment the *processed result* is ready (or sent to an output topic). This required including a timestamp in the original message payload and logging timestamps at various stages of processing.

Initially, I just used `time.perf_counter()` around the `process_message` call.

```python
# Inside the consume_and_process loop, after getting msg
# import time
# ...
# start_time = time.perf_counter()
# processed_result = await process_message(msg.value, redis_client)
# end_time = time.perf_counter()
# processing_duration_ms = (end_time - start_time) * 1000
# print(f"Processing duration: {processing_duration_ms:.3f} ms")
```

This only measured the *processing* time *within* my consumer. It didn't account for the time the message spent in Kafka queues or network travel. To measure end-to-end latency, I had to embed a timestamp at the source.

Let's say the source message looked like `{"id": "abc", "value": 100, "timestamp_sent": <unix_epoch_ms>}`.

My processing logic would then become:

```python
async def process_message(msg, redis_client): # Pass the full msg object
    import json
    # ... JSON decoding ...
    try:
        data = json.loads(msg.value)
        timestamp_sent = data.get("timestamp_sent")
        if timestamp_sent is None:
             print(f"Message missing timestamp_sent: {msg.value}")
             # Decide how to handle - skip? Add current time?
             timestamp_sent = int(time.time() * 1000) # Add current time as a fallback?

        lookup_key = data.get("key")
        # ... rest of the processing ...

        # After processing is complete (and result is ready)
        # Calculate latency
        current_time_ms = int(time.time() * 1000)
        e2e_latency_ms = current_time_ms - timestamp_sent
        print(f"Message ID {data.get('id', 'N/A')} E2E Latency: {e2e_latency_ms} ms")

        # ... return processed_result ...

    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {msg.value}")
        return None
    except Exception as e:
        print(f"Error processing message {msg.value}: {e}")
        return None
```

This was better, but still relied on system clocks being perfectly synchronized, which isn't always reliable. For a student project though, and running everything on `localhost` initially, it was a reasonable approximation. If I were doing this professionally, I'd look into dedicated tracing tools or Kafka's own timestamp features more deeply.

Optimizing for sub-millisecond latency was iterative.
1.  **Initial implementation:** Just getting the async Kafka consumer and async Redis client working together. Latency was higher, maybe 5-10 ms on average, with spikes.
2.  **Profiling:** Using Python's `cProfile` or even just adding strategically placed `time.perf_counter()` calls helped identify where time was being spent. Turns out, initial JSON decoding and even printing to the console were relatively slow. Removed unnecessary prints in the hot path.
3.  **Redis Connection:** Was a single Redis connection sufficient? With `aioredis`, a single client typically manages a connection pool behind the scenes, which is good. But ensuring the connection setup wasn't happening *per message* was important. The client should be initialized once and reused.
4.  **Serialization/Deserialization:** JSON parsing adds overhead. For maximum performance, using something like `ujson` instead of the standard `json` library, or even considering binary formats like Protocol Buffers or MessagePack, could yield gains. For this project, sticking with standard `json` was simpler and sufficient after other optimizations, aligning with the "student project, realistic constraints" idea – sometimes the easiest tool is good enough *after* bottleneck analysis.
5.  **Kafka Configuration:** `auto_offset_reset='latest'` meant I wasn't reprocessing old messages on startup, which is good for a real-time processor. `enable_auto_commit=True` was the simplest way to handle commits, though for guaranteed exactly-once processing (not a primary goal here, latency was), manual commits would be necessary. The `poll()` timeout in `getone()` was also important; a very short timeout means the loop spins more often checking for messages, potentially using more CPU but reacting faster; a longer timeout uses less CPU when idle but adds potential delay if a message arrives just after a poll returns empty. I settled on 1ms timeout (`timeout=0.001`) during peak optimization, which felt aggressive but necessary for the target. This looked like:

```python
# Inside the consume_and_process loop
while True:
    try:
        # Aggressive timeout for low latency
        msg = await asyncio.wait_for(consumer.getone(), timeout=0.001)
        # ... process message ...
    except asyncio.TimeoutError:
        # Expected frequently, just means no messages in this 1ms window
        pass
    except Exception as e:
        print(f"Error in main consumer loop: {e}")
        # ... error handling ...
```

This aggressive polling felt a bit like busy-waiting, but with `asyncio`, the `await` on `getone` *does* yield control back to the event loop. The `timeout` just dictates how long `wait_for` will wait before raising `TimeoutError` if `getone` doesn't return. It's not a tight CPU loop unless the event loop itself is overloaded.

Achieving consistent sub-millisecond latency was tough. On my local machine, without much other load, I could get average latencies well under 1ms for the core processing path (after the message was received by the consumer). The end-to-end number, including Kafka queue time, was harder to control and measure precisely without dedicated tools. However, for simple messages and a warm Redis cache, the path *from consumer receiving the message to processed result being ready* could reliably be brought under 1ms. This felt like a significant achievement, demonstrating that the `asyncio` + Redis approach worked for low-latency in-memory lookups within the processing pipeline.

The project taught me a lot about the practicalities of asynchronous programming, the challenges of real-time data processing, and the importance of careful measurement and profiling when chasing performance targets like sub-millisecond latency. While a production system would need more robust error handling, monitoring, scaling considerations, and potentially a more sophisticated latency measurement framework, this project provided a solid foundation and proved that achieving such targets is possible with the right tools and careful implementation. It wasn't just theoretical anymore; I had the logs and the timing data to prove it.