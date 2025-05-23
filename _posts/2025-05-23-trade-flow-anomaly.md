---
layout: post
title: Real-Time Anomaly Detection in Trade Flows
---

## Real-Time Anomaly Detection in Trade Flows: A Deep Dive

This project has been a significant undertaking, pushing my understanding of distributed systems, machine learning, and low-latency processing. The goal was to build a system capable of detecting anomalies in financial trade flows in real-time, specifically targeting sub-millisecond processing for each trade event. The core stack I ended up using was Kafka for message queuing, Flink for stream processing, C++ for the core processing logic within Flink, and a PyTorch-based autoencoder for the anomaly detection itself.

### The Foundations: Kafka, Flink, and C++

The decision to use Kafka wasn't too difficult. Its reputation for handling high-throughput, fault-tolerant event streams is well-deserved. Setting up a local Kafka cluster was relatively straightforward, though I did burn a bit of time initially ensuring my Flink application could correctly connect and consume from the designated topic – a classic case of misconfigured `bootstrap.servers` in the Flink Kafka consumer properties.

Choosing Flink was a more involved decision. I considered other stream processors like Apache Spark Streaming, but Flink's focus on true streaming (event-at-a-time processing) and its robust support for stateful operations seemed better suited for the low-latency requirement. The real challenge, and a conscious one, was deciding to implement the core Flink operators in C++. Flink offers a C++ API, and my hypothesis was that for the sub-millisecond target, I'd need the performance control that C++ provides over Java. This was a gamble, as the C++ API for Flink is less commonly discussed online than its Java or Scala counterparts, meaning fewer readily available solutions for any obscure issues.

My initial Flink job was a simple C++ `FlatMapFunction` just to parse incoming trade data from Kafka. A trade message might look something like this (simplified JSON over Kafka):

`{"trade_id": "t123", "instrument": "EUR/USD", "price": 1.0850, "volume": 100000, "timestamp": 1678886400000}`

The C++ operator needed to deserialize this. I opted for a lightweight JSON parser rather than something heavier to keep overhead minimal.

```cpp
#include <nlohmann/json.hpp> // A popular C++ JSON library
#include "data_types.h" // My custom struct for TradeData

class TradeDeserializer : public org::apache::flink::api::common::functions::FlatMapFunction<std::string, TradeData> {
public:
    void flatMap(std::string value, org::apache::flink::api::java::typeutils::runtime::Collector<TradeData>* out) override {
        try {
            auto json_val = nlohmann::json::parse(value);
            TradeData td;
            td.trade_id = json_val.at("trade_id").get<std::string>();
            td.instrument = json_val.at("instrument").get<std::string>();
            td.price = json_val.at("price").get<double>();
            td.volume = json_val.at("volume").get<long>();
            td.timestamp = json_val.at("timestamp").get<long long>();
            // Some basic validation could happen here too
            out->collect(td);
        } catch (const nlohmann::json::parse_error& e) {
            // Log error, maybe send to a side output
            // std::cerr << "JSON parsing error: " << e.what() << std::endl;
        } catch (const nlohmann::json::out_of_range& e) {
            // Log error for missing fields
            // std::cerr << "JSON missing field error: " << e.what() << std::endl;
        }
    }
};
```
My `TradeData` struct is just a plain C++ struct. Initially, I had issues with Flink's serialization for custom C++ types. The documentation wasn't immediately clear on how to implement custom serializers for the C++ API that would be efficient. I eventually figured out I needed to ensure my types were trivially copyable or provide custom `TypeSerializer` implementations, which added some boilerplate I hadn't anticipated. I stuck with making my structs as simple as possible to leverage Flink's default serializers for basic types where possible.

### The Brains: An Autoencoder in PyTorch

For anomaly detection, I needed an unsupervised approach because labelled anomalous trade data is scarce and often context-dependent. An autoencoder felt like a good fit. The idea is to train a neural network to reconstruct its input. When trained on "normal" trade data, it should become very good at reconstructing normal trades with low error. Anomalous trades, being different from what the model has learned, should result in higher reconstruction errors.

I built the autoencoder using PyTorch. The architecture itself is not overly complex: an encoder that compresses the input features into a lower-dimensional latent space, and a decoder that tries to reconstruct the original features from this latent representation.

Here's a rough idea of the PyTorch model definition in Python:

```python
import torch
import torch.nn as nn

class TradeAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(TradeAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU() # Using ReLU in the bottleneck, though sometimes tanh is used
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            # No activation here if inputs are not bounded or using MSELoss
            # If inputs were, say, normalized to, a Sigmoid might be here
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Example instantiation:
# input_features = 5 # e.g., price, volume, and 3 other derived features
# bottleneck_dimension = 8
# model = TradeAutoencoder(input_features, bottleneck_dimension)
```

Training this was an iterative process. Feature scaling was absolutely critical. Initially, my reconstruction errors were all over the place. I realized that features like `price` and `volume` are on vastly different scales. Normalizing them (e.g., using MinMaxScaler or StandardScaler from scikit-learn *before* feeding them to PyTorch) made a huge difference in convergence. I spent quite a bit of time tweaking the number of layers, neurons per layer, and the `encoding_dim`. Too small an `encoding_dim` and it couldn't reconstruct normal trades well; too large and it would reconstruct anomalies too easily. I used Mean Squared Error (MSE) as the loss function.

### Bridging Worlds: Flink C++ with LibTorch

This was, by far, the most challenging part: getting the C++ Flink operator to call the PyTorch model for inference. PyTorch provides LibTorch, its C++ API. The first step was to serialize my trained PyTorch model into Torch Script format using `torch.jit.trace`.

Loading the Torch Script model in C++ wasn't too bad once I got the CMake configuration right to link against LibTorch. The documentation here is quite good, but integrating it into Flink's build system required some fiddling.

The real headache came with data conversion. My Flink C++ operator receives `TradeData`. This needs to be converted into a `torch::Tensor` to feed into the model. Then, the output tensor needs to be processed to calculate the reconstruction error.

```cpp
// Assume this is part of a Flink RichFlatMapFunction or similar
//
// // At the top of my Flink operator class
// torch::jit::script::Module module;
//
// // In the open() method:
// try {
//     // Deserialize the ScriptModule from a file previously created by torch.jit.save().
//     module = torch::jit::load("path/to/my/traced_trade_autoencoder.pt");
//     module.to(torch::kCPU); // Or torch::kCUDA if I had a GPU and compiled LibTorch for it
//     module.eval(); // Set to evaluation mode
// }
// catch (const c10::Error& e) {
//     // std::cerr << "Error loading the model\n";
//     // Proper error handling / job failure
//     throw std::runtime_error("Failed to load PyTorch model: " + std::string(e.what()));
// }
//
// // In the flatMap() or map() method, after deserializing TradeData:
//
// TradeData trade = ...; // Deserialized trade data
//
// // Prepare input tensor - this was tricky!
// // Assuming 'input_features' is the number of features the model expects
// std::vector<float> input_features_vec;
// // Populate input_features_vec from 'trade' fields, ensuring correct order and scaling
// // This step MUST match the preprocessing done during Python training
// input_features_vec.push_back(static_cast<float>(normalize_price(trade.price)));
// input_features_vec.push_back(static_cast<float>(normalize_volume(trade.volume)));
// // ... add other features
//
// torch::Tensor input_tensor = torch::tensor(input_features_vec).reshape({1, static_cast<long>(input_features_vec.size())});
// input_tensor = input_tensor.to(torch::kCPU); // Ensure it's on the correct device
//
// std::vector<torch::jit::IValue> inputs;
// inputs.push_back(input_tensor);
//
// // Execute the model
// at::Tensor output_tensor = module.forward(inputs).toTensor();
//
// // Calculate reconstruction error (e.g., MSE)
// torch::Tensor diff = input_tensor - output_tensor;
// torch::Tensor mse = torch::mean(diff.pow(2));
// double reconstruction_error = mse.item<double>();
//
// if (reconstruction_error > anomaly_threshold) {
//     // Mark as anomaly, send to an alert stream, etc.
//     // std::cout << "Anomaly detected! Trade ID: " << trade.trade_id << ", Error: " << reconstruction_error << std::endl;
// }
```

One particular "gotcha" was ensuring the input tensor shape and data type (`dtype`) exactly matched what the model expected. LibTorch is quite specific, and a mismatch often leads to cryptic runtime errors. I recall a frustrating afternoon debugging an error that turned out to be because my C++ code was creating a `torch::kDouble` tensor while the model was trained with `torch::kFloat32`. The `normalize_price` and `normalize_volume` functions also had to be C++ reimplementations of the exact same scaling logic I used in Python with scikit-learn. Any deviation here would lead to incorrect reconstruction errors. I actually wrote small unit tests comparing the output of my Python scaling functions and C++ scaling functions with the same inputs to ensure they were identical.

Another subtle issue was thread safety if the model was shared across parallel Flink operator instances. `torch::jit::script::Module` is supposed to be thread-safe for inference, but I was initially cautious and considered creating a model instance per thread (per Flink task slot). For now, a single loaded model per operator instance seems to be working, but for a production system, I'd need to revisit this with more rigorous testing. I consulted several forum posts on the PyTorch forums about LibTorch thread safety to gain some confidence.

### The Quest for Sub-Millisecond Processing

Achieving sub-millisecond processing per event within the Flink C++ operator (excluding network latency to/from Kafka and the PyTorch model inference if it were on a GPU, though I ran it on CPU) required careful attention to detail.
1.  **Minimal Allocations:** In the hot path of the C++ operator, I tried to avoid dynamic memory allocations as much as possible. Object pooling for `TradeData` objects was considered, but Flink's own memory management is quite sophisticated, so I focused on making my user code efficient.
2.  **Efficient Data Structures:** Using `std::vector` for feature preparation before tensor conversion was okay for the small number of features, but if there were many more, I might have looked into more direct ways to construct the tensor from raw data.
3.  **LibTorch on CPU:** The autoencoder I designed is relatively small. Running inference on the CPU via LibTorch was fast enough. If the model were much larger, a GPU would be necessary, which would introduce complexities like data transfer to/from GPU memory and managing CUDA contexts within Flink. For this project, CPU inference kept the deployment simpler.
4.  **Profiling:** I used `perf` on Linux to get a sense of where time was being spent within my C++ operator. Initially, JSON parsing, even with a lightweight library, showed up. This reinforced the need to ensure the input format was as simple as possible if further optimization were needed, perhaps moving to something like FlatBuffers or Cap'n Proto if JSON became a true bottleneck at much higher volumes.
5.  **Flink Configuration:** Tuning Flink's parallelism and ensuring no unnecessary data shuffling or network communication between tasks was also important. For this single operator, it was mostly about ensuring it could process events as fast as they came from Kafka.

A breakthrough moment was when I realized that the `torch::NoGradGuard` scope in C++ could provide a small but noticeable speedup for inference by disabling gradient calculations, which are not needed at inference time.

```cpp
// Inside the map/flatMap function
// ...
// { // Critical section for inference
//     torch::NoGradGuard no_grad; // Disables gradient calculations
//     at::Tensor output_tensor = module.forward(inputs).toTensor();
//     // ... calculate error
// }
// ...
```
It's a small thing, but these accumulate. The actual processing time for my C++ logic plus the LibTorch CPU inference for a single event eventually came down to well under a millisecond on my development machine, typically in the hundreds of microseconds range, which was the target.

### Reflections and Next Steps

This project was a serious lesson in integrating diverse technologies. The journey from raw trade data in Kafka, through C++ processing in Flink, to anomaly scoring with a PyTorch model, was complex but incredibly rewarding. Each component had its learning curve, and making them communicate efficiently was the central challenge. The C++/LibTorch interface, while powerful, requires meticulous attention to data types and tensor shapes.

There are many areas for future work. The anomaly threshold is currently static; a dynamic threshold or a more sophisticated statistical method for setting it would be an improvement. Exploring more complex model architectures or even different unsupervised learning techniques could yield better detection rates. Also, rigorous stress testing of the entire pipeline at much higher data volumes is necessary to identify true bottlenecks.

Overall, I'm pleased with how the system turned out. It's a solid foundation, and the experience gained in low-latency C++ stream processing and deploying PyTorch models in a C++ environment is invaluable.