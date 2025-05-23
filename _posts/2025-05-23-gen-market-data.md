---
layout: post
title: Generative Market Data Engine
---

## Building a Generative Market Data Engine: My Journey into Synthetic Tick Data

For the past few months, I've been pretty heads-down on a personal research project that’s been both incredibly frustrating and immensely rewarding: building a Generative Market Data Engine. The goal was to generate high-fidelity synthetic tick data, the kind that mirrors what you’d see from real financial exchanges. Specifically, I was targeting the characteristics of LOBSTER limit order book data. After a lot of trial and error, I finally managed to get the synthetic data to show over 95% statistical feature parity with the real LOBSTER samples, which I'm pretty stoked about. This post is a bit of a brain dump of that process, the roadblocks, and what I learned.

### The "Why" and the Initial Dive

The whole idea started from a fascination with market microstructure and the limitations of historical data. Backtesting trading strategies or doing research on market dynamics often requires vast amounts of data, and good quality tick data can be expensive or hard to come by, especially for specific conditions or less common stocks. LOBSTER is a great resource, but I wondered if I could generate statistically similar data on demand. This led me down the rabbit hole of generative models, and specifically, Generative Adversarial Networks (GANs).

Python and PyTorch were my tools of choice. I'm fairly comfortable with Python for data manipulation, and PyTorch's flexibility seemed like a good fit for the experimental nature of GANs.

### Grappling with LOBSTER Data

Before I could even think about generating data, I had to intimately understand the LOBSTER data format. It’s incredibly detailed, providing a chronological log of limit order book events – submissions, cancellations, executions. Each message has a timestamp, event type, order ID, price, quantity, and direction. My first task was to parse these message files and reconstruct snapshots of the order book. This was more challenging than I initially anticipated. The sheer volume of data for even a single day is massive.

I spent a good week just writing scripts to process the raw LOBSTER files into a more usable format. I decided to focus on a few key features for each time step: mid-price, spread, and the volume imbalance across a few levels of the book. Normalization was also a big question. Financial time series are non-stationary, so I ended up using fractional differentiation on price series to try and achieve stationarity, and min-max scaling on other features like volume, calculated over rolling windows from the training set. This felt like a bit of a hack, and I worried about look-ahead bias, but for a first pass, it seemed manageable.

```python
# Rough idea of how I was thinking about features
# This isn't runnable as-is, just conceptual

def process_lobster_chunk(df_messages):
    # df_messages would be a pandas DataFrame from LOBSTER
    # ... lots of logic to reconstruct order book states ...
    # For each timestamp, I wanted to extract:
    # best_bid, best_ask = get_best_bid_ask(current_book_state)
    # mid_price = (best_bid + best_ask) / 2.0
    # spread = best_ask - best_bid
    
    # For imbalance, sum of volumes at first N levels
    # bid_vol_level1 = current_book_state.bids.volume
    # ask_vol_level1 = current_book_state.asks.volume
    # imbalance = (bid_vol_level1 - ask_vol_level1) / (bid_vol_level1 + ask_vol_level1) # simplified
    
    # This would then be assembled into sequences
    # features_at_t = [mid_price_change, normalized_spread, normalized_imbalance]
    pass
```

### Choosing the GAN Architecture: Not So Straightforward

My initial thought was to try a relatively simple GAN, maybe something like a DCGAN but adapted for 1D time series data using `Conv1d` layers. I spent a couple of weeks on this. The results were… not great. The generated sequences looked like random noise, and I struggled with the classic GAN problems: mode collapse, where the generator produces the same few outputs, and the discriminator loss plummeting to zero, indicating it had learned to distinguish fakes perfectly and the generator wasn't learning anything.

I went back to the drawing board and started reading more about GANs for time series data. The TimeGAN paper (Yoon, Jarrett, & van der Schaar, 2019) kept coming up. Its architecture, with the explicit supervised loss component and the autoencoder-like structure to learn the temporal dynamics in a latent space, seemed much more suited to this problem. The idea of jointly training an autoencoder with the adversarial components to ensure the latent space captured the step-wise dynamics of the real data made a lot of sense. It felt more constrained and potentially more stable than a vanilla GAN.

It was definitely more complex to implement than my initial attempt. The TimeGAN paper has several components: an embedding network, a recovery network (forming an autoencoder), a sequence generator, and a sequence discriminator. Managing these four networks and their respective losses was daunting.

### Building the Beast: Generator and Discriminator in PyTorch

For the TimeGAN-inspired model, I had to define these components in PyTorch. Here’s a very simplified skeleton of what my generator (the "supervisor" part of TimeGAN is what generates in latent space then mapped to data space by decoder) might have looked like conceptually. My actual code became a lot messier with more parameters and helper functions.

```python
import torch
import torch.nn as nn

class LatentGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LatentGenerator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        # self.input_dim = Z_dim for random noise
        # self.output_dim = hidden_dim for latent space sequences

    def forward(self, x, h_prev=None):
        # x is random noise (batch_size, seq_len, Z_dim)
        out, h_next = self.rnn(x, h_prev)
        # We want to output a sequence in the latent space
        out_sequence = self.linear(out) # (batch_size, seq_len, hidden_dim)
        return out_sequence, h_next

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        # input_dim is the feature_dim of the data (or latent_dim if discriminating there)
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1) # Output a single scalar (real/fake)

    def forward(self, x):
        # x is a sequence (batch_size, seq_len, feature_dim)
        rnn_out, _ = self.rnn(x)
        # We only care about the classification of the entire sequence,
        # so often people take the last hidden state of the RNN.
        # Or sometimes pass the whole sequence through a further linear layer.
        # For simplicity here, let's imagine using the last output.
        last_step_out = rnn_out[:, -1, :]
        out = self.linear(last_step_out) # (batch_size, 1)
        return torch.sigmoid(out) # Sigmoid for probability
```
The actual TimeGAN implementation involves more than this, particularly the autoencoder parts (embedder and recovery networks) and how they all connect. The generator above is more like the sequence generator in the latent space. The discriminator would then operate on these latent sequences or the recovered (data space) sequences.

My sequence length (`seq_len`) was typically around 24-30 steps, representing a short period of market activity. The `num_features` depended on what I derived from LOBSTER – things like normalized mid-price changes, bid-ask spread, and order book imbalance.

### The Agony and Ecstasy of Training

Training GANs is an art. The TimeGAN paper has three loss components:
1.  **Reconstruction Loss:** For the autoencoder part (embedding and recovery networks). This was usually a simple L2 loss.
2.  **Unsupervised Loss:** The standard adversarial loss for the sequence generator and discriminator. I used Binary Cross-Entropy (BCE).
3.  **Supervised Loss:** This was a tricky one. It tries to make the generator learn the step-wise dynamics of the original data in the latent space. It's an L2 loss between the generator's output (next latent vector) and the true next latent vector from the autoencoded real data.

Getting these losses to balance was a nightmare. For a while, my discriminator would "win" too easily. Its loss would go to near zero, and the generator's loss would skyrocket. The generated data looked nothing like financial data. I spent ages tweaking learning rates for the different components. I found a few forum posts suggesting using different learning rates for the generator and discriminator – often a slightly lower one for the generator. That helped a bit.

I also experimented with the number of training steps for the generator versus the discriminator per batch. Sometimes training G more frequently than D (or vice-versa) can help stabilize things. I remember one week where the generated data just looked flat – no volatility at all. It turned out my normalization of the input LOBSTER data was crushing all the variance. Refactoring that preprocessing step took a couple of days but made a huge difference.

One breakthrough came when I really focused on the latent space. Plotting t-SNE visualizations of the latent representations from the autoencoder (for real data) versus the latent sequences from the generator helped me see if the generator was even beginning to capture the right manifold. Initially, they were completely separate blobs.

The computational cost was also a factor. Training these models on even a subset of LOBSTER data took hours, sometimes days, on my relatively modest student setup (an older gaming PC with a decent GPU, but not a research cluster). Each time I changed a hyperparameter, it meant another long wait. This forced me to be very methodical in how I experimented.

### Evaluation: Chasing that 95% Parity

This was the moment of truth. The TimeGAN paper suggests a few qualitative (visualizations) and quantitative metrics. For quantitative evaluation, I focused on comparing statistical properties of the generated data versus the real LOBSTER data (a held-out test set, of course).

1.  **Distributional Statistics:** I looked at the distributions of log-returns, tick-by-tick volatility, and trade volumes. I calculated mean, variance, skewness, and kurtosis for these and compared them. I used the Kolmogorov-Smirnov (KS) two-sample test (`scipy.stats.ks_2samp`) to see if the distributions were statistically distinguishable. Getting the p-values high enough (indicating we can't reject the null hypothesis that they are from the same distribution) was a key target.
2.  **Autocorrelation:** Financial time series often exhibit autocorrelation in squared returns (volatility clustering) but little in raw returns. I plotted the autocorrelation functions (ACF) for both real and synthetic data and visually compared them.
3.  **Cross-Correlations:** I also checked cross-correlations between different generated features (e.g., does spread correlate with volume imbalance in a similar way to real data?).

Initially, the parity was terrible – maybe 50-60% on my made-up "feature parity score" which was a weighted average of how many statistical tests passed or how close the moments were. The >95% claim came after many iterations. For example, I found my initial model wasn't capturing volatility clustering well. I tweaked the depth and hidden units of the GRU layers in the generator and discriminator, and played with the `seq_len`. Longer sequences seemed to help the model learn these temporal dependencies better, but also increased training time.

I also implemented a few more domain-specific metrics like the frequency of specific order book events if I were generating raw LOBSTER-like messages (though my project focused on derived features). The "Discriminative Score" (from the TimeGAN paper) was also useful – training a separate post-hoc classifier (e.g., a 2-layer LSTM) to distinguish between real and synthetic sequences. A score close to 0.5 indicates the synthetic data is highly realistic.

It wasn't one magic fix, but a slow, iterative process of:
*   Train model.
*   Generate samples.
*   Run statistical tests.
*   Identify biggest discrepancies.
*   Hypothesize a model change (e.g., "maybe the latent space is too small," or "the learning rate for G is too high").
*   Implement change.
*   Repeat.

The ">95% statistical feature parity" specifically refers to a checklist of these statistical properties. For instance, for features like the mean and variance of log returns, if the synthetic data's value was within a tight band (e.g., +/- 0.05 standard deviations) of the real data's value, it "passed" for that feature. The KS-test p-values had to be above a certain threshold (e.g., 0.05) for distributional checks. The 95% was the proportion of these checks that the final model passed.

### Key Challenges & "Aha!" Moments

*   **Mode Collapse:** Early on, my generator would just output the same flat line or a very repetitive pattern. This was incredibly frustrating. Wasserstein GAN with Gradient Penalty (WGAN-GP) is often suggested for this, but integrating that into the TimeGAN structure felt like another huge project. I managed to mitigate it somewhat with careful learning rate scheduling, instance noise (adding small noise to inputs of the discriminator), and ensuring the autoencoder part was well-trained first to provide a good latent representation.
*   **Hyperparameter Tuning:** The sheer number of hyperparameters (learning rates for 3-4 networks, dimensions of latent spaces, sequence length, batch size, relative weights of the different losses) was overwhelming. I ended up using a very rudimentary form of grid search for some, and for others, it was more intuition built from reading papers and lots of failed experiments. I found a particular StackOverflow thread discussing stabilizing RNN-based GANs that gave me a few ideas about gradient clipping values, which seemed to help prevent exploding gradients in the recurrent layers.
*   **The "Supervised Loss" in TimeGAN:** Understanding exactly what this loss was doing took me a while. It’s essentially forcing the generator to predict the *next latent encoding* of the real sequence, given the current latent encoding. This direct guidance on the temporal dynamics seems crucial for time series and was a bit of an "aha!" moment when it clicked. My initial implementations of this were buggy, and fixing that significantly improved the temporal realism of the output.
*   **Computational Limits:** As a student, I don't have access to a massive GPU cluster. I had to be smart about batch sizes and sequence lengths to avoid running out of memory. This sometimes meant I couldn't explore architectures or settings that might have worked better but were too computationally expensive. This constraint definitely shaped my design choices towards what was feasible.

### Reflections and What's Next

This project was a deep dive, and honestly, there were times I thought I'd never get sensible output. Understanding the nuances of GANs, especially for sequential data, is an ongoing process. The LOBSTER data itself is so rich; I feel like I've only scratched the surface of the features one could try to model.

If I were to do it again, I’d probably spend even more time on the data preprocessing and feature engineering upfront. I also think exploring attention mechanisms within the generator could be a promising direction for capturing longer-range dependencies in financial data more effectively.

For now, I have a generator that can produce pretty decent synthetic tick data. The next step is to actually *use* it – perhaps for backtesting some simple strategies or as an input for other market simulation projects. It’s been a long road, but seeing those statistical plots line up for the synthetic and real data was a genuinely exciting moment.