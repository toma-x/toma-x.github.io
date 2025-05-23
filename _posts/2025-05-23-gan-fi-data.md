---
layout: post
title: GANs for Synthetic Fixed Income Scenario Generation
---

## GANs for Synthetic Fixed Income Data: A Deep Dive into My Latest Project

This semester has been a whirlwind, juggling coursework with a personal project that's taken up more late nights than I'd care to admit: building Generative Adversarial Networks (GANs) to create synthetic fixed income scenarios. Specifically, I’ve been wrestling with Wasserstein GANs with Gradient Penalty (WGAN-GP) in PyTorch, trying to coax them into generating realistic interest rate swap rate scenarios. The main goal? To see if these generated scenarios could be useful for augmenting datasets, especially for more robust backtesting and stress-testing of derivative models – something I’ve been increasingly interested in.

### The Starting Point: Why GANs and Why Swap Rates?

I first stumbled upon GANs in a machine learning course, and the idea of networks battling it out to create new data seemed fascinating. Most examples were about images, but I wondered about financial data. Fixed income, particularly interest rate swaps, seemed like a challenging but rewarding area. Swap curves aren't just random numbers; they have complex interdependencies and dynamics (parallel shifts, twists, etc.). If a GAN could learn these, it would be pretty powerful. The existing historical data is what it is; being able to generate plausible new scenarios could really help in understanding model behavior under conditions we haven't exactly seen before.

My initial research led me through a maze of GAN architectures. Vanilla GANs seemed too unstable for this kind of structured data. I read a few papers and a bunch of blog posts, and the consensus for more stable training pointed towards Wasserstein GANs. The WGAN-GP variant, with its gradient penalty, appeared to be the state-of-the-art for avoiding mode collapse and ensuring smoother training, so I decided to bite the bullet and aim for that.

Data acquisition was the first real hurdle. I managed to get my hands on a decent dataset of historical USD interest rate swap rates for various tenors (1Y, 2Y, 5Y, 10Y, 30Y) spanning several years. Cleaning it was a whole sub-project in itself – dealing with missing entries (had to use some interpolation for those), ensuring consistency, and then figuring out how to normalize it. I settled on min-max normalization for each tenor across the dataset to scale rates into a [0, 1] range, which I later realized I should probably change to [-1, 1] to better suit the Tanh activation in the generator's output layer. That was an early facepalm moment.

### Wrestling with PyTorch: Building the WGAN-GP

I chose PyTorch because I'd used it a bit before and found it more intuitive than TensorFlow for research-y projects, especially with dynamic graphs. Setting up the environment wasn't too bad, though I did have a brief scare making sure CUDA was playing nice with PyTorch for GPU acceleration – a must for training GANs in a reasonable timeframe.

My Generator and Critic networks are pretty standard fully connected architectures. I didn't go overly complex because I was worried about overfitting with the amount of data I had and, frankly, my own ability to debug something enormous.

For the Generator:
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_dim=256):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh() # To output values between -1 and 1
        )
        self.output_dim = output_dim

    def forward(self, z):
        # z is the input noise vector
        output = self.net(z)
        return output
```
The `output_dim` would correspond to the number of swap rate tenors I was trying to generate, say 5 for the different points on the curve. The `noise_dim` I set to 100, which seemed like a common choice.

The Critic (or Discriminator in WGAN terminology) is sort of a mirror image:
```python
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1) # No sigmoid! Outputs a score
        )
        self.input_dim = input_dim

    def forward(self, S):
        # S is a batch of real or generated swap curve scenarios
        score = self.net(S)
        return score
```
Getting the gradient penalty calculation right was a major pain. The WGAN-GP paper is clear, but translating that into correct PyTorch code that handles batch operations and gradients took me a few attempts. I remember staring at the `.backward(retain_graph=True)` calls and the use of `torch.autograd.grad` for a long time. My initial implementation was slow and I suspected I wasn't calculating the gradients correctly with respect to the interpolated samples. A few StackOverflow threads about WGAN-GP implementations were my lifeline here.

The training loop itself is where I spent most of my "debugging" time. I set the critic to train more frequently than the generator (5 critic updates per generator update was a common recommendation I followed).
```python
# Inside the training loop (conceptual)
# for epoch in range(num_epochs):
#     for i, real_scenario_batch in enumerate(dataloader):
#         # Train Critic
#         critic_optimizer.zero_grad()
#         noise = torch.randn(batch_size, noise_dim, device=device)
#         fake_scenario_batch = generator(noise)
#
#         critic_real_output = critic(real_scenario_batch)
#         critic_fake_output = critic(fake_scenario_batch.detach()) # detach generator history
#
#         # Gradient Penalty Calculation
#         epsilon = torch.rand(batch_size, 1, device=device)
#         epsilon = epsilon.expand_as(real_scenario_batch) # expand to same shape as data
#         interpolated_scenarios = epsilon * real_scenario_batch + (1 - epsilon) * fake_scenario_batch
#         interpolated_scenarios.requires_grad_(True)
#
#         prob_interpolated = critic(interpolated_scenarios)
#         gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated_scenarios,
#                                          grad_outputs=torch.ones_like(prob_interpolated),
#                                          create_graph=True, retain_graph=True)
#         gradients = gradients.view(gradients.size(0), -1)
#         gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#
#         critic_loss = critic_fake_output.mean() - critic_real_output.mean() + gradient_penalty
#         critic_loss.backward()
#         critic_optimizer.step()
#
#         # Train Generator (less frequently)
#         if i % n_critic == 0:
#             generator_optimizer.zero_grad()
#             gen_noise = torch.randn(batch_size, noise_dim, device=device)
#             generated_output_for_gen = generator(gen_noise)
#             critic_eval_on_generated = critic(generated_output_for_gen)
#             generator_loss = -critic_eval_on_generated.mean() # maximize critic's confusion
#             generator_loss.backward()
#             generator_optimizer.step()
```
I used Adam for both optimizers. Learning rates were a constant source of anxiety. I started with something like `1e-4` for both, with betas `(0.5, 0.9)` which some sources suggested for GANs. There was a period where the critic loss would go to zero and stay there, meaning the generator wasn't learning anything useful. I must have tweaked learning rates and the `lambda_gp` (gradient penalty coefficient, I used 10) dozens of times. One specific week was brutal – the generator would sometimes produce okay-ish individual rates, but the *curve* made no sense, with long-term rates lower than short-term rates in ways that just don't happen often. This felt like a subtle form of mode collapse or the generator not grasping the relationships between tenors.

### Representing and Validating Swap Scenarios

A "scenario" in my case was a vector of 5 swap rates for the chosen tenors (1Y, 2Y, 5Y, 10Y, 30Y). My input data to the GAN was just batches of these vectors. The challenge was that these rates are not independent; they move together in patterns.
After training for many, many epochs (I let it run overnight multiple times, checking TensorBoard obsessively in the morning for the loss curves), I started to get outputs that, after un-normalizing, looked somewhat like swap rates.

Validation was, and still is, tricky. How do you *quantify* "realistic" for synthetic financial data?
1.  **Basic Statistics:** I first compared the mean and standard deviation of each generated tenor against the historical data. This was okay, but doesn't tell the whole story.
2.  **Correlation Matrix:** This was a big one. Swap rates are highly correlated. I plotted the correlation matrix of my real data and compared it to the correlation matrix of a large batch of generated scenarios. Initially, my generated rates were far too uncorrelated. This improved as training progressed and I tweaked hyperparameters, but getting it to closely match the real data's richness was tough. It got better when I played around with the network depth and hidden units, giving the model more capacity.
3.  **Visual Inspection:** Just plotting the generated curves. Do they look like plausible yield curves? Do they exhibit some of the typical shapes (upward sloping, flat, slightly inverted)? This was subjective but helpful.
4.  **Principal Component Analysis (PCA):** I ran PCA on both the real and synthetic datasets to see if the main modes of variation (often interpreted as level, slope, and curvature in interest rate models) were similar. The GAN did seem to pick up the dominant components, but the higher-order ones were less well-represented.

I wouldn't say the generated data is perfect by any means. Sometimes it still produces curves that are a bit "wiggly" or have relationships between tenors that are rare, but it's significantly better than random noise. The distributions of individual rates started to look quite similar to the historical ones after enough training.

### Exploring Deployment: gRPC for C# Integration

The final part of my project was to think about how this could actually be *used*. Many risk management systems in banks are built using C#/.NET. So, I wanted to explore how I could serve my PyTorch model to a C# application. gRPC seemed like the way to go – it’s efficient and designed for cross-language communication.

This was a completely new area for me.
1.  **Defining the `.proto` file:** I started by defining a simple service.
    ```protobuf
    syntax = "proto3";

    service ScenarioGenerator {
      rpc GenerateSwapScenario (ScenarioRequest) returns (SwapScenario);
    }

    message ScenarioRequest {
      int32 num_scenarios = 1; // How many scenarios to generate
      int32 seed = 2; // Optional seed for reproducibility
    }

    message SwapScenario {
      repeated double rates = 1; // The generated rates for different tenors
    }
    ```
    Even this took a couple of iterations. Should `SwapScenario` return a list of scenarios? For now, I kept it simple, one request gets one scenario (or a batch defined by some other parameter). I decided `GenerateSwapScenario` should return one `SwapScenario` which contains `repeated double rates` representing a single curve. If I wanted multiple, the client could call it multiple times or I could wrap it in another message.

2.  **Python gRPC Server:** I then built a Python server that loads my trained PyTorch generator model and implements the `GenerateSwapScenario` method. This involved taking the request, generating noise, feeding it to the model, un-normalizing the output, and then packaging it into the protobuf message. Passing PyTorch tensors to lists of doubles for the protobuf message felt a bit clunky but worked.

3.  **C# Client:** Then came the C# client. Setting up the gRPC tools for C# was straightforward enough with NuGet. Writing the client code to call the Python service was also surprisingly easy.
    ```csharp
    // Conceptual C# client snippet
    // var channel = GrpcChannel.ForAddress("http://localhost:50051");
    // var client = new ScenarioGenerator.ScenarioGeneratorClient(channel);
    // var request = new ScenarioRequest { NumScenarios = 1 }; // Simplified for this example
    // try
    // {
    //     var response = await client.GenerateSwapScenarioAsync(request);
    //     // Now response.Rates would have the generated swap rates
    //     // Log them, use them, etc.
    //     Console.WriteLine("Generated rates: " + string.Join(", ", response.Rates));
    // }
    // catch (RpcException e)
    // {
    //     Console.WriteLine("RPC failed: " + e.Status);
    // }
    ```
    The main challenge here was less the gRPC mechanics and more about managing the Python environment where the server was running, especially if it had GPU dependencies. For a student project, running it locally was fine, but for something more robust, I'd need to think about Docker or a more formal model serving solution. I spent a good afternoon debugging a `System.Net.Http.HttpRequestException` on the C# side which turned out to be my Python server not even starting up properly because of a path issue with the saved model file. Classic.

### Reflections and What's Next

This project has been an incredible learning experience. GANs are finicky beasts, and financial data adds its own layer of complexity. There were moments I was ready to give up, especially when staring at nonsensical output after hours of training. But seeing those first few plausible-looking swap curves come out of the generator was incredibly rewarding.

Key takeaways for me:
*   **Patience is key with GANs:** They don't just work on the first try. Or the tenth.
*   **Validation is crucial and hard:** Defining "good" synthetic data is non-trivial.
*   **Cross-disciplinary thinking:** Combining ML with financial concepts was challenging but super interesting.

If I had more time (famous last words of every student, I know):
*   **Conditional GANs (cGANs):** Generate scenarios conditional on some input, like the current state of the curve, or a macro variable. This would be much more powerful for stress-testing specific "what-if" scenarios.
*   **More Sophisticated Architectures:** Perhaps explore attention mechanisms or transformers if I were modeling sequences of curves over time.
*   **Rigorous Backtesting Integration:** Actually plug these scenarios into a toy derivative pricing model and see how the backtesting results compare when using augmented vs. only historical data.

For now, I'm pretty happy with how far this has come. It’s definitely solidified my interest in the intersection of machine learning and quantitative finance. And I've got a newfound respect for anyone who trains GANs for a living!