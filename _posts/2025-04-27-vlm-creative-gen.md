---
layout: post
title: Multimodal VLM for Creative Generation
---

Okay, so this past semester I've been chipping away at a personal project – building a multimodal Vision-Language Model (VLM) focused on generating image descriptions, but specifically trying to steer those descriptions towards particular creative styles. It started because I was tired of standard, factual captions and wanted something that felt... different. More evocative, maybe.

The core idea was to train a VLM that could take an image and a text prompt describing a *style* (like "noir detective story" or "surrealist painting") and output a caption reflecting both the image content and that requested style.

My first thought was to use something off-the-shelf, maybe fine-tune a massive model. But I wanted to understand the pieces, the architecture, the training loops myself. So, I decided to build something more manageable, focusing on the alignment part.

The stack I ended up with was primarily **PyTorch** for the deep learning framework and Hugging Face's **`transformers`** library, which honestly saved me a ton of time on the base components like image encoders and text decoders.

Getting started, the data was a big hurdle. I needed image-text pairs, but also some way to evaluate if a generated text fit a *style*. I ended up curating a dataset myself from various sources, pairing images with initial, factual descriptions. This was just standard crawler stuff, nothing fancy. The real challenge was figuring out how to represent the "style." Initially, I just thought I could append style prompts like `"[STYLE: noir]"` to the input, but that felt too simplistic and didn't really *teach* the model the style, just conditioned it on a token.

For the base VLM, I used a simple setup: a pre-trained vision transformer (like a ViT) to get image embeddings and a pre-trained text decoder (like a GPT-2 slice) to generate text, with a simple cross-attention mechanism connecting them. Training this base model on the image-text pairs was fairly standard – compute the cross-entropy loss on the generated text tokens given the image embedding and the previous tokens.

```python
# Initial VLM training loop sketch
# This was on my old Dell XPS 15, took forever per epoch
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss() # For text generation

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        images, input_ids, attention_mask = batch
        # Move to GPU if available (my ancient 1050 Ti)
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        # Forward pass - model outputs logits for next token
        # This part was tricky - feeding images and text tokens correctly
        # Initially messed up masking, generated nonsense
        outputs = model(image=images, input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Calculate loss - need to shift tokens for language modeling
        # This line caused a shape mismatch error for hours the first night
        # "RuntimeError: The size of tensor a (128) must match the size of tensor b (129) at non-singleton dimension 1"
        # Realized I needed [:, :-1] for input and [:, 1:] for targets
        loss = criterion(logits[:, :-1].reshape(-1, vocab_size), input_ids[:, 1:].reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch} Loss: {total_loss / len(dataloader)}")

```

The real beast was the **RLHF-inspired alignment**. I wasn't building a full-blown reinforcement learning setup with human feedback loops – that was way beyond my resources and time. Instead, I borrowed the core idea: use a *reward signal* to guide the generation towards desirable properties (fitting the style prompt) rather than just maximizing likelihood of the training text.

I needed a way to quantify how well a generated caption fit a style. My proxy for this was another smaller model, a classifier, trained separately to score text based on its style. For instance, given a caption and the "noir" style prompt, the classifier would output a score indicating how "noir" that caption felt. This classifier was trained on examples of texts labeled with styles. This was a quick and dirty approach, I know, but it was feasible on my laptop.

The VLM's generation process was then framed sort of like a policy. Instead of just sampling the next token based on the highest probability from the cross-entropy trained model, I wanted to nudge it. I implemented a form of policy gradient, specifically REINFORCE. The "reward" for a generated sequence was the score from my style classifier, plus a term to penalize deviation too much from the base VLM's probabilities (to prevent generating complete nonsense).

This RL part was brutal. Implementing the REINFORCE update correctly took probably three days, mostly debugging gradient calculations and ensuring the reward signal was propagating back through the text generation steps. I kept hitting issues with `requires_grad` and detaching tensors.

```python
# Sketch of the RL-inspired training step
# This was way more complex in reality, especially reward calculation
# and handling batched generation. This snippet is simplified.

# Assume 'model' is the VLM, 'style_critic' is the style classifier
# 'image' is the input image tensor, 'style_prompt_id' is the target style ID

model.train()
optimizer.zero_grad()

# Generate a caption using sampling (e.g., greedy or beam search initially)
# This is the 'action' in RL terms
generated_sequence, log_probs = model.generate_with_log_probs(image, style_prompt_id) # Hypothetical function

# Calculate reward for the generated sequence
# Reward combines style score and maybe a fluency/diversity term
# Getting style_score from a separate model was a pain - needed to handle its forward pass
style_score = style_critic(generated_sequence, style_prompt_id)

# Add penalty for deviating from base model probabilities (PLIC - Policy Loss with Information Constraint?)
# This was based on reading bits of the original RLHF paper and some blog posts
base_log_probs = base_vlm_model.get_log_probs(image, generated_sequence) # Need log probs from vanilla model
kl_penalty = calculate_kl_divergence(log_probs, base_log_probs)

reward = style_score - alpha * kl_penalty # alpha is a hyperparameter

# Calculate loss using REINFORCE
# The loss is - log_probs * reward
# This assumes reward is constant for the sequence, a simplification of real RLHF
# This line failed repeatedly with "RuntimeError: gradient computed with respect to a non-leaf tensor"
# tracing it back found issues in how log_probs were tracked during sampling
loss = -(log_probs * reward).mean() # Maximize reward -> Minimize negative reward

loss.backward()
optimizer.step()

```

Debugging the RL training was an exercise in patience. Gradient values would be `NaN` constantly. The generated text would become completely nonsensical after a few batches. I spent hours staring at `loss.backward()` calls and the computation graph in VS Code's debugger (when it worked). At one point, around 2 AM on a Friday, I realized I was calculating the KL penalty incorrectly, not using detached probabilities for the baseline model comparison, which was messing up the gradients for the main VLM. Another time, I spent an entire Saturday morning wrestling with CUDA out-of-memory errors because I wasn't clearing the cache or properly managing tensor lifetimes within the sampling loop. `torch.cuda.empty_cache()` became my best friend.

I considered using proximal policy optimization (PPO) instead of REINFORCE, which is more common in complex RL tasks and generally more stable. But implementing the clipping and multiple epochs per batch felt too complicated given the time constraints and the fact that I was already struggling with basic gradients. Sticking to REINFORCE, despite its higher variance, seemed more achievable. Plus, my "reward" was a relatively simple signal from a separate model, not a complex environment interaction.

The results are... mixed, but promising. The model *does* seem to pick up on the style cues, sometimes producing genuinely creative descriptions. For an image of a rainy street, with the "noir" prompt, it might generate "The city wept, asphalt tears reflecting the neon wounds of a forgotten night," instead of just "A street at night with rain." Other times, it completely fails, generating repetitive phrases or ignoring the style prompt entirely. The style classifier isn't perfect, and neither is my simplified RL setup.

It's definitely not a production-ready system, but as a learning experience in combining multimodal models with alignment techniques beyond simple supervised learning, it was incredibly valuable. The challenges with data, model architecture choices, and especially the finicky nature of gradient-based optimization in an RL context were significant and taught me a lot more than just running `trainer.train()` on a pre-packaged task.

Next steps, if I revisit this, would be exploring better ways to represent style, maybe trying PPO, and definitely using a more robust style evaluation mechanism than my simple classifier. But for now, I'm happy I got the core idea working, even imperfectly, on my little setup.