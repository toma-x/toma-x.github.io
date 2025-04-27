---
layout: post
title: RLHF for Visual Instruction Following
---

Reflections on Implementing a Simplified RLHF Loop for Visual Instruction Following

Over the past few months, I've been chipping away at a personal project to better understand Reinforcement Learning from Human Feedback (RLHF) and its application beyond purely text-based models. The goal was ambitious for my setup: adapt a simplified RLHF loop to align the outputs of a multimodal model for visual instruction following. Specifically, I wanted the model's text response to a visual input and instruction to better match human preferences, moving away from just factual correctness towards helpfulness or relevance in a more subjective sense.

The core idea was straightforward enough in theory: take a pre-trained multimodal model, use human preference data to train a reward model, and then use that reward model to fine-tune the multimodal model using an RL algorithm. My personal constraints, primarily compute on my late-2019 MacBook Pro (yeah, not ideal, I know) and time, meant everything had to be drastically simplified.

First hurdle: selecting the multimodal model. I couldn't realistically train something like a full Flamingo or Llava from scratch. The strategy was to use a smaller, readily available checkpoint that could fit into memory for fine-tuning. I settled on experimenting with a version of Llava, specifically a 7B parameter variant. Getting this loaded and running inference locally was the first mini-project in itself. The `transformers` and `accelerate` libraries from Hugging Face were indispensable here, though navigating the exact versions required to avoid `bitsandbytes` errors on my specific OS and Python version took the better part of an afternoon. I remember hitting a persistent `CUDA error: initialization error` for ages before realizing it was a simple version mismatch between `torch`, the NVIDIA driver on the remote machine I sometimes borrowed access to, and the `bitsandbytes` build. Locally, it was dealing with quantization just to get the model to *load* without instantly OOMing.

Next, the dataset. Full-scale RLHF datasets are massive. My approach was to curate a *tiny* dataset. I spent a few days collecting pairs of (image, text instruction). For each pair, I generated a few different potential text responses from the initial Llava model. Then, I manually ranked these responses based on my own preference. Was the response helpful given the instruction and image? Did it follow the instruction correctly? Was it concise? This yielded about 150 preference tuples (e.g., (prompt, response_A, response_B), where response_A is preferred over response_B). This is minuscule, I know, but the goal was proof of concept, not state-of-the-art performance. Storing this in a simple JSONlines format felt most manageable:

```jsonl
{"image_path": "path/to/img1.jpg", "instruction": "Describe the main object.", "response_a": "A cat is sitting on a mat.", "response_b": "There is an animal.", "preference": "response_a"}
{"image_path": "path/to/img2.png", "instruction": "What color is the sky?", "response_a": "The sky is blue.", "response_b": "It's daytime.", "preference": "response_a"}
...
```
This structure allowed me to easily load and process the data for training the reward model.

The Reward Model (RM) was the next piece. The standard approach is often a separate model trained on preference data to output a scalar score. Given my limited data and compute, training a large model for this felt inefficient. I decided on a simpler approach: use the existing Llava model's embedding space. I planned to train a small, simple regression head on top of the multimodal model's representation of the (image, instruction, response) triplet. The idea was that this small head would learn to predict a preference score based on the multimodal features. This turned out to be overly complex and unstable with my tiny dataset.

After much frustration, I pivoted. Instead of training a separate RM from scratch or grafting onto Llava, I used a pre-trained text-only reward model (like one of the helpfulness-focused OpenAssistant models available on HF) and adapted it. The challenge was incorporating the visual information. My crude solution was to generate a detailed text description of the image using a separate captioning model *first*, and then concatenate this description with the instruction and response before feeding it to the text-only RM. This felt like a hack, losing potential rich multimodal features, but it was a pragmatic decision driven by the need to make *any* progress.

```python
# Simplified RM input preparation (initial attempt)
def prepare_rm_input(image_path, instruction, response):
    # This part was the hacky bit - generate caption first
    image_caption = generate_caption(image_path) # Using a separate model/script
    input_text = f"Image description: {image_caption}\nInstruction: {instruction}\nResponse: {response}"
    return input_text

# ... inside RM training loop ...
# Load preference pair
text_a = prepare_rm_input(data['image_path'], data['instruction'], data['response_a'])
text_b = prepare_rm_input(data['image_path'], data['instruction'], data['response_b'])

# Feed through text-only RM
score_a = rm_model(text_a)
score_b = rm_model(text_b)

# Train RM to maximize score_a - score_b
loss = -torch.log(torch.sigmoid(score_a - score_b))
# Backpropagate... This loss function took a bit to get right,
# initially used simple MSE which didn't make sense for ranking.
```
This approach worked... sort of. Training the text-only RM on my generated text inputs and preference pairs yielded a model that had *some* correlation with my preferences, but it was noisy. Debugging the training loop, ensuring the loss was correctly implemented for pairwise ranking, and getting the data pipeline right took a solid two days. I remember staring at monotonically increasing loss curves around 2 AM one night before spotting a simple issue with gradient accumulation setup.

Finally, the RL part. This is where `trl` library from Hugging Face became essential. Trying to implement PPO from scratch on top of a multimodal model using my custom RM felt like an impossible task within my timeframe. `trl` provides PPO trainers designed to work with `transformers` models. The PPO setup involved:
1.  The "Policy" model: My Llava model, adapted to output probabilities over the vocabulary given (image, instruction).
2.  The "Reference" model: A frozen copy of the initial Llava model to calculate KL divergence penalty.
3.  The Reward function: Using my trained (and somewhat dodgy) text-only RM to score generated responses.

The process was iterative. For a given (image, instruction) input:
-   Generate responses using the current Policy model (using sampling strategies like beam search or sampling with temperature).
-   Score the generated responses using the RM.
-   Calculate the KL divergence between the Policy model's probabilities and the Reference model's probabilities for the generated response tokens.
-   Use the RM score and KL penalty to construct the RL reward signal.
-   Train the Policy model using PPO on this reward signal.

This was the hardest part. The `trl` PPO trainer abstraction is helpful, but configuring it correctly for a multimodal input where the "prompt" involves both image and text, and the output is text, required digging into their examples and source code. Getting the input format right for the trainer (`{'input_ids': ..., 'attention_mask': ..., 'pixel_values': ...}`) and ensuring the rewards were correctly aligned with the generated sequences was a major headache.

Initial RL training runs were disastrous. The model would either produce garbage, repeat tokens endlessly, or its output quality would collapse entirely within a few steps. This pointed to issues with the reward signal being noisy, the KL penalty not being strong enough (or too strong), or unstable PPO hyperparameters. I spent another few days just tweaking `learning_rate`, `ppo_epochs`, `mini_batch_size`, and the `gamma`/`lam` parameters for the PPO algorithm. The `clip_range` also seemed critical; values too high led to instability, too low prevented learning. Monitoring the `ppo/returns/mean` and `ppo/policy/approxkl` logs from `trl`'s logging helped diagnose some issues, but it often felt like shooting in the dark due to the small dataset size amplifying any instability.

One specific error I wrestled with for hours was related to padding and tensor shapes when preparing batches for the PPO updates. Because the generated responses had varying lengths, padding was necessary, but getting the attention masks and reward signals correctly aligned with the padded sequences, especially across devices if I tried using a GPU, was tricky. Locally on my CPU, it was just slow, but on a GPU, mismatching shapes led to immediate, cryptic CUDA errors.

```python
# Snippet from my PPO training loop adaptation
# This is roughly how I prepared inputs for the trl PPO trainer
# Had to ensure 'input_ids' included both prompt and generated response,
# and attention masks matched.
# The 'rewards' tensor needed careful alignment with the non-padded tokens.

def prepare_ppo_batch(image_tensor, instruction_input_ids, instruction_attention_mask, generated_response_ids):
    # Combine instruction and generated response IDs
    # Need padding here! trl expects batch of sequences
    # This part was surprisingly bug-prone due to variable lengths
    full_sequence_ids = torch.cat([instruction_input_ids, generated_response_ids], dim=-1)
    # ... apply padding ...
    padded_sequence_ids, padded_attention_mask = pad_sequences(full_sequence_ids, ...) # Implemented padding logic

    # Need to create a dummy batch for the multimodal model input
    # Llava takes pixel_values and input_ids
    # This was a bit awkward to fit into trl's expected format initially
    model_input = {
        "input_ids": padded_sequence_ids,
        "attention_mask": padded_attention_mask,
        "pixel_values": image_tensor.repeat(padded_sequence_ids.size(0), 1, 1, 1), # Broadcast image tensor for the batch
    }

    # Prepare reward tensor - needs to be the same shape as generated_response_ids,
    # with zeros for padding and prompt tokens. This alignment caused many bugs.
    rewards = calculate_rewards(model_input, generated_response_ids, rm_model) # This function was complex
    # ... ensure rewards are correctly shaped and aligned with generated tokens ...

    return model_input, rewards
```

Despite the challenges and the small dataset, after much trial and error with hyperparameters and debugging data pipelines, the RL training showed some signs of life. The average reward metric tracked by the `trl` trainer started increasing, and qualitatively, the model's responses for the specific instructions in my tiny dataset seemed slightly better aligned with my preferences than the base model outputs. For instance, if my preference was for a more concise description, the fine-tuned model was slightly more likely to produce one. This was a modest result, largely limited by the dataset scale and my compute, but seeing any positive movement in the metrics after days of debugging felt like a significant breakthrough around 3 AM one particular morning.

Reflecting on the project, the primary lessons were about the practical difficulties of implementing these complex pipelines even in a simplified setting. Data curation, aligning different model components (RM, policy, reference model), and debugging the RL training loop itself were far more time-consuming than initially anticipated. Libraries like `transformers` and `trl` are powerful, but getting them to work together for a non-standard use case like this, particularly with multimodal inputs and limited resources, required deep dives into documentation and source code, alongside frequent visits to GitHub issues and StackOverflow. It's a testament to the complexity of aligning large models, even when you're just trying to make a small dent with limited tools.
