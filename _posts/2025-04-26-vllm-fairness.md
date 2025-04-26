---
layout: post
title: Trying to Poke the Bias Bear: Building Fairness Probes for VLLMs
---

Hey everyone,

Been a little while since my last LeetCode dive! I've been buried in a project that took me down a pretty different rabbit hole: exploring fairness in Vision Language Large Models (VLLMs). Specifically, I wanted to see if I could figure out a systematic way to check if these models, which generate text descriptions for images (image captioning), were showing biases based on gender or ethnicity, and then maybe even try to nudge them towards being fairer.

## The Worry: When Captions Go Wrong

We see these cool VLLMs generating captions for photos, and it seems almost magical. But what happens if the model learns associations from its training data that aren't exactly fair? For example, does it always assume a person in a kitchen is a woman? Does it associate certain jobs or activities primarily with one ethnic group? These aren't just hypothetical worries; biases in AI models can reflect and even amplify societal stereotypes, which is pretty concerning. I wanted to understand this better for image captioning models.

## My Plan: A "Fairness Probe" Framework

I decided to build a framework to "probe" a VLLM for potential biases. The basic idea was:
1.  Get a VLLM that does image captioning.
2.  Find or create a set of specific images designed to test for bias (the "probe set").
3.  Feed these images to the model and get its captions.
4.  Analyze the captions systematically to look for patterns suggesting gender or ethnic bias.
5.  (The ambitious part) Try using concepts from Reinforcement Learning from Human Feedback (RLHF) to maybe reduce the bias during caption generation *without* full retraining.

I decided to build this framework using **PyTorch** since I'm most comfortable with it and it works well with many pre-trained models.

## Step 1: Getting the Tools and Data

First, I needed a VLLM. Training one from scratch was obviously out of the question! I looked around for accessible pre-trained models that handle image captioning. Let's just say I used a readily available PyTorch implementation of a VLLM (like maybe a smaller variant or something based on BLIP architectures that I could run locally or on a modest cloud GPU).

Next, the **probe dataset**. This was actually one of the trickiest parts. Standard image datasets aren't necessarily designed to test for social bias. I needed images that could potentially trigger biased associations. I ended up trying a couple of things:
*   Looking at existing fairness benchmark datasets (like subsets of FACET or images from projects focusing on fairness in computer vision).
*   Manually searching for images depicting people in various roles (e.g., doctor, engineer, caregiver, athlete) trying to find examples across different perceived genders and ethnicities. This was *hard* and subjective. My goal wasn't a perfectly balanced dataset (that's a whole research project in itself!), but a set of images that could serve as a starting point for probing.

## Step 2: Building the Probe in PyTorch

The core probing loop involved loading the model and processing the images. Here’s a simplified conceptual PyTorch-like structure:

```python
import torch
from PIL import Image
# Assuming 'load_model' and 'preprocess_image' functions exist for the chosen VLLM
# Assuming 'generate_caption' is a method of the model

model, processor = load_model('my_vllm_checkpoint') # Load the VLLM and its processor
probe_image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ...] # My curated probe images

results = {}

model.eval() # Set model to evaluation mode
with torch.no_grad(): # No need to track gradients for inference
    for img_path in probe_image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt") # Preprocess image

            # Generate caption using the VLLM
            # This function might involve beam search, etc.
            outputs = model.generate(**inputs, max_length=50) 
            
            # Decode the generated token IDs into text
            caption = processor.decode(outputs, skip_special_tokens=True)
            
            results[img_path] = caption
            print(f"Image: {img_path}, Caption: {caption}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# --- Next step: Analyze the 'results' dictionary ---
```

## Step 3: Analyzing the Captions

Okay, so I got a bunch of captions. Now what? I needed to analyze them for bias. This wasn't super sophisticated, honestly. I started with simple things:
*   **Keyword Spotting:** Looking for gendered words ("man", "woman", "he", "she") or terms associated with specific roles ("nurse", "engineer", "CEO", "homemaker").
*   **Frequency Counts:** For images showing similar activities (e.g., people using computers, people cooking), how often did gendered terms appear? Did captions for people of certain apparent ethnicities consistently use different adjectives?
*   **Comparison:** If I had images designed to be ambiguous or counter-stereotypical (e.g., a man arranging flowers, a woman doing construction), what did the model say?

I wrote some basic Python scripts using regex and simple string matching to parse the `results` dictionary and tally these patterns.

**What I Found (Example Patterns):** It wasn't perfect, but I did see some trends. For instance, the model sometimes defaulted to gendered terms based on the activity rather than visual evidence (e.g., captioning "A woman is programming" even if the person's gender wasn't obvious in the image). I also noticed some role associations seemed skewed – "doctor" might be more often associated with male pronouns in the captions compared to "nurse". It wasn't universal, but the patterns suggested some level of learned bias.

## Step 4: Trying to Mitigate Bias with RLHF Ideas (During Inference)

This was the really experimental part. Full RLHF involves massive datasets of human preferences and retraining. I didn't have that. But I got inspired by the *idea* of RLHF – using rewards to guide generation. Could I apply a *penalty* during caption generation *at inference time* if the model started generating words I identified as potentially biased in certain contexts?

My idea:
1.  Create a simple "bias score" function. It checks the sequence being generated. If it sees a potentially problematic word (e.g., "woman" combined with "computer programmer" if I'm trying to counter that specific bias), it returns a penalty.
2.  Modify the VLLM's generation process (e.g., beam search or sampling) to factor in this penalty. The hope was to make paths leading to biased captions less likely.

Here's a *very conceptual* sketch of how you might try to influence beam search (in reality, this involves digging into the generation logic of the library, which can be complex):

```python
# --- Conceptual sketch of modifying generation ---

def calculate_bias_penalty(sequence_ids, vocab):
    """
    Checks a generated sequence (token IDs) for biased patterns.
    Returns a penalty score (e.g., a large negative number if bias found, 0 otherwise).
    This needs access to the model's vocabulary (vocab) to decode IDs.
    """
    text_sequence = processor.decode(sequence_ids) 
    penalty = 0.0
    # Simple example: Penalize if "woman" and "engineer" appear together
    if "woman" in text_sequence.lower() and "engineer" in text_sequence.lower():
        penalty = -10.0 # Assign a penalty
    # ... more complex rules could be added ...
    return penalty

# --- Inside the model's beam search logic (Hypothetical modification) ---
# During beam search, when calculating scores for candidate next tokens/sequences:

# original_score = model_probability_score(sequence) 
# bias_penalty = calculate_bias_penalty(sequence, processor.tokenizer.vocab)
# final_score = original_score + bias_penalty # Adjust score based on bias

# The beam search would then favor sequences with lower bias penalties.
# --- End Conceptual Sketch ---

```

**Challenges & Honesty:** Okay, this RLHF-at-inference idea was *really hard* to implement effectively.
*   Modifying the internals of `model.generate()` often requires deep library knowledge or rewriting parts of it.
*   Defining the `calculate_bias_penalty` function robustly is tough. Simple keyword matching is brittle. What's "biased" is highly context-dependent.
*   It sometimes worked a little – reducing the frequency of the most obvious targeted biased phrases.
*   But other times, it just made the captions weird, grammatically incorrect, or less descriptive overall because it was crudely steering away from certain words. It felt like playing whack-a-mole with words, and sometimes the "fix" was worse than the original caption.
*   It definitely didn't *solve* the bias problem. It was more like trying to put a band-aid on the output layer.

## What I Learned

This project was a huge learning curve!
*   **VLLMs are Complex:** Getting under the hood of these models, even just for inference, takes effort.
*   **PyTorch Power:** PyTorch was great for setting up the pipeline and handling the model/data loading.
*   **Bias is Subtle:** Identifying and measuring bias systematically is challenging. My simple analysis methods were just scratching the surface.
*   **Mitigation is Hard:** Trying to "fix" bias without full retraining or proper RLHF is tricky. My inference-time penalty idea was interesting to explore but had major limitations. It showed me why techniques like fine-tuning or proper RLHF are often necessary for deeper changes.
*   **Data Matters Most:** The difficulty in creating a good probe dataset highlighted how crucial data is for both training *and* evaluating these models fairly.

## Wrapping Up

So, I built a basic PyTorch framework to probe a VLLM for gender/ethnic biases in image captions and experimented with a simple inference-time mitigation technique inspired by RLHF. The probing part helped identify some potential issues, confirming that this is an area needing attention. The mitigation part was... well, an *experiment*. It showed how hard it is to steer these large models without more sophisticated methods.

It's definitely not a solved problem, but it was a fascinating project to work on. There's so much more that could be done – better datasets, more nuanced bias metrics, exploring adapter-based fine-tuning for mitigation, testing more VLLMs. For now, I have a much better appreciation for the complexities of fairness in AI!

Happy to chat more if anyone has thoughts or similar experiences!
