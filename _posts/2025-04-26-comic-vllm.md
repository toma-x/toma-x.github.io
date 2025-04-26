---
layout: post
title: Generative VLLM for Comic Narration
---

Hey everyone,

So, after my adventures with things like stock prediction algorithms and VQA agents, I wanted to try something a bit more creative but still technically challenging. I'm a big fan of comics, and I got this idea: could I use AI to generate narrative descriptions for comic book panels? Like, automatically create the little text boxes that explain what's happening, but have the AI understand the image *and* the context from previous panels? This led me down the path of fine-tuning a Vision Language Large Model (VLLM) and then trying to improve its storytelling using Reinforcement Learning from Human Feedback (RLHF), but with a twist – using CLIP scores instead of actual human feedback.

The plan was ambitious: fine-tune a ViT-GPT2 model on comic panels and then use RLHF to make the generated descriptions more coherent.

## Choosing the Model: ViT-GPT2

First off, I needed a model that could handle both images (the comic panels) and text (the descriptions). I landed on using a ViT-GPT2 architecture. Why? Honestly, availability and existing examples played a big role. There were some implementations and papers connecting Vision Transformers (ViT) for image understanding with GPT-2 for text generation. GPT-2 seemed like a reasonable choice for generation – powerful enough to potentially write coherent text, but maybe not as massive and resource-hungry as some newer models, which I hoped would make fine-tuning slightly more feasible on the GPUs I had access to (mostly Google Colab).

The basic setup involves feeding the comic panel image through ViT to get image embeddings, projecting those embeddings into the same space as GPT-2's word embeddings, and then feeding both the image representation and potentially some starting text (like the previous panel's description) into GPT-2 to generate the next description. I found some pre-trained checkpoints or setups online that connected ViT and GPT-2, which gave me a starting point.

## The Nightmare: Building the Comic Dataset

This was probably the hardest, most underestimated part. You can't just download a "comic panel narration" dataset. I had to build my own. I spent *weeks* scraping panels from copyright-free or old, public domain comics online. Then, for each panel, I had to:
1.  Extract the image.
2.  Write a short description myself, trying to capture what was happening visually and link it to the previous panel's description (if applicable).

This was incredibly tedious. My "custom dataset" ended up being quite small – maybe a few thousand panel-description pairs, across different comic styles. I knew this wasn't ideal; large models need large datasets, but it was the best I could manage as a solo student project. The quality was also probably inconsistent because I was writing the descriptions myself. This data limitation definitely impacted the whole project.

## Fine-tuning: Getting Something Basic Working

With my small dataset ready, I started fine-tuning the ViT-GPT2 model. I used PyTorch and Hugging Face's `transformers` library. The goal was to teach the pre-trained model to generate descriptions relevant to the input panel image.

```python
# Conceptual training loop structure (simplified)
# Assume 'model' is the ViT-GPT2 multimodal model
# Assume 'dataset' yields {'image': panel_image, 'description': text}
# Assume 'processor' handles image preprocessing and tokenization

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5) # Example learning rate

# Freezing parts - crucial for memory!
# Freeze ViT completely? Freeze most of GPT-2? Experimented a lot here.
# Often ended up only training the projection layer and maybe top layers of GPT-2.
# for param in model.vision_encoder.parameters(): param.requires_grad = False 
# ... etc ...

model.train()
for batch in dataloader: # Assume dataloader prepares batches
    images = batch['image'].to(device)
    texts = batch['description'] # Target descriptions

    # Prepare inputs (image processing, tokenization, combining features)
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Prepare labels for GPT-2 (usually involves shifting input_ids)
    labels = inputs['input_ids'].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100 # Ignore padding in loss
    # Might need more careful label creation depending on model structure

    optimizer.zero_grad()
    
    # Forward pass
    # The exact input format depends heavily on the specific ViT-GPT2 implementation
    outputs = model(**inputs, labels=labels) 
                                    
    loss = outputs.loss
    loss.backward()
    # Gradient clipping often needed for stability
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Log loss, maybe calculate BLEU on validation set periodically
```

**Challenges:**
*   **GPU Memory:** Even GPT-2 isn't tiny, and combined with ViT, it constantly pushed the limits of Colab GPUs. I had to use tiny batch sizes (like 2 or 4), gradient accumulation, and aggressively freeze model layers.
*   **Slow Training:** Because of the small batches and complexity, training took ages. Iterating on hyperparameters was painful.
*   **Initial Results:** The first fine-tuned model generated *okay* descriptions. They were usually relevant to the image content, but often generic ("A man is standing there.", "Two people are talking."). Crucially, they often lacked narrative coherence – the description for panel 3 wouldn't necessarily follow logically from panel 2.

## The Coherence Problem & Trying RLHF with CLIP

The lack of story flow was the main issue. The model treated each panel almost independently. I'd read about RLHF being used to align LLMs with human preferences (like making them more helpful or harmless). I thought, maybe I could use RLHF to align the model with the preference of "narrative coherence"?

But training a reward model based on human feedback was way too complex. I needed a *proxy* reward signal. My idea: use **CLIP scores**. CLIP is great at measuring image-text similarity. My hypothesis was: a good, coherent description for Panel N should not only be relevant to Panel N's image (high CLIP score with Image N) but maybe also somewhat related to the *previous* description (Panel N-1's text).

So, the plan became: use RL (specifically, I tried using Proximal Policy Optimization - PPO, because libraries like `trl` support it) to further tune the ViT-GPT2 model, rewarding sequences that had high CLIP similarity to their corresponding image.

## Implementing CLIP Reward and RLHF

This part got messy. I used the `trl` library from Hugging Face, which helps set up PPO training for language models.

1.  **Policy Model:** My fine-tuned ViT-GPT2 model acted as the policy network.
2.  **Reference Model:** I kept a copy of the initial fine-tuned model as a reference to calculate KL divergence (a common PPO technique to prevent the policy from changing too drastically).
3.  **Reward Calculation:** This was the core custom part. After the policy model generated a description for a panel image during RL training:
    *   I loaded a pre-trained CLIP model (`openai/clip-vit-base-patch32` usually).
    *   Calculated the cosine similarity between the generated description's text embedding and the panel image's embedding using CLIP.
    *   This similarity score (scaled appropriately) became the reward signal fed into the PPO algorithm.

```python
# Conceptual Reward Calculation (within the RL loop)
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model (only need to do this once)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

def calculate_clip_reward(generated_texts, images):
    rewards = []
    with torch.no_grad():
        # Process images and generated texts for CLIP
        image_inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
        text_inputs = clip_processor(text=generated_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Get embeddings
        image_features = clip_model.get_image_features(**image_inputs)
        text_features = clip_model.get_text_features(**text_inputs)
        
        # Normalize features (important for cosine similarity)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity (batch-wise dot product)
        # Scale by logit_scale? CLIP often uses a learned temperature scale. Let's use raw similarity here.
        # similarity = torch.diag(text_features @ image_features.T) # If batch sizes match
        # More robustly if calculating per item:
        similarities = (text_features * image_features).sum(dim=1) # Dot product if normalized
        
        # Convert similarity score to reward (e.g., scale it)
        # Raw similarity is between -1 and 1. Maybe scale to be positive?
        # reward_tensor = (similarities + 1.0) / 2.0 # Scale to 0-1 range
        reward_tensor = similarities # Keep raw similarity, maybe PPO handles scaling? Needs tuning.

    return reward_tensor # Return tensor of rewards for the batch
```

Using `trl` involved setting up the `PPOConfig`, `PPOTrainer`, loading the models, and running a loop where the policy generates text, the reward is calculated, and `ppo_trainer.step` updates the policy model.

**RLHF Challenges:**
*   **Instability:** PPO training was incredibly finicky. The loss would often explode, or the model would start generating gibberish. Tuning the PPO hyperparameters (learning rate, KL coefficient, clipping values) was mostly guesswork and took many failed runs.
*   **CLIP Reward Limitations:** While CLIP measures image-text relevance, it doesn't directly measure narrative flow or coherence. Rewarding high CLIP scores sometimes led to descriptions that were *very* literal descriptions of the image but still didn't connect well to the previous panel. I tried incorporating similarity to the previous description into the reward, but that added more complexity and tuning knobs.
*   **Slow Feedback Loop:** RL training was even slower than fine-tuning because it involves generation and reward calculation steps.

## Results: BLEU +4 pts? Sort Of...

After *a lot* of painful tuning, I did manage to get the RLHF process to run somewhat stably and saw an improvement. I measured the performance using BLEU score against the descriptions I had originally written for my custom dataset. The RLHF-tuned model achieved a **BLEU score about 4 points higher** than the model that was only fine-tuned.

What did this mean in practice? The generated descriptions *felt* slightly better. They seemed to stick closer to the specific objects and actions in the panel (likely due to the CLIP reward reinforcing image relevance). There was maybe a marginal improvement in coherence, but it wasn't a magic bullet. The descriptions still sometimes felt disjointed. The +4 BLEU points sounded good, but BLEU isn't great for measuring creativity or narrative flow, so I took it with a grain of salt. It mostly showed the model got better at matching keywords from my reference descriptions, partly thanks to CLIP focusing it on the image content.

## What I Learned (The Hard Way)

*   **Data is Everything:** My small, self-created dataset was a major limitation. A larger, more consistently annotated dataset would likely have made a huge difference for both fine-tuning and evaluating coherence.
*   **RLHF is Hard:** Implementing RLHF, even with a proxy reward like CLIP, is way harder than standard supervised fine-tuning. It's unstable, sensitive to hyperparameters, and the reward design is critical and non-trivial.
*   **CLIP isn't a Coherence Oracle:** Using CLIP as a reward for relevance is smart, but it doesn't inherently understand narrative. It helped ground the descriptions in the image, but didn't solve the story flow problem entirely.
*   **Multimodal Debugging:** Debugging issues when you have both vision and language components, plus an RL layer, is complex. Is the problem in the image features, the text generation, the reward calculation, or the RL update? Often hard to tell.

## What I'd Do Differently

*   **Start Simpler:** Maybe focus *only* on generating a single panel description first, making sure the ViT-GPT2 connection works perfectly before adding context or RL.
*   **Better Data Strategy:** Invest way more time in dataset creation or find better existing resources, even if smaller but higher quality.
*   **Alternative Reward:** Explore other reward signals besides just CLIP image-text similarity. Maybe incorporate text-based coherence metrics (like sentence embedding similarity to the previous description) more carefully into the reward.
*   **Simpler RL?** Maybe try simpler RL algorithms before jumping straight to PPO, or even explore non-RL techniques for controllable generation.

## Conclusion

This project was a rollercoaster. Getting a VLLM to generate descriptions for comic panels was cool, and seeing *some* improvement from the RLHF/CLIP approach felt like a small victory. The final BLEU score increase was encouraging on paper. However, the process was fraught with challenges – data limitations, unstable RL training, and the imperfections of using CLIP as a reward for narrative coherence. It definitely wasn't a complete success in terms of creating a perfect comic narrator, but I learned an incredible amount about multimodal models, the practical difficulties of fine-tuning and RLHF, and the importance of data. It was frustrating at times, but pushing the boundaries and trying to connect these different AI techniques was a super valuable experience.

Happy to chat if anyone has thoughts or has tried similar crazy projects!