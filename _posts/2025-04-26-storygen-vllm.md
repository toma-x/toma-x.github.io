---
layout: post
title: Controllable Multimodal Story Generation
---
```

Hey everyone,

So, after spending time on LeetCode, poking at VLLM fairness, and building that VQA agent, I kind of got hooked on these multimodal models. The VQA project, where the model answered questions about an image using ViT and Llama 2, got me thinking: could I push it further? Instead of just answering a simple question, could I get a model to tell a *story* based on an image? And even better, could I have some *control* over that story, like setting a mood or style? That sounded like a cool challenge, so I decided to dive in.

My project idea was: fine-tune a Vision Large Language Model (VLLM) to generate stories from images, and then try to implement some techniques, inspired by Reinforcement Learning from Human Feedback (RLHF), to control the output. Spoiler alert: it was definitely a journey, with lots of "aha!" moments mixed with plenty of "why isn't this working?!" headaches.

**Getting Started: The VLLM and Fine-tuning**

First things first, I needed a VLLM. Based on my VQA project experience, I decided to stick with a similar architecture conceptually: combining a pre-trained vision encoder (like ViT) with a pre-trained language model (like a Llama variant or maybe something smaller like OPT or GPT-2 if resources were tight, just to get started). The key is having a model that can process both the image features and text prompts. I used Hugging Face's `transformers` library again because it makes loading these pre-trained parts way easier.

The core task was sequence generation: input an image, maybe a starting prompt like "Tell a story about this image:", and output a narrative. This meant fine-tuning was essential. The base models are usually trained on tasks like image captioning (short descriptions) or generic text generation, not multi-paragraph storytelling tied to an image.

This led to the first major hurdle: **data**. Standard datasets like COCO provide images and short captions (e.g., "a dog catching a frisbee"). I needed image-story pairs. Finding a good dataset for this was *tough*. There are some niche ones like VIST (Visual Storytelling dataset), but they can still be limited or tricky to work with. I ended up trying to use a combination: some standard caption data (like COCO) mixed with whatever story data I could find (like VIST), hoping the model would learn to generate longer, more narrative text while staying grounded in the image. The reality was that the data was mismatched – fine-tuning on short captions often didn't encourage the model to suddenly write long stories. I suspect this data limitation was a big factor in the final quality.

The fine-tuning setup looked something like this (simplified PyTorch-style pseudocode, similar to my VQA setup):

```python
# Assume 'vllm_model' includes vision encoder, projection, and LLM
# Assume 'tokenizer' and 'image_processor' are loaded
# Assume 'dataset' yields {'image': img_tensor, 'story_text': text}

optimizer = torch.optim.AdamW(vllm_model.parameters(), lr=5e-5) # Tune learning rate carefully!

# Freeze parts of the model to save memory/time?
# e.g., for param in vllm_model.vision_encoder.parameters(): param.requires_grad = False

for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['image'].to(device)
        stories = batch['story_text'] # List of strings

        # Prepare inputs for the VLLM
        # This involves getting image embeddings, tokenizing text, combining them
        inputs = prepare_multimodal_inputs(images, stories, tokenizer, image_processor, device)
        
        # Prepare labels (target token IDs, shifted for causal LM)
        tokenized_stories = tokenizer(stories, return_tensors="pt", padding="longest", truncation=True).to(device)
        labels = tokenized_stories.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100 # Ignore padding in loss calculation

        optimizer.zero_grad()
        
        # Forward pass - inputs might be dict with 'pixel_values', 'input_ids', 'attention_mask', etc.
        # Or maybe pass 'inputs_embeds' if combining manually
        outputs = vllm_model(**inputs, labels=labels) 
                                    
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}") 
```

Getting this loop to run without crashing due to GPU memory errors was a constant battle. VLLMs are *massive*. I had to use small batch sizes, gradient accumulation, mixed precision (`torch.cuda.amp`), and aggressively freeze parts of the model (like the vision encoder and most of the LLM layers, only training the projection layer and maybe the final LLM layers). Even on a Colab Pro GPU, training was slow and tedious.

**Trying to Add Control: The RLHF-Inspired Bit**

Okay, so after some fine-tuning, the model might generate *something* resembling a story. But often, it was generic, repetitive, or lost coherence quickly. I wanted control – to guide it towards a certain style ("make it adventurous") or just make it stick to the plot better.

Full RLHF involves training a separate reward model on human preferences and then using reinforcement learning (like PPO) to fine-tune the generator. That was way too complex for me. Inspired by my fairness probe project where I tried modifying generation at inference time, I decided to try something similar but simpler here:

1.  **Control via Prompting:** The easiest thing was just adding instructions to the prompt, e.g., "Tell a funny story about this image:" or "Write a suspenseful narrative based on this picture:". This sometimes worked okay-ish for style, but the model often ignored the instruction or the effect was very superficial. It didn't help much with coherence.

2.  **Inference-Time Reward/Penalty:** I tried to build a simple function that would score the generated sequence *as it was being generated* (e.g., during beam search). The idea was to penalize bad sequences or reward good ones.
    *   For **coherence**, I tried simple things like penalizing excessive repetition of words or maybe using a basic sentence embedding model to check if consecutive sentences were drifting too far apart topic-wise (very crude!).
    *   For **style**, I tried rewarding the presence of certain keywords associated with a style (e.g., "adventure", "mystery", "laugh" for funny).

Here’s a *very conceptual* idea of how I tried to hack this into beam search:

```python
# --- Conceptual sketch of influencing beam search ---

def calculate_story_quality_bonus(sequence_ids, target_style="adventurous"):
    """ VERY simple function to reward/penalize sequence """
    text = tokenizer.decode(sequence_ids, skip_special_tokens=True)
    bonus = 0.0
    
    # Style example: reward keywords
    if target_style == "adventurous":
        if "explore" in text or "danger" in text or "treasure" in text:
            bonus += 0.5 
            
    # Coherence example: penalize repetition (very basic)
    words = text.split()
    if len(words) > 10 and len(set(words[-10:])) < 5: # If last 10 words have few unique words
         bonus -= 1.0

    return bonus

# --- Inside hypothetical beam search logic ---
# When scoring potential next sequences:
# original_score = model_probability(sequence)
# quality_bonus = calculate_story_quality_bonus(sequence, current_style_goal)
# final_score = original_score + quality_bonus # Adjust score

# Beam search then prioritizes beams with higher final_scores.
# --- End Conceptual Sketch --- 
```

**Did it work?** Eh... kinda, sometimes, not really reliably. Modifying the generation process directly is tricky. The simple reward functions were often too naive. Rewarding keywords could lead to stories just stuffing those words in unnaturally. Penalizing repetition sometimes made the text stilted. Getting coherence right with simple checks was almost impossible. It felt less like "control" and more like crudely bumping the generation process in a certain direction, often with weird side effects. It definitely didn't feel like the sophisticated control shown in RLHF papers.

**More Challenges and Faceplants**

*   **Story Coherence:** This was the biggest failure. The model struggled to maintain a consistent narrative thread. Stories would start okay but then meander, contradict themselves, or just stop making sense. The connection to the image also sometimes weakened as the story went on.
*   **Evaluation Hell:** How do you even measure if a story is "good"? Or if the "control" worked? Metrics like ROUGE or BLEU are useless for creativity and coherence. I ended up just reading the outputs myself, which is super subjective and slow. I didn't have a good quantitative way to track progress.
*   **Resource Drain:** I can't emphasize enough how much GPU memory and time this took. Fine-tuning even for short periods was computationally expensive. Forget about extensive hyperparameter searches.

**What I Actually Learned**

Despite the struggles, I learned a ton:

*   **Multimodal Complexity:** Generating coherent, long-form content that stays grounded in an image is *way harder* than simple VQA or captioning. The interaction between vision and language needs to be much deeper.
*   **Data is Crucial:** The lack of large-scale, high-quality image-to-story datasets felt like a major bottleneck. Garbage in, garbage out (or maybe just slightly less garbage out).
*   **Control is Hard:** Simple hacks for controlling generation (like my inference-time penalties) are brittle. Real control likely needs more sophisticated methods like proper RLHF or maybe controllable generation techniques designed specifically for LLMs.
*   **Appreciation for Engineering:** Keeping the training loop running, managing memory, debugging weird PyTorch errors – it really builds character (and debugging skills!).

**Final Thoughts & What Next**

So, did I end up with a system that consistently generates amazing, controllable stories from images? Honestly, no. The results were often interesting, sometimes surprisingly creative, but also frequently incoherent or uncontrolled. The fine-tuning helped push the model towards longer text, but the "storytelling" aspect was weak, and the RLHF-inspired control felt more like a gimmick than a robust feature in my implementation.

If I were to do it again, I'd focus almost entirely on **finding or creating better data** first. Maybe start with a much simpler goal, like generating a coherent paragraph instead of a whole story. I might also explore techniques other than inference-time hacks for control, perhaps looking into prompt-tuning or adapter-based approaches specifically for style.

It was a frustrating but fascinating project. It gave me a real appreciation for how complex creative generation is for AI and how far the tools still have to go, especially when working with limited resources. It definitely showed me the gap between reading about cool techniques like RLHF and actually getting them to work effectively on your own. Still, pushing these models and seeing what they *can* do, even imperfectly, is pretty exciting.

Happy to chat if anyone has struggled with similar things or has better ideas!