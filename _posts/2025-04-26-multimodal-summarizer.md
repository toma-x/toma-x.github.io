---
layout: post
title: Building a Multimodal Document Summarizer with VLLM Concepts and Fine-Tuned CLIP
---

Hey everyone,

It's been a little while since my last post diving into LeetCode problems. I've been heads-down working on a project that I'm pretty excited about and wanted to share some details about it. I decided to tackle the challenge of summarizing documents that contain not just text, but also images. Think reports with charts, articles with diagrams, or webpages with figures – how can we get a concise summary that captures information from *both* modalities?

## The Challenge: Beyond Text-Only Summarization

Most summarization tools work great on plain text, but they completely ignore images. Often, these images contain crucial information (like trends in a graph or details in a diagram) that's lost in a text-only summary. I wanted to build something that could look at both the words and the pictures to create a more complete summary. This field is often called multimodal learning, and specifically, I was looking at multimodal summarization.

## My Approach: A Vision Language Large Model (VLLM)

I figured this would require a model that understands both vision and language – a VLLM. Now, I didn't build a massive foundation model from scratch (I wish!). Instead, I focused on taking concepts from recent powerful architectures, like the ideas seen in things like Llama 3.2, and adapting them for this specific multimodal task. The core idea was to use a pre-trained language model for text understanding and combine it with a vision model that could process the images.

## Architecture Breakdown

Here’s a rough idea of the components I put together:

1.  **Language Model Backbone:** I used a pre-trained transformer-based language model as the foundation. The goal wasn't to reinvent the wheel for text understanding, but to leverage existing strong language capabilities. I looked at how models like Llama 3.2 structure their layers and attention mechanisms as inspiration for how the text part should work.
2.  **Vision Encoder (CLIP):** For understanding the images, I turned to CLIP (Contrastive Language–Image Pre-training). CLIP is really cool because it's trained on a massive dataset of images and their corresponding text descriptions. This means it already has a pretty good "understanding" of what's in an image and how to represent it in a way that might align with text. I didn't just use CLIP off-the-shelf though; I decided to **fine-tune** the CLIP vision encoder specifically on the types of images found in my documents. My hypothesis was that this would help it focus on the relevant visual details needed for summarization.
3.  **Connecting Text and Vision (Adapter Modules):** This was the tricky part: how do you get the language model to "see" the image features from CLIP? Retraining the entire LLM is computationally expensive (and way beyond my resources!). So, I explored using **adapter modules**. These are small, lightweight neural network layers that you can insert into the pre-trained LLM. The idea is to freeze most of the large language model's parameters and only train these adapters (along with fine-tuning CLIP). The adapters learn how to inject the visual information from CLIP into the LLM's processing flow, hopefully allowing it to consider both text and image context when generating the summary.

Here's a very simplified conceptual diagram:

```
[ Document Text ] --> [ Language Model Backbone ] --+
                                                    |
                                                    V [ Adapter ] --> [ Summary Generation ]
[ Document Image ] --> [ Fine-tuned CLIP Encoder ] --+                 (using LLM)
```

## Data, Data, Data

Standard summarization datasets are text-only. For this project, I needed documents with both text and related images, plus a "gold standard" summary that reflected information from both. Finding such a dataset was tough. I ended up having to curate a **custom dataset**. This involved gathering documents (like academic paper snippets, reports, and informative web articles) that had text sections clearly related to embedded images/charts. Then came the hard part: writing summaries myself that explicitly drew information from both the text and the images. This was time-consuming but crucial – the model needs good examples to learn from.

## The Fine-Tuning Journey

Training involved feeding the model pairs of (document text + document image) and trying to get it to generate a summary close to the one I wrote.

Key parts of the process:

*   **Freezing:** Most of the LLM backbone parameters were frozen to save computation and retain its language capabilities.
*   **Training Targets:** The main components being trained were the adapter modules and the CLIP vision encoder.
*   **Loss Function:** Standard sequence-to-sequence loss (like cross-entropy) to compare the generated summary with the target summary.

Here’s a pseudo-code snippet illustrating the conceptual training step:

```python
# Conceptual Training Step (Simplified)

def training_step(batch, model, optimizer):
    texts = batch['text_input']
    images = batch['image_input']
    target_summaries = batch['target_summary']

    # Get image features from fine-tuned CLIP
    image_features = model.clip_encoder(images) 
    
    # Generate summary using LLM backbone + adapters + image features
    # (This is the complex part involving the adapter fusion)
    generated_summaries = model.generate(texts, image_features) 
                                      
    # Calculate loss
    loss = calculate_loss(generated_summaries, target_summaries)
    
    # Backpropagate and update weights (only for adapters and CLIP)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

```

**Challenges:**

*   **Alignment:** Getting the text and image information to properly "fuse" via the adapters was tricky. Sometimes the model would lean too heavily on text, ignoring the image, or vice-versa. Tuning the adapter architecture and learning rates was key.
*   **Computational Resources:** Even with adapters, training models like this takes significant GPU memory and time. I had to use techniques like gradient accumulation and mixed-precision training.
*   **Data Quality:** Ensuring the custom dataset summaries accurately reflected both modalities was an ongoing effort.

## Evaluation and Results

How do you measure success? I used the **ROUGE** metric, which is standard for summarization tasks. Specifically, I focused on **ROUGE-L** (which measures the longest common subsequence, capturing sentence-level structure similarity).

After a lot of tuning and experimentation, I managed to achieve a **ROUGE-L score consistently above 0.4** on my test set.

Is ROUGE-L > 0.4 good? For extractive text summarization, scores can be higher. But for abstractive summarization (where the model generates new sentences), and especially for a complex multimodal task with a custom (and likely relatively small) dataset, getting above 0.4 felt like a solid achievement. It indicated the model was definitely capturing relevant information from both sources, although there's room for improvement.

**Example (Conceptual):**

Imagine a document snippet:
*Text:* "Figure 3 shows a significant increase in user engagement following the Q2 product launch. The daily active users metric climbed steadily throughout July."*
*Image (Figure 3):* A line graph showing daily active users sharply increasing after a point labeled "Q2 Launch" in late June/early July.*

*Generated Summary:* "User engagement, measured by daily active users, significantly increased after the Q2 launch, as shown in Figure 3's graph depicting a steady climb in July." *(Notice how it references both the text's statement and confirms it with the graph's visual trend)*.

## What I Learned

This project was a massive learning experience:

*   **Power of Pre-training & Fine-tuning:** Leveraging giants like CLIP and LLM backbones is incredibly powerful. Fine-tuning CLIP and using adapters felt much more manageable than training from scratch.
*   **Adapters are Cool:** I was impressed by how effective adapter modules can be for injecting new information or capabilities into large models without full retraining. They seem like a very promising direction for efficient model adaptation.
*   **Multimodal is Hard:** Combining information from different modalities smoothly is non-trivial. The alignment and fusion steps are critical and challenging.
*   **Data is King (Still):** Building the custom dataset was maybe the most laborious part, but also one of the most critical for success.

## Future Ideas

There's definitely more that could be done:

*   Explore different adapter architectures or fusion methods.
*   Expand the custom dataset significantly.
*   Experiment with different vision encoders or LLM backbones.
*   Evaluate with human judgment alongside ROUGE scores.

## Conclusion

Building this multimodal summarizer was challenging but incredibly rewarding. It pushed my understanding of vision-language models, fine-tuning strategies, and the practical difficulties of working with multimodal data. Adapting concepts from architectures like Llama 3.2 and fine-tuning components like CLIP proved to be a viable path, and achieving a ROUGE-L score over 0.4 on my custom dataset feels like a good step forward. It's not perfect, but it's a working VLLM summarizer that considers both text and images!

Happy to discuss this more if anyone has questions or similar experiences!
