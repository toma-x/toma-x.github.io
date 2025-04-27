---
layout: post
title: Vision Encoder Analysis \& Optimization
---

### Project Reflection: Vision Encoder Analysis & Optimization

Alright, documenting this project feels like closing a significant loop. This whole "Vision Encoder Analysis & Optimization" thing started less as a structured project and more as a deep dive born out of frustration with some baseline multimodal model performance I was seeing. The goal solidified into trying to understand how different vision backbones impacted downstream tasks in VLM contexts and then seeing if I could actually *improve* things, specifically targeting performance boosts through data-centric approaches.

My setup for this was pretty standard for me: my trusty old ThinkPad P53, running Ubuntu 20.04, mostly living inside VS Code. The first hurdle, predictable as ever, was environment management. I needed PyTorch, torchvision, Hugging Face transformers (for some model loading and pipeline components), and a few data handling libraries like pandas and numpy. Getting CUDA drivers, PyTorch with CUDA support, and everything else playing nicely took... let's just say more than one evening. I remember getting a persistent `CUDA error: initialization error` that turned out to be a conflict between the NVIDIA driver version and the CUDA toolkit version I had installed via `conda`. Reinstalling the CUDA toolkit globally *after* making sure the correct driver was active finally fixed it. Spent probably six hours just on that one, mostly sifting through NVIDIA's complex documentation and various forum posts.

The core idea was benchmarking vision encoders. I focused on ViT variants (specifically `vit-base-patch16-224`) and a couple of ResNet flavors (`resnet50`, `resnet101`) since they are common starting points and represent different architectural paradigms (transformer vs. CNN). I wanted to see how their raw feature extraction performance varied and, more importantly, how that translated when hooked up to a simple VLM head. I wasn't building a VLM from scratch; the plan was to use pre-trained components and fine-tune or evaluate on a downstream task. The datasets were the tricky part – "large-scale multimodal" sounds nice on paper, but handling millions of image-text pairs on a laptop is... challenging. I ended up working with a subset of Conceptual Captions and a smaller, more specialized internal dataset we had access to. Loading and preprocessing this volume of data was a major bottleneck. My initial data loader was a disaster, reading images one by one with PIL, which was excruciatingly slow. Switched to using `torchvision.datasets.ImageFolder` (adapting it slightly for multimodal pairs) and implementing proper multiprocessing with `num_workers > 0` in the PyTorch `DataLoader`. That sped things up dramatically, but figuring out the right number of workers to not overload my CPU while keeping my GPU fed was another mini-project in itself.

Benchmarking involved training a simple linear layer on top of the frozen vision encoder outputs for a specific classification or regression task derived from the multimodal data. This gave a rough idea of the quality of features. The OOM errors were constant companions during this phase, especially with the ViT models which have higher memory footprints. I had to aggressively reduce batch sizes, sometimes down to 8 or even 4, which, while necessary, made training times even longer. This was a constant trade-off: bigger batch size for faster training vs. smaller batch size to avoid crashing.

Moving to the optimization phase, data filtering felt intuitive. The hypothesis was that noisy or irrelevant image-text pairs in large datasets dilute the training signal. My filtering approach involved two main stages:
1.  **Image quality/relevance:** Using a simple off-the-shelf image aesthetic score predictor and discarding images below a certain threshold.
2.  **Text relevance/alignment:** This was harder. I experimented with CLIP score (the cosine similarity between image and text embeddings from a frozen CLIP model) as a proxy for semantic alignment. The idea was to keep pairs where the image and text were strongly related according to CLIP's understanding.

Implementing the text relevance filter was where I hit a nasty bug. My initial code for calculating CLIP scores looked something like this:

```python
# Initial attempt - buggy!
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

filtered_data = []
for img_path, text in raw_data:
    try:
        image = Image.open(img_path).convert("RGB")
        # Process image and text - this part was okay
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        # Calculate score
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate cosine similarity - THIS WAS WRONG
        # I mistakenly calculated similarity between raw logits or something
        # instead of normalized embeddings
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        
        # BUG: This doesn't calculate semantic similarity correctly
        score = torch.cosine_similarity(logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0))

        if score > clip_threshold:
            filtered_data.append((img_path, text))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue # Skip bad samples

# ... rest of the filtering logic
```

I spent a day wondering why the scores seemed nonsensical and filtering wasn't helping. At 2 AM, fueled by too much coffee, I re-read the CLIP paper and documentation more carefully and realized `logits_per_image` and `logits_per_text` are the *similarity scores* themselves after temperature scaling, not the raw embeddings. The correct way to use the *embeddings* for custom similarity (or other downstream tasks) is via `outputs.image_embeds` and `outputs.text_embeds`, which are already normalized.

Corrected code snippet for score calculation:

```python
# Corrected CLIP score calculation using embeddings
# ... (model and processor initialization same) ...

with torch.no_grad():
    outputs = model(**inputs)

# Get normalized embeddings
image_embeddings = outputs.image_embeds
text_embeddings = outputs.text_embeds

# Calculate cosine similarity correctly between normalized embeddings
score = torch.cosine_similarity(image_embeddings, text_embeddings)

# ... rest of filtering logic ...
```

This correction was a breakthrough. Applying filtering based on meaningful CLIP scores (along with the image quality filter) immediately showed promise.

Next was data augmentation. Standard image augmentations (`RandomResizedCrop`, `ColorJitter`, `RandomHorizontalFlip`) from `torchvision.transforms` were easy to integrate. The challenge was *multimodal* augmentation – techniques that modify both the image *and* text consistently. I didn't go too deep here due to time constraints, but I experimented with simple text augmentations like random word deletion or synonym replacement using the `nlpaug` library. However, ensuring the text augmentation didn't break the image-text correspondence was tricky. I mostly stuck to image augmentations as they seemed less likely to introduce noise relative to the potential benefit, given my limited time. My reasoning for sticking to simpler augmentations was primarily practical: complex multimodal augmentations require careful tuning and validation, which I didn't have time for, and standard image augmentations are a proven baseline.

After applying both the filtering and augmentation strategies to the training data subset, I retrained the simple VLM head on top of the best performing vision encoder from the initial benchmarking phase (which turned out to be the ViT base model for the classification task I focused on). The results were encouraging. Evaluating on a held-out test set (carefully ensuring no filtered data leaked into the test set!), I saw an improvement in the primary evaluation metric (an F1 score variant for the classification task) from around 0.75 to 0.86. This represented roughly a 15% relative improvement over the baseline trained on the unfiltered, non-augmented data.

This project was a solid reminder that while model architecture is important, the data you feed into it is often the low-hanging fruit for performance gains, especially when you're working with noisy, large-scale web data. Debugging the data pipeline and filtering logic took significantly more time than experimenting with models, but the payoff was clear. There's still a lot more that could be done – exploring more advanced multimodal augmentations, more sophisticated filtering based on vision-language models themselves, or training the VLM head end-to-end rather than just a linear layer. But for now, hitting that 15% improvement felt like a hard-earned win after many hours staring at logs and debugging data pipelines.