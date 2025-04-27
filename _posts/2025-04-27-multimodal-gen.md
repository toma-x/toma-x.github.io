---
layout: post
title: Multimodal Story Generation AI
---

Alright, so after a few months of chipping away at it, the multimodal story generation project is finally at a point where it feels... *finished*. At least the first iteration is. The core idea was pretty straightforward: could you feed an AI an image and have it write a coherent story *based* on that image? Not just image captioning, but using the visual context as a launchpad for narrative.

The initial thought process was, okay, large language models are good at generating text, right? But they only see text. Images are a whole different ballgame. So, the fundamental challenge was getting the image information *into* the language model in a way it could understand and use.

My approach settled on fine-tuning a pre-trained language model. Starting from scratch felt completely out of scope given the time I had. I landed on using a smaller GPT-2 model from the `transformers` library. The reasoning was partly practical – I needed something that could realistically train on my machine, a Lenovo Legion with a decent RTX 3060, without taking weeks. The sheer size of larger models was a non-starter.

The main technical hurdle, the one that ate up most of my evenings and weekends, was integrating the image data. The LLM takes sequences of token embeddings. An image is... pixels. How do you bridge that gap? This is where the custom vision encoder came in. I didn't want to just use a pre-trained vision model like ResNet or CLIP and slap a linear layer on it. While practical, building my own encoder felt like a critical learning step. Plus, I wanted control over the feature extraction process, even if it was a basic CNN.

So, the vision encoder is a relatively simple convolutional neural network built with PyTorch's `nn` module. It has a few convolutional layers, batch normalization, ReLU activations, and max pooling, ending with a flattening step and a final linear layer.

```python
import torch
import torch.nn as nn

class SimpleVisionEncoder(nn.Module):
    def __init__(self, output_dim=768): # output_dim should match LLM embedding size
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # Input is RGB image
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces spatial dimensions

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Need to calculate the output size after pooling to know input dim for linear layer
            # This was a pain point initially, had to manually calculate or use a dummy tensor
        )

        # Let's assume image size is 256x256 for calculation example
        # After 3 max pools (stride 2), spatial dims become 256/8 = 32x32
        # Number of channels is 128
        # Linear layer input: 128 * 32 * 32
        self.fc = nn.Linear(128 * 32 * 32, output_dim) # Project to LLM embedding dimension

    def forward(self, x):
        x = self.conv_layers(x)
        # Flatten the tensor for the linear layer
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch size
        x = self.fc(x)
        return x
```
Getting the input dimensions right for the linear layer (`128 * 32 * 32` in this example, assuming 256x256 input) was a classic PyTorch headache. My first try just had some arbitrary number there, which obviously blew up with a `RuntimeError: size mismatch` during the first forward pass. Spent a good hour just tracing the tensor shape through the conv and pool layers, adding `print(x.shape)` after each step, until I could manually calculate the flattened size. There are fancier ways using dummy tensors, but at 1 AM, just printing shapes felt like the most direct approach.

The real puzzle was getting the vision features into the LLM. The LLM expects a sequence of token embeddings. The vision encoder outputs a single vector (or a sequence, depending on design, but I went with a single vector summary initially). The simplest way seemed to be adding this visual embedding *before* the text sequence embeddings.

So, the input sequence to the LLM would be `[visual_embedding, text_token_embedding_1, text_token_embedding_2, ...]`. The `output_dim` of the vision encoder's final linear layer had to match the embedding dimension of the GPT-2 model I was using (768 in my case).

This is where I hit a wall for about two days. I was concatenating the tensors like this: `torch.cat([vision_output.unsqueeze(1), text_embeddings], dim=1)`. The `unsqueeze(1)` was to make the vision embedding a sequence of length 1, matching the `(batch_size, sequence_length, embedding_dim)` shape of the text embeddings. However, the LLM (specifically, the `forward` method in the `transformers` library) expects `input_ids` for its initial input to handle embeddings and attention masks correctly. Passing raw embeddings directly required bypassing the standard input handling, which felt hacky and led to issues with attention masks.

I spent ages reading through the `transformers` source code on GitHub and documentation pages, trying to figure out the clean way to inject custom embeddings. Most examples were for modifying intermediate layers, not the initial input. A crucial StackOverflow thread pointed out that you could pass `inputs_embeds` directly to the model's `forward` method *instead* of `input_ids`, provided you also managed the `attention_mask` appropriately. The attention mask needed an extra `1` added at the beginning to account for the visual token.

```python
# Inside the training loop, preparing inputs
# Assuming 'vision_features' is output from SimpleVisionEncoder (batch_size, embedding_dim)
# Assuming 'input_ids' is tokenized text input (batch_size, seq_len)
# Assuming 'attention_mask' is the standard mask for input_ids (batch_size, seq_len)

# Get text embeddings from the model's embedding layer
# model is the GPT-2 model
text_embeddings = model.transformer.wte(input_ids) # shape (batch_size, seq_len, embedding_dim)

# Add sequence dimension to vision features to match text embeddings
vision_features_seq = vision_features.unsqueeze(1) # shape (batch_size, 1, embedding_dim)

# Concatenate vision features and text embeddings along the sequence dimension (dim=1)
multimodal_input_embeds = torch.cat([vision_features_seq, text_embeddings], dim=1) # shape (batch_size, seq_len + 1, embedding_dim)

# Create a new attention mask: prepend a 1 for the visual token
visual_attention_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
multimodal_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1) # shape (batch_size, seq_len + 1)

# Now pass these directly to the model's forward method
outputs = model(
    inputs_embeds=multimodal_input_embeds,
    attention_mask=multimodal_attention_mask,
    labels=labels # Assuming labels are shifted inside the model or handled elsewhere
)
# Note: Labels also need to be shifted or aligned correctly depending on the LLM implementation.
# The visual token doesn't have a corresponding label token in the text sequence.
# The loss calculation typically ignores the first position if using standard language modeling loss.
```
This little `unsqueeze(1)` and `torch.cat` with the attention mask adjustment took a solid day and a half to properly debug and understand why it wasn't working initially (tensor shape mismatches were the main culprit, again). It required diving into how `inputs_embeds` is handled internally by the `transformers` model.

Training itself was an exercise in patience. The dataset wasn't huge – maybe 50k image-story pairs scraped from various creative writing sites (with permission where possible, mostly focusing on public domain or explicitly open licenses). Training on my laptop's GPU meant small batch sizes (4 or 8 depending on sequence length) and watching the loss slowly tick down over hours. I tried larger batch sizes, but quickly ran into CUDA out of memory errors, typical for training large models on consumer hardware. Learning rate tuning was another cycle of frustration; too high and the loss diverged, too low and it barely moved. AdamW with a small learning rate (around 1e-5) and a few warm-up steps seemed to work best, found through pure trial and error over several training runs that had to be stopped early.

Evaluating the output is tricky. How do you objectively score a story? I ended up doing a lot of manual inspection. Some generations were completely off-topic, ignoring the image entirely. Others picked up on elements but produced nonsensical sentences. The best ones managed to weave details from the image (like a cat on a fence, or a stormy sky) into a simple, coherent narrative.

For example, feeding it an image of an old, empty swing set in a park often yielded stories about loneliness, childhood memories, or abandonment. This felt like a success – the model was picking up on the *mood* or potential themes suggested by the visual. Feeding it a picture of a bustling market might generate text about crowds, smells, or transactions.

There were definitely cases where it generated repetitive text or just hallucinated things not present in the image or even the general context. That's expected with these models, especially with limited fine-tuning data and computational resources.

Reflecting on it, the biggest takeaway was the sheer complexity of handling multimodal data correctly at the model input level. It wasn't just training a model; it was fundamentally rethinking how different data types need to be structured and presented to a neural network designed for a single modality. Debugging tensor shapes and understanding library-specific input handling (`inputs_embeds` vs `input_ids`) were the most time-consuming parts. If I were to do it again, I'd probably spend more time initially prototyping the data pipeline and model input structure in isolation before integrating it with the full training loop. Also, finding or curating a larger, cleaner multimodal dataset would be crucial for better performance. But for now, having a working prototype that can take an image and produce *something* resembling a relevant story feels like a significant step.