---
layout: post
title: Multimodal Vision-Language Reasoning
---

Continuing my exploration into multimodal AI, I recently wrapped up a project focused on enhancing Visual Question Answering (VQA) accuracy. The goal was pretty direct: build a system that could answer questions about images more effectively than standard unimodal or simpler fusion approaches, specifically by leveraging the strengths of large, pre-trained vision and language models.

My approach centered on integrating features from DINOv2, a strong self-supervised vision transformer, with Llama 3, a powerful large language model. The core technical challenge was designing and implementing a mechanism to effectively fuse the visual information (image patch embeddings from DINOv2) with the linguistic information (text token embeddings from Llama 3).

The initial phase involved getting the components loaded and understanding their outputs. Loading DINOv2 was straightforward using the Hugging Face `transformers` library. The key was figuring out which output to use. DINOv2 provides a CLS token embedding and a sequence of patch embeddings. For VQA, I needed the spatial information encoded in the patches. Extracting these and ensuring they were in the right shape (Batch x Num_Patches x Feature_Dim) took a few tries. I remember hitting a `ValueError` because I initially just grabbed the last hidden state sequence and had to carefully index to exclude the CLS token and reshape based on the image patch grid size. This part, including setting up the image preprocessing pipeline (resizing, normalization required by DINOv2), probably took about two evenings.

Loading Llama 3 was similar, also via `transformers`. The decision here was whether to use it as a full generative model or primarily as an encoder to get rich text representations. Given the VQA task structure (question + image -> answer), using it as an encoder seemed more sensible for fusion. I extracted the hidden states for the input question tokens. Aligning the DINOv2 feature dimension with Llama 3's hidden state dimension was necessary. I added a simple linear projection layer on top of the DINOv2 features to match the dimension of the Llama 3 embeddings.

The main technical hurdle, and where I spent the majority of my time – roughly a week of focused effort – was implementing the fusion mechanism. My goal was to allow the image patches and text tokens to "attend" to each other, enabling the model to understand which parts of the image are relevant to which parts of the question and vice-versa. I decided on a cross-attention mechanism. I envisioned the text tokens as queries attending to the image patches (keys and values), and also image patches attending to text tokens.

My initial implementation of the cross-attention module was a mess of shape errors. I kept mixing up the sequence lengths (number of text tokens vs. number of image patches) and the feature dimensions. Debugging involved printing `.shape` everywhere and drawing out the matrix multiplications on paper. I distinctly remember a `RuntimeError: The size of tensor a (1024) must match the size of tensor b (768) at non-singleton dimension 2` when I tried to multiply a query tensor (Batch x Seq_Len_A x Head_Dim) with a key tensor (Batch x Seq_Len_B x Head_Dim) where Seq_Len_A was text tokens and Seq_Len_B was image patches, but I had transposed the wrong dimension for the matrix multiplication. Consulting the PyTorch documentation for `torch.matmul` and some tutorials on implementing Transformer attention from scratch helped clarify the required transpositions.

Another significant challenge was handling the attention masks correctly. Llama 3 inputs have padding tokens, which shouldn't attend to or be attended by real tokens. I needed to create appropriate masks for both the text-to-image and image-to-text attention layers. Initially, I forgot the mask for the image patches attending to text, leading to the model trying to use padding tokens in its visual reasoning. This manifested as confusingly poor performance despite low training loss. Adding the masks correctly required careful broadcasting to match the attention scores' shape (Batch x Num_Heads x Seq_Len_A x Seq_Len_B).

Here's a simplified snippet showing the conceptual structure of the fusion module I built, highlighting some of the shape considerations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer # Assuming these are loaded elsewhere

class MultimodalFusion(nn.Module):
    def __init__(self, text_embed_dim, vision_embed_dim, num_attention_heads, hidden_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_embed_dim, text_embed_dim) # Project vision feats to text dim
        self.num_heads = num_attention_heads
        self.head_dim = text_embed_dim // num_attention_heads # Assume dims are divisible

        # Cross-attention: Text tokens attending to Image Patches
        self.text_to_image_attention = nn.MultiheadAttention(embed_dim=text_embed_dim, num_heads=num_attention_heads, batch_first=True)

        # Cross-attention: Image Patches attending to Text tokens (Optional, but helpful)
        # self.image_to_text_attention = nn.MultiheadAttention(embed_dim=text_embed_dim, num_heads=num_attention_heads, batch_first=True) # <- Considered this, added later

        # Feed-forward network after attention
        self.mlp = nn.Sequential(
            nn.Linear(text_embed_dim + text_embed_dim, hidden_dim), # Concatenating original text + attended image feature
            nn.ReLU(),
            nn.Linear(hidden_dim, text_embed_dim) # Project back to text embed dim? Or leave concatenated?
        ) # Initial thought: just use the output of attention. Realized concatenating original text features was better.

        self.norm1 = nn.LayerNorm(text_embed_dim) # Pre-LN or Post-LN? Experimented with Post-LN initially

    def forward(self, text_embeddings, vision_features, text_attention_mask):
        # text_embeddings shape: (batch_size, num_text_tokens, text_embed_dim)
        # vision_features shape: (batch_size, num_image_patches, vision_embed_dim)
        # text_attention_mask shape: (batch_size, num_text_tokens) -> boolean or byte mask

        # Project vision features to match text embedding dimension
        projected_vision_features = self.vision_proj(vision_features) # (batch_size, num_image_patches, text_embed_dim)

        # --- Text attending to Image ---
        # Queries: text_embeddings
        # Keys/Values: projected_vision_features
        # Need a mask for text tokens (to prevent attending from padding)
        # MultiheadAttention expects key_padding_mask for V/K. Here K/V are image patches, no padding needed.
        # But if I wanted image patches to mask *text* queries, that's different.
        # The standard MultiheadAttention key_padding_mask applies to the KEY/VALUE sequence length.
        # So for text_to_image_attention, no key_padding_mask on vision_features.
        # However, the input queries (text_embeddings) might have padding, which should NOT generate queries.
        # The MultiheadAttention source_mask/attn_mask is for preventing queries from attending to *each other* or certain keys.
        # My use case is query (text token) -> key/value (image patch). No source_mask needed for text queries attending images.
        # Let's check MultiheadAttention docs again... key_padding_mask is applied to the *key* tensor. Okay, confirmed no mask needed *on vision*.
        # But how to prevent padded text tokens from performing attention queries? The standard Transformer implementation usually handles this by NOT generating queries for padded tokens or masking the output.
        # Let's assume the text_attention_mask passed in can be used to zero out the attention output for padded text tokens.
        # Or, more correctly, the self-attention BEFORE this fusion would handle text padding. Here, we just need to make sure image patches don't attend to padding.

        # Re-reading MultiheadAttention docs... ah, `key_padding_mask` masks keys. `attn_mask` masks queries from attending keys.
        # If I want text queries (Q) to ignore image keys/values (K/V) based on the text mask, that's not what key_padding_mask does.
        # Let's use `attn_mask` but need to format it correctly for cross-attention.
        # MultiheadAttention `attn_mask` is (N, S) or (S, S) where N is query sequence length, S is key sequence length.
        # So for text_to_image, query length is text_tokens, key length is image_patches. Mask should be (num_text_tokens, num_image_patches).
        # How to derive this from text_attention_mask (batch_size, num_text_tokens)?
        # If text_attention_mask is 1 for real tokens, 0 for padding:
        # Mask should be True for positions that should be ignored. So inverse of text_attention_mask, and expand for image patches.
        # text_attn_mask = (text_attention_mask == 0) # True for padding
        # text_attn_mask = text_attn_mask.unsqueeze(2).expand(-1, -1, projected_vision_features.size(1)) # (batch, num_text_tokens, num_image_patches) <- This looks right!
        # No, the mask needs to be broadcast over heads and batch. MultiheadAttention expects mask shape (batch_size * num_heads, num_text_tokens, num_image_patches) or just (num_text_tokens, num_image_patches) if not batch_first.
        # Or maybe just pass the original text_attention_mask as `key_padding_mask` if I swap Query/Key? No, that's not the standard cross-attention formulation.
        # This was confusing. After looking at open source VQA implementations using attention, the common pattern was to compute the attention scores (Q @ K.T) and then add a large negative number to masked positions *before* the softmax. MultiheadAttention handles this internally if `attn_mask` is provided in the right shape.

        # Correct approach based on examples:
        # Create mask for text queries from text_attention_mask (1 for real, 0 for padding).
        # Need a mask of shape (batch_size, num_heads, num_text_tokens, num_image_patches)
        # This felt overly complicated for torch.nn.MultiheadAttention. Let's try implementing it manually for clarity first.
        # Q = self.q_linear(text_embeddings).view(B, N_txt, H, Dk).transpose(1, 2) # (B, H, N_txt, Dk)
        # K = self.k_linear(projected_vision_features).view(B, N_img, H, Dk).transpose(1, 2) # (B, H, N_img, Dk)
        # V = self.v_linear(projected_vision_features).view(B, N_img, H, Dv).transpose(1, 2) # (B, H, N_img, Dv)
        # Scores = (Q @ K.transpose(-2, -1)) / sqrt(Dk) # (B, H, N_txt, N_img)
        # Mask: Need to prevent text padding tokens (at index i in N_txt) from attending.
        # Mask should be True at (b, h, i, j) if text_attention_mask[b, i] is 0 (padding).
        # text_padding_mask_expanded = (text_attention_mask == 0).unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, projected_vision_features.size(1)) # (B, H, N_txt, N_img)
        # Add large negative value where mask is True
        # Scores = Scores.masked_fill(text_padding_mask_expanded, -1e9)
        # Attention_weights = F.softmax(Scores, dim=-1) # (B, H, N_txt, N_img)
        # Output = Attention_weights @ V # (B, H, N_txt, Dv)
        # Concatenate heads and final linear layer...

        # Okay, let's go back to nn.MultiheadAttention, it MUST have a way.
        # The `attn_mask` parameter documentation: "If a BoolTensor mask is given, the positions with the value of `True` will be ignored, the attention scores will be set to `-inf` there."
        # Shape: (L, S) or (N*H, L, S) where L is query sequence length, S is key sequence length.
        # For text_to_image: L=num_text_tokens, S=num_image_patches. Mask shape (num_text_tokens, num_image_patches).
        # Still need to expand from text_attention_mask (Batch, num_text_tokens) to (Batch, num_text_tokens, num_image_patches) and handle batching/heads.
        # The batch_first=True mode helps. Mask shape should be (Batch, num_text_tokens, num_image_patches).
        # text_mask_for_attn = (text_attention_mask == 0).unsqueeze(2).expand(-1, -1, projected_vision_features.size(1)) # (batch, num_text_tokens, num_image_patches)
        # This mask should be True where attention *should be ignored*. So True for padding positions.
        # This looks right! Let's try this.

        # Using nn.MultiheadAttention batch_first=True
        # query: text_embeddings (B, N_txt, E)
        # key: projected_vision_features (B, N_img, E)
        # value: projected_vision_features (B, N_img, E)
        # attn_mask: (B, N_txt, N_img) True where attention should be prevented (text padding query positions)
        # key_padding_mask: (B, N_img) True where key (image patch) should be ignored (no image padding here)

        # text_mask_for_attn = (text_attention_mask == 0).unsqueeze(2).expand(-1, -1, projected_vision_features.size(1)) # (batch, num_text_tokens, num_image_patches)

        attended_image_features, _ = self.text_to_image_attention(
            query=text_embeddings,
            key=projected_vision_features,
            value=projected_vision_features,
            attn_mask=text_mask_for_attn, # This should mask queries (text) based on their padding status
            # key_padding_mask=None # No padding on image keys
        ) # Shape: (batch_size, num_text_tokens, text_embed_dim)

        # Concatenate original text embeddings with the attended image features
        combined_features = torch.cat([text_embeddings, attended_image_features], dim=-1) # (batch_size, num_text_tokens, text_embed_dim * 2)

        # Pass through MLP
        fused_output = self.mlp(combined_features) # (batch_size, num_text_tokens, text_embed_dim) # Or maybe hidden_dim if last layer doesn't project back

        # Add residual connection and LayerNorm (Post-LN)
        # fused_output = self.norm1(text_embeddings + fused_output) # This would be if MLP projected back to text_embed_dim

        # Let's refine the MLP output dim based on VQA head
        # If the VQA head pools over text tokens, maybe the MLP output can be higher dim?
        # Let's assume the MLP outputs a fixed size representation per text token.
        # And the VQA head will pool these tokens (e.g., mean pool, or attend to tokens again).

        # Okay, simplest VQA head: pool the fused_output over tokens, then linear layer to answer logits.
        # Mean pooling is simple but might lose info. Attention pooling over tokens?
        # Let's mean pool for now.

        # pooled_output = fused_output.mean(dim=1) # (batch_size, text_embed_dim) # Need to mask padding tokens before pooling!
        # Proper pooling: sum real tokens, divide by count of real tokens
        # real_token_mask = text_attention_mask.unsqueeze(-1) # (batch_size, num_text_tokens, 1)
        # masked_fused_output = fused_output * real_token_mask
        # pooled_output = masked_fused_output.sum(dim=1) / real_token_mask.sum(dim=1) # (batch_size, text_embed_dim)

        # Let's return the fused_output sequence; the VQA head will handle pooling/prediction
        return fused_output, text_attention_mask # Return mask too for downstream use
```
Implementing and debugging this attention logic, particularly getting the masks right for `torch.nn.MultiheadAttention` with `batch_first=True` in a cross-modal setting, was the most time-consuming part. StackOverflow threads discussing `attn_mask` and `key_padding_mask` usage in `MultiheadAttention` were invaluable here.

After the fusion module was stable, I integrated it into a full VQA model. The architecture was straightforward: DINOv2 -> Vision Projection -> Fusion Module (with Llama 3 embeddings) -> VQA Head. For the VQA head, I used a simple mean-pooling layer over the fused text-token representations (making sure to mask out padding tokens before pooling), followed by a linear layer to predict answer probabilities over a predefined vocabulary of common answers (a common approach for VQAv2 evaluation). Training this end-to-end model was the next phase, taking about a week running on a single GPU (RTX 3090).

Memory was a constraint. Llama 3 (even a smaller version) and DINOv2 together are large. I couldn't fine-tune the full DINOv2 or Llama 3 weights. I opted to freeze both the DINOv2 backbone and the Llama 3 backbone, only training the vision projection layer, the cross-attention fusion module, and the VQA head. This kept the number of trainable parameters manageable and allowed training on my hardware. I used the AdamW optimizer with a cosine learning rate schedule. Initial training was a bit wobbly; I had to experiment with the learning rate (started too high) and gradient accumulation to simulate larger batch sizes than physically fit on the GPU.

Evaluation was done on the VQAv2 validation set. Compared to a baseline using simpler concatenation of pooled vision features and text embeddings, my attention-based fusion model showed a noticeable improvement in VQA accuracy, specifically around 3-4 percentage points. While not a state-of-the-art result compared to massive models trained on vast resources, it validated the hypothesis that explicitly attending between image patches and text tokens helps the model ground the question in relevant visual evidence. I observed qualitative improvements particularly on questions requiring understanding spatial relationships or identifying specific attributes of objects mentioned in the question.

The stack for this project was primarily PyTorch, Hugging Face `transformers` and `datasets`, and standard Python libraries for data handling.

Overall, this project reinforced the importance of careful alignment and interaction when combining different modalities. Debugging the attention mask logic in particular was a deep dive into PyTorch's `nn.MultiheadAttention` internals. Freezing large backbones and strategically training only the new components was essential for fitting the project onto limited hardware. The experience provided concrete insights into the practical challenges of building multimodal models beyond just reading about them. Future steps could involve trying more sophisticated fusion mechanisms or exploring methods like LoRA to fine-tune parts of the backbone models within hardware constraints.