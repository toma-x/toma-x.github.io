---
layout: post
title: Custom Vision-Language Encoder
---

Okay, so I finally finished this project I've been hacking away at for... feels like forever. Like, started it last semester break and just wrapped up the evaluation part last week. The goal was to build something that could understand both images and text and figure out if they match. Kinda like, show it a picture of a cat and a caption "A fluffy cat sleeping," and it says "yeah, that looks right," but show it "A car driving down the street" and it says "nah, totally wrong."

Called it my "Custom Vision-Language Encoder." Sounds kinda official, right? The idea was to take an image, crunch it down into a vector (a bunch of numbers representing its features), and take a sentence, crunch *that* into a vector, and then see how "close" those vectors are. Close vectors mean they match.

First hurdle: the image part. How do you turn an image into a vector? Read a bunch of papers, and everyone uses these big convolutional neural networks. ResNet came up a lot. Seemed like a solid choice. I decided to use a pre-trained ResNet-50 from `torchvision` because training one from scratch? Yeah, my laptop would probably burst into flames.

```python
import torchvision.models as models
import torch
import torch.nn as nn

# Started with ResNet18, but read 50 is better for more complex tasks
# resnet_base = models.resnet18(pretrained=True) # nope, upgrade!
resnet_base = models.resnet50(pretrained=True)

# Okay, how to get features *before* the final classification?
# print(resnet_base) # Gotta peek inside...
# Looks like the last layer is 'fc'. Need to remove that.
# Also has an 'avgpool'. Let's keep that for now, output shape seems easier.

# This builds a new model without the last layer
modules = list(resnet_base.children())[:-1]
self.vision_encoder = nn.Sequential(*modules)

# Trying it out with a dummy image tensor
# dummy_img = torch.randn(1, 3, 224, 224) # Batch size 1, 3 channels, 224x224 pixels
# features = self.vision_encoder(dummy_img)
# print(features.shape) # Got torch.Size(). Good. Need to flatten later though.
```
Getting that vision encoder set up wasn't too bad, thanks to `torchvision`. The tricky part was figuring out *exactly* which layer to snip off and what the output shape would be. Took me a bit of trial and error printing the tensor shapes.

Next, the text part. How to turn a sentence into a vector? Transformers are the hot thing now. Hugging Face `transformers` library saved my life here. Picked a pre-trained BERT model (`bert-base-uncased`).

```python
from transformers import BertModel, BertTokenizer

# standard BERT uncased model
self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and get output
# text = "A brown dog running in the park."
# encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=77) # Padding is important! Max length too.
# output = self.text_encoder(**encoded_input)

# How to get a single vector for the whole sentence?
# Option 1: Use the [CLS] token output? (Common practice)
# sentence_embedding = output.last_hidden_state[:, 0, :] # This is the [CLS] token embedding
# print(sentence_embedding.shape) # torch.Size() for bert-base. Okay.

# Need to make image and text vectors the same size! ResNet gave 2048, BERT gave 768.
# Add linear layers to project them to a common dimension, picked 512.
self.vision_projection = nn.Linear(2048, 512)
self.text_projection = nn.Linear(768, 512)

# Then later, pass the embeddings through these layers:
# img_vec = self.vision_projection(img_features.squeeze(-1).squeeze(-1)) # Flatten to before projection
# text_vec = self.text_projection(sentence_embedding)
```
Integrating the Transformer was mostly following Hugging Face examples, but deciding *how* to get a single sentence vector and then making sure its dimension matched the image vector's dimension after the ResNet was... a moment. Just slapping on a linear layer felt kinda basic, but hey, gotta start simple.

The whole point was to compare these image and text vectors. Cosine similarity seemed like the standard way to measure how "aligned" they are. Higher similarity = better match.

```python
import torch.nn.functional as F

# Assuming img_vec and text_vec are normalized (unit length)
# For a batch of N images and N texts (correct pairs)
# img_batch_vecs = torch.randn(N, 512) # Dummy batch
# text_batch_vecs = torch.randn(N, 512) # Dummy batch

# Important: normalize before computing similarity
# img_batch_vecs = F.normalize(img_batch_vecs, dim=1)
# text_batch_vecs = F.normalize(text_batch_vecs, dim=1)

# The similarity matrix: sim[i, j] is similarity between image i and text j
# sim_matrix = torch.matmul(img_batch_vecs, text_batch_vecs.T) # N x N matrix

# Diagonal elements are the positive pairs. Off-diagonals are negatives.
```

The hard part was training it. Had to use a contrastive loss function, specifically InfoNCE. This loss pushes the similarity of correct pairs (the diagonal of the matrix above) high and the similarity of incorrect pairs (off-diagonal) low. Implementing this correctly in PyTorch took me *ages*. Indexing, matrix transposes, applying the loss... so many subtle bugs. Spent three days debugging just the loss function calculation, kept getting weird gradients or the loss wouldn't go down. Turns out, I messed up the indexing for the text-to-image part of the loss calculation. At 2 AM I finally realized I needed to transpose the similarity matrix before applying the cross-entropy loss for the text side. Duh.

```python
# Simplified InfoNCE loss implementation sketch
# Assume sim_matrix is N x N, already divided by temperature tau
# sim_matrix = sim_matrix / self.tau

# Target is the diagonal - pairs (0,0), (1,1), ..., (N-1, N-1)
# labels = torch.arange(sim_matrix.size(0)).long().to(sim_matrix.device) # Need labels 0, 1, ..., N-1

# Image-to-text loss: for each image i, maximize similarity with text i
# cross_entropy expects logits (sim_matrix) and target indices (labels)
# image_loss = F.cross_entropy(sim_matrix, labels)

# Text-to-image loss: for each text j, maximize similarity with image j
# Need to transpose the matrix to swap image/text axes
# text_loss = F.cross_entropy(sim_matrix.T, labels) # <--- THIS LINE was the bug for days! Forgot .T initially.

# Total loss is the average
# total_loss = (image_loss + text_loss) / 2
```
Dataset: Used COCO. It's massive, like, over 100k images, each with 5 captions. Getting the data loaded correctly, pairing images with their corresponding 5 captions, creating batches... that was another mini-project. Used the Karpathy splits to make sure my evaluation was comparable to others. Had to write custom dataset and dataloader classes for PyTorch.

Training on my Ubuntu laptop with a modest GPU (GeForce GTX 1050 Ti, bless its heart) was *painful*. Batch size had to be tiny, like 32, to avoid `RuntimeError: CUDA out of memory`. Switched to mixed precision training which helped a bit but was another configuration headache. Training took forever. Like, days for just a few epochs. Had to save checkpoints religiously in case my machine crashed or I needed to stop.

The breakthrough moment wasn't one big thing, but a series of small wins. Fixing the loss function, getting the data loading right, finding a learning rate that didn't make the loss explode. Seeing the retrieval metrics (Recall@K) actually increase during training was so satisfying. Like, after a week of just flat loss, the numbers finally started moving!

For evaluation, I computed Recall@K (specifically R@1, R@5, R@10) for both image-to-text and text-to-image retrieval on the COCO test set. This is basically asking: if I give the model an image, is the correct text description found within the top K most similar texts? And vice versa.

Managed to get R@1 scores something like ~42% for image-to-text and ~58% for text-to-image. Maybe not absolute top-of-the-leaderboards SOTA, but compared to baseline (random guessing is basically 0%) and results I saw reported by other students or in older papers, this felt pretty damn good. Like, yeah, this custom little system I built actually works and performs competitively *on my setup* and with my limited training time. Calling it "state-of-the-art" feels right for *my* achievement level, you know?

Overall? This project was a beast. Learned more debugging PyTorch and understanding model architecture than in any class. Hit so many walls, consulted so many StackOverflow threads and paper implementations. Would I do it differently? Probably try a more complex cross-attention mechanism between vision and text features instead of just projecting them and doing cosine similarity. But for a first deep dive into multimodal stuff, building this modular vision+text encoder and getting it to work felt like a huge win. Definitely worth the headaches and late nights. What's next? Maybe fine-tune the encoders instead of just using pre-trained? Or try a different loss? Who knows... gotta recover first.
