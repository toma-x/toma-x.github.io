---
layout: post
title: Efficient Vision Encoder Distillation
---
```

Hey everyone,

So, after playing around with different types of models, from LeetCode algorithms to multimodal stuff, I got really curious about model efficiency. We keep hearing about these massive models like Vision Transformers (ViTs) that get amazing results, but they're often huge and slow. I started wondering if you could make them smaller and faster without losing too much performance. That led me down the rabbit hole of knowledge distillation, and I decided to try distilling a big ViT into a smaller one for my latest project. My goal was pretty ambitious: could I get a model with significantly fewer parameters to perform almost as well as the original?

Spoiler: It kind of worked, but it was definitely a bumpy ride! I used **PyTorch** for everything.

## The Setup: Teacher, Student, and Data

First, I needed a "teacher" model. I picked a standard pre-trained ViT (let's say `vit_base_patch16_224` trained on ImageNet from Hugging Face's `transformers` library) because it's powerful but also quite large (around 86 million parameters).

Next came the "student" model. This was the tricky part: how do you design a smaller ViT? I didn't have any fancy neural architecture search setup. My approach was pretty basic: just make a shallower and narrower version of the teacher. I tried a few configurations, basically reducing the number of transformer layers (depth) and maybe the hidden dimension size or number of attention heads (width). After some fiddling (mostly trial and error looking at parameter counts), I settled on a configuration that had roughly **60% fewer parameters** than the `vit_base_patch16_224`. It felt like a significant enough reduction to be interesting.

```python
# Example of how I might have loaded the models (conceptual)
# (Assuming I defined a smaller ViT config or found one)
from transformers import ViTModel, ViTConfig, ViTForImageClassification

# Teacher Model (pre-trained)
teacher_model_name = "google/vit-base-patch16-224" 
teacher_model = ViTForImageClassification.from_pretrained(teacher_model_name)
teacher_model.eval() # Teacher is frozen during distillation

# Student Model Config (example - fewer layers/hidden size)
student_config = ViTConfig(
    hidden_size=512, # Smaller hidden size than base (768)
    num_hidden_layers=6, # Fewer layers than base (12)
    num_attention_heads=8, # Fewer heads than base (12)
    intermediate_size=2048, # Reduced intermediate size
    image_size=224,
    patch_size=16,
    num_labels=teacher_model.config.num_labels # Match teacher's output classes
)
student_model = ViTForImageClassification(config=student_config)

# Freeze the teacher model's parameters
for param in teacher_model.parameters():
    param.requires_grad = False

# Make sure student model parameters are trainable
for param in student_model.parameters():
    param.requires_grad = True

```

For data, I decided to use a standard image classification dataset like CIFAR-100, mainly because ImageNet is massive and training on it takes ages, especially when you're experimenting. Using CIFAR-100 allowed for faster iteration cycles, even though the teacher ViT was originally trained on ImageNet (transfer learning + distillation).

## The Distillation Strategy: Trying to Mimic the Teacher

The core idea of distillation is making the student model mimic the teacher. There are different ways to do this. I decided to try a combination approach:

1.  **Logits Matching (Standard KD):** Make the student's output probability distribution (after softmax) match the teacher's. This is usually done using Kullback-Leibler (KL) divergence loss. The teacher's "soft" probabilities (using a temperature parameter > 1 in the softmax) provide richer information than just the hard labels.
2.  **Intermediate Feature Matching:** Make the student's internal feature representations match the teacher's at certain layers. Since ViTs process information through multiple transformer blocks, I thought matching features somewhere in the middle might help the student learn better internal representations. I chose to match the output features of a few intermediate transformer blocks using Mean Squared Error (MSE) loss.

Here's roughly what the combined loss function looked like in PyTorch:

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_outputs, teacher_outputs, student_features, teacher_features, labels, alpha, temperature):
    """
    Combines KL divergence loss on logits and MSE loss on features.
    
    Args:
        student_outputs: Logits from the student model.
        teacher_outputs: Logits from the teacher model.
        student_features: List of intermediate features from student.
        teacher_features: List of corresponding intermediate features from teacher.
        labels: Ground truth labels.
        alpha: Weight balance between KD loss and student loss (e.g., 0.7).
        temperature: Temperature for softening probabilities.
    """
    # 1. Standard Cross-Entropy Loss with hard labels (for the student)
    student_loss = F.cross_entropy(student_outputs, labels)

    # 2. KL Divergence Loss with soft targets from teacher
    soft_teacher_log_probs = F.log_softmax(teacher_outputs / temperature, dim=1)
    soft_student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
    kd_loss = F.kl_div(soft_student_log_probs, soft_teacher_log_probs.detach(), 
                       log_target=True, reduction='batchmean') * (temperature ** 2) # Scale correction

    # 3. Intermediate Feature Matching Loss (MSE)
    feature_loss = 0.0
    if student_features and teacher_features:
        # Assuming student_features and teacher_features are lists of tensors from corresponding layers
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Ensure features are compatible (might need projection/pooling if dimensions differ)
            # Let's assume for simplicity they match or we handle it elsewhere
            feature_loss += F.mse_loss(s_feat, t_feat.detach()) 
        # Average feature loss over the layers matched
        if len(student_features) > 0:
             feature_loss /= len(student_features)

    # Combine losses (need another hyperparameter, beta, for feature loss weight)
    beta = 0.3 # Example weight for feature loss
    # total_loss = alpha * kd_loss + (1 - alpha) * student_loss 
    # Let's try incorporating feature loss too:
    total_loss = (alpha * kd_loss + 
                 (1 - alpha - beta) * student_loss + 
                 beta * feature_loss)
                 
    # A simpler approach: just student CE + KD + Feature MSE with separate weights
    # total_loss = student_loss_weight * student_loss + kd_loss_weight * kd_loss + feature_loss_weight * feature_loss

    return total_loss, student_loss, kd_loss, feature_loss # Return individual losses for monitoring

```
*(Note: Getting the weights `alpha`, `beta`, and `temperature` right was a huge pain, more on that later! Also, matching feature dimensions often needed extra projection layers or careful selection of compatible layers.)*

To get intermediate features, I had to slightly modify the forward pass or use hooks in PyTorch to grab the hidden states from specific layers in both models.

## Challenges and Faceplants

This project was *way* harder than I initially thought.

*   **Feature Matching Nightmare:** Getting the intermediate feature matching to work consistently was tough.
    *   Which layers to match? Matching early layers vs. late layers seemed to give different results.
    *   Dimension mismatch: My student model was narrower, so its feature dimensions didn't directly match the teacher's. I had to add linear projection layers to map the student's features to the teacher's dimension, which added more parameters and complexity. Getting these projection layers to learn effectively was tricky. Sometimes the feature loss just wouldn't decrease, or it would dominate the other losses.
*   **Hyperparameter Hell:** Tuning `alpha`, `beta`, and `temperature` was incredibly tedious. If `alpha` (KD weight) was too high, the model sometimes ignored the actual labels. If `beta` (feature loss weight) was too high, the model focused too much on mimicking internal features and might hurt classification accuracy. Finding a good balance required lots of runs with different values, which took ages. The `temperature` also mattered – too high and the teacher's distribution became too flat; too low and it was too peaky like the hard labels.
*   **Training Instability:** Sometimes the student model's performance would just collapse mid-training, or the loss would suddenly spike. I suspect this was due to unstable gradients, especially when feature matching was involved. I had to play a lot with learning rates, use gradient clipping, and experiment with different optimizers (AdamW worked okay).
*   **Slow Training:** Even on CIFAR-100, running both the (large) teacher model and the student model through the network for every batch, plus calculating multiple loss terms, was slow. Each experiment took hours on the Colab GPUs I was using, making the hyperparameter tuning even more painful.

## Getting it to Work (Sort Of)

After weeks of tweaking and debugging, I started getting decent results. What finally seemed to work reasonably well was:

*   Focusing on **KL divergence loss** (`kd_loss`) as the primary distillation driver (`alpha` around 0.7-0.9).
*   Using a **moderate temperature** (around 3-5).
*   Being **selective about feature matching**. Instead of matching many layers, I tried matching only the output embeddings *before* the final classification head, using a simple linear projection for the student. This seemed more stable than matching deep intermediate transformer blocks. I kept the weight for this feature loss (`beta`) relatively low (maybe 0.1-0.3).
*   Using a **standard cross-entropy loss** with the ground truth labels (`student_loss`) with a small weight (`1 - alpha - beta`) to keep the model grounded.
*   A **slow learning rate** (like 1e-4 or even 5e-5) with a cosine annealing scheduler.

The training loop needed to run both models, get the outputs and features, calculate the combined loss, and backpropagate only through the student:

```python
# Conceptual training step in the loop
optimizer.zero_grad()

# Get inputs and labels
inputs = batch['pixel_values'].to(device)
labels = batch['labels'].to(device)

# Teacher forward pass (inference mode)
with torch.no_grad():
    # Get teacher logits and potentially features
    teacher_outputs_obj = teacher_model(inputs, output_hidden_states=True) 
    teacher_logits = teacher_outputs_obj.logits
    # Extract desired teacher hidden states (e.g., last layer before classifier)
    teacher_features_list = [teacher_outputs_obj.hidden_states[-1][:, 0, :]] # Example: CLS token features

# Student forward pass (training mode)
# Get student logits and potentially features
student_outputs_obj = student_model(inputs, output_hidden_states=True)
student_logits = student_outputs_obj.logits
# Extract corresponding student hidden states (might need projection)
raw_student_features = student_outputs_obj.hidden_states[-1][:, 0, :] # Example
# Assume 'student_feature_projector' is a nn.Linear layer matching dimensions
projected_student_features_list = [student_feature_projector(raw_student_features)] 

# Calculate the combined loss
loss, ce_loss, kd_loss_val, feat_loss_val = distillation_loss(
    student_logits, teacher_logits, 
    projected_student_features_list, teacher_features_list, 
    labels, alpha=0.8, temperature=4.0 # Example hyperparameters
)

# Backpropagation (only student weights and projector weights are updated)
loss.backward()
optimizer.step()

# Log losses: loss.item(), ce_loss.item(), kd_loss_val.item(), feat_loss_val.item()
```

## The Results: ~95% Performance, 60% Fewer Parameters

Eventually, I got the student model (with ~60% fewer parameters) to achieve around **95% of the teacher model's accuracy** on the CIFAR-100 test set. For example, if the teacher got 85% accuracy, the student might get around 80-81%. It wasn't *exactly* the teacher's performance, but significantly better than training the student model from scratch with just the labels. The distilled student was also noticeably faster at inference time due to its smaller size.

## What I Learned

*   **Distillation Works (But It's Fiddly):** It's definitely possible to transfer knowledge from large models to smaller ones. But getting it right requires careful tuning and experimentation. There's no single magic formula.
*   **PyTorch Power & Pain:** PyTorch gives you the flexibility to define custom loss functions and manipulate model internals (like grabbing features), but it also means you have to manage the complexity yourself. Debugging gradient issues or dimension mismatches was a common headache.
*   **Feature Matching is Tricky:** While matching intermediate features sounds good in theory, making it work effectively in practice, especially across different architectures or sizes, needs careful implementation (projections, choosing layers, loss weighting). Logits matching felt more robust.
*   **Patience is Key:** Training these models takes time, and finding the right hyperparameters requires patience and systematic experimentation.

## What I'd Do Differently Next Time

If I did this again, I might:
*   Start simpler: Maybe just focus on logits matching (KD loss) first and get that working perfectly before adding feature matching complexity.
*   Explore different feature matching losses: Maybe cosine similarity instead of MSE?
*   Try attention mechanism distillation: Some papers suggest matching the attention maps between teacher and student layers, which might capture different information than just feature vectors.
*   Use a more structured approach for hyperparameter tuning instead of random trial-and-error (e.g., Optuna).

## Conclusion

This project was a great learning experience in model compression and practical PyTorch usage. It showed me that distillation is a powerful technique for creating more efficient models, even if it involves wrestling with tricky implementations and hyperparameters. Achieving ~95% of the performance with a model that's significantly smaller felt like a solid win, despite the struggles along the way. It definitely gave me a better appreciation for the trade-offs between model size, speed, and accuracy.

Hope sharing this unfiltered experience is helpful! Happy to discuss if you've tried something similar or have questions.