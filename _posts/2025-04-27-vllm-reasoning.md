---
layout: post
title: Fine-tuning VLLM for Visual Reasoning
---

Okay, so this project. Fine-tuning a VLLM, sounds fancy right? Vision-Language Model, basically something that looks at a picture *and* reads text at the same time. My goal was Visual Question Answering (VQA) – feed it an image and a question about it, and get an answer back. Simple enough in theory.

Remember how on my resume I mentioned 'Fine-tuning VLLM for Visual Reasoning'? Yeah, this is that. Link's there if you wanna see the messy GitHub eventually, though it's not super polished, be warned: https://toma-x.github.io/vllm-reasoning

First hurdle: data. VQA needs image-question-answer triplets. Loads of public datasets exist, but I wanted something a bit specific, maybe focusing on certain types of questions or images, or honestly, just wanted the challenge of building my *own* little pipeline. Also, playing with Hugging Face stuff, it felt like getting a custom dataset ready was half the battle with fine-tuning.

So, data curation. Ugh. My initial idea was scraping stuff, but that got complicated real fast with licenses and just general messiness. Ended up pulling from a few different smaller sources, manually checking a bunch, and writing this absolutely *janky* Python script to pair image paths with question-answer JSONs. It was a mess. Took like, three days just getting the data into a consistent format. At one point, I swear, the script kept failing because one source used backslashes in paths on my Ubuntu machine, and another used forward slashes. So dumb, but took hours to track down because the error message was something useless about file not found, obviously.

Here's a snippet of that lovely script, slightly cleaned up but you get the idea. Look at those `replace` calls, total hackjob but hey, it worked.

```python
import json
import os

def curate_data(image_dir, qa_file, output_json):
    data = []
    with open(qa_file, 'r') as f:
        qa_pairs = json.load(f) # Assumes a list of {image_id: ..., question: ..., answer: ...}

    # Make a set of existing image files for quick lookup
    valid_images = set(os.listdir(image_dir))

    for item in qa_pairs:
        # This part was the headache
        image_filename = item['image_id'] # or whatever key they used
        # Handle potential path weirdness - spent ages on this
        if '\\' in image_filename:
            image_filename = image_filename.replace('\\', '/')

        image_path = os.path.join(image_dir, image_filename)

        if image_filename in valid_images: # Check if the image file actually exists
            data.append({
                "image_path": image_path,
                "question": item['question'],
                "answer": item['answer']
            })
        else:
            # print(f"Warning: Image {image_filename} not found in {image_dir}") # uncommented this line so much
            pass # Just skip missing images for now, gotta get *some* data

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage (don't judge the hardcoded paths)
# curate_data('/path/to/my/downloaded/images', '/path/to/the/qa_data.json', 'my_curated_dataset.json')
```
Yeah, that `pass` for missing images? Totally lazy, but I had a deadline (self-imposed, but still) and just needed data *now*. Better data curation is definitely a thing I learned is critical and takes forever.

Next, the model itself. VLLM. I decided on one of the larger open ones available via Hugging Face because, well, PyTorch and Transformers are what I know best right now. Loading the pre-trained weights was straightforward enough with `transformers`. The *real* fun was figuring out *what* to fine-tune. Do I train the whole thing? Just the vision part? Just the language part? The projection layer that connects them?

Read a bunch of papers and blog posts (and yes, trawled StackOverflow and the Hugging Face forums). General consensus seemed to be training a small part of the model, maybe just the final layers or a specific adapter module, is more memory efficient and prevents catastrophic forgetting of the pre-training knowledge. My laptop (a decent one, but not a server) wasn't gonna handle training a multi-billion parameter model end-to-end anyway. So, I decided to freeze most of the base model and only train the new head I added for the VQA task and maybe a few layers related to the cross-modal attention.

Here’s a bit of the training setup code. Notice the `requires_grad = False` loops. This was after I accidentally tried training everything and immediately got CUDA out of memory errors. Took a while to pinpoint exactly which parameter groups to unfreeze for the best balance of performance and memory.

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

# Using VILT as an example, because that's one I looked at
model_name = "dandelin/vilt-b32-finetuned-vqa" # This is actually a finetuned one, but imagine it's a base VILT
processor = ViltProcessor.from_pretrained(model_name)
model = ViltForQuestionAnswering.from_pretrained(model_name)

# Freeze most of the model
for param in model.vilt.parameters():
    param.requires_grad = False

# Unfreeze the classifier head (the VQA specific part)
# The exact layer names took some digging in the model's source code/documentation
# This is just illustrative, actual names differ!
for param in model.classifier.parameters():
    param.requires_grad = True

# Maybe unfreeze some specific attention layers too?
# This was trial and error based on what seemed reasonable from papers
# Example (might not be the real name):
# for i in range(model.vilt.encoder.layer): # Loop through layers
#     if i >= 10: # Unfreeze the last few transformer layers
#          for param in model.vilt.encoder.layer[i].parameters():
#              param.requires_grad = True

# Optimizer setup - AdamW is standard
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Loss function - Cross-Entropy for classification (assuming VQA is treated as classification over possible answers)
# Getting the answer space mapped to IDs was another data prep step
# criterion = torch.nn.CrossEntropyLoss()
```
Debugging the training loop itself was... fun. Learning rate too high? Loss goes to NaN. Batch size too big? OOM error again. Learning rate too low? Takes forever and doesn't learn. Spent a good chunk of a weekend just fiddling with hyperparameters and watching the loss curve jump all over the place on TensorBoard. At like 2 AM one night, I finally got a stable loss curve that looked like it was actually decreasing consistently. Felt like winning the lottery.

Evaluation was pretty standard. Load the best checkpoint, run it on the test set, calculate accuracy. Simple, right? Except getting the model output (logits over possible answers) mapped back to the actual text answers needed careful indexing and making sure the answer vocabulary used during training matched the evaluation script. Another place for off-by-one errors or key mismatches.

```python
# Snippet from evaluation script
import torch
# Assuming model and processor are loaded and data loader is ready
# Assuming answer_idx_to_str maps index back to text answer

model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad(): # Super important! Don't calculate gradients during eval
    for batch in test_dataloader:
        images = batch["pixel_values"].to("cuda") # If using GPU
        questions = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda") # Ground truth answer indices

        outputs = model(input_ids=questions, pixel_values=images, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        predicted_indices = torch.argmax(logits, dim=-1) # Get the index of the highest logit

        # Map predicted indices back to text answers if needed, or compare indices directly if labels are indices
        correct_predictions += (predicted_indices == labels).sum().item()
        total_predictions += labels.size(0)

accuracy = correct_predictions / total_predictions
print(f"Test Accuracy: {accuracy:.4f}")

# This accuracy number felt hard-earned, let me tell you.
```

Honestly, the whole process was a massive headache sometimes, mostly due to the data side and then the fiddly bits of getting the model training stable. But seeing it actually answer questions about images, even if it wasn't perfect and sometimes hallucinated weird stuff, was pretty cool. It hammered home how important good data is, and how much trial and error goes into making these models work for a specific task. Definitely learned way more doing it than just reading about it.

Alternatives? Thought about trying something like CLIP orFlorence, but VILT seemed like a solid starting point available in Hugging Face that was designed for vision-language tasks from the get-go. Also considered maybe using a different finetuning strategy, like LoRA, but adding adapters felt like another layer of complexity I didn't have time to fully dive into and debug within my project timeline. Gotta pick your battles when you're coding against the clock (even a self-imposed one).

Anyway, project done. Learned a ton about VLLMs, data pipelines (even janky ones), and the sheer patience required for debugging deep learning models. Onto the next thing I guess.