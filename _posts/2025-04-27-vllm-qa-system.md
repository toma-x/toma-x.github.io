---
layout: post
title: Multimodal Visual QA System
---

Okay, so, finished this project recently. It's a multimodal visual question answering system, kinda like asking a computer about a picture and getting a text answer back. The main thing I used was this model called BLIP-2. Specifically, the VLLM version? Or was it fine-tuning BLIP-2 *with* a VLLM backbone? Ugh, terminology is still messy. Anyway, the point is, it connects vision and language stuff.

Getting started was... well, you know. Setting up the environment on my Ubuntu laptop was the usual dance. Had PyTorch already, thank god, but getting all the dependencies for the BLIP-2 stuff and the dataset loading libraries playing nice? Took a solid afternoon. Had some weird `ImportError`s about `transformers` versions clashing with something else, think it was `accelerate`? Had to downgrade `transformers` then upgrade `accelerate`, or maybe the other way around. Honestly, just fiddled with `pip install -r requirements.txt` and manually fixing versions based on StackOverflow until it stopped screaming. Classic.

The dataset was a curated visual question answering one. Curated by someone else, not me, thank goodness. Loading it was okay, just standard PyTorch `Dataset` and `DataLoader` stuff. Had to write a custom collate function, which always feels a bit hacky but whatever, it works. Needed to process the images (resize, normalize, standard vision transforms) and tokenize the questions and answers. Used the tokenizer that came with BLIP-2.

Fine-tuning the BLIP-2 VLLM model was the core bit. Loaded the pre-trained weights first. PyTorch is pretty straightforward for this, mostly just defining the model, the optimizer (AdamW, duh), and a learning rate scheduler (CosineAnnealing, felt fancy).

The initial training loop was bog standard:

```python
# training loop pseudocode, rough sketch from memory
model.train()
optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    images = batch['images'].to(device)
    questions = batch['questions'].to(device)
    answers = batch['answers'].to(device) # These are target tokens

    # forward pass
    # BLIP-2 takes image and text input, outputs text logits
    # The exact input format was a bit confusing at first
    # Had to check the docs like 10 times to get the padding/attention masks right
    outputs = model(images, questions, answers) # Or something like this? Need to check notes...
    loss = outputs.loss # Assuming model returns loss directly, which BLIP-2 often does
    
    loss.backward()
    optimizer.step()
    lr_scheduler.step() # If using step-based scheduler
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# ... rest of training, validation, saving checkpoints
```
This part was okay, just slow. Training took ages even on a decent GPU (had access to one through my university account, not my laptop, thank god). Watching the loss go down was satisfying, but when I tested the model, the answers were... inconsistent. Like, ask the same question about the same image twice, get slightly different or just plain wrong answers sometimes. Super annoying.

The biggest headache was trying to fix this answer inconsistency. The project goal was to reduce it by 15%. How do you even measure 'inconsistency' precisely? We decided on evaluating a fixed test set multiple times and counting how often the model gave a different answer for the same input, and also doing some qualitative human evaluation.

My initial thought was just *more* data or *more* training. Didn't help much. Then I looked into data augmentation. For the images, simple stuff like random crops, flips, color jitter. For the text, it was harder. Found some papers talking about paraphrasing questions, but implementing that reliably felt like a rabbit hole I didn't have time for. So I stuck to image augmentation, which helped a little, maybe shaved off a few percent?

The main push came from trying to apply RLHF principles. Reinforcement Learning from Human Feedback. Sounded super complex, and honestly, it *was*. The idea is to train a reward model that scores how 'good' an answer is, then use that reward model to further fine-tune the VQA model using RL.

Building the reward model itself was a mini-project. Needed a dataset of (image, question, model answer, human score) triples. Had to generate a bunch of answers from the initial fine-tuned model and manually score them. This took forever, like, several days just labeling data. The reward model was basically another small language model (or maybe just a regression head on top of a model) trained to predict the human score given the image, question, and generated answer.

```python
# Sketch of the reward model structure
class RewardModel(nn.Module):
    def __init__(self, base_model): # base_model could be something that processes image+text
        super().__init__()
        self.base = base_model # e.g., frozen BLIP-2 components or a different smaller model
        self.regressor = nn.Linear(base.output_dim, 1) # Predict a scalar score

    def forward(self, images, questions, answers):
        # Process inputs through base model
        features = self.base(images, questions, answers) # Need to figure out how base model takes all 3
        score = self.regressor(features)
        return score
```
Training the reward model had its own issues. Getting the scores to be consistent felt like trying to nail jello to a wall. Loss would jump around. Realized at one point (it was like 1 AM, classic late-night coding revelation) I was feeding the *token IDs* of the answer to the regressor directly instead of the *embeddings*. Facepalm. Fixed that, and the reward model training became slightly more stable.

Then came the RL part. This was the deepest end. Used Proximal Policy Optimization (PPO) from some RL library (can't remember which one off the top of my head, maybe `trl` or was it a custom thing? Notes are somewhere...). The VQA model became the 'policy', generating answers. The reward model provided the 'reward'. The state was the image and question. The actions were generating tokens.

This PPO training loop was brutal.

```python
# PPO training sketch - very simplified
ppo_trainer = PPOTrainer(...) # Configured with policy model (VQA), reward model, etc.

for epoch in range(num_ppo_epochs):
    for step, batch in enumerate(dataloader): # Use the VQA dataset again
        images = batch['images'].to(device)
        questions = batch['questions'].to(device)

        # Generate answers using the current VQA policy model
        # This part needs sampling, not greedy decoding
        # Getting the sampling parameters right was trial and error
        response, attention_mask, response_logprobs = ppo_trainer.generate(...) 

        # Calculate reward for the generated responses
        # This needed careful formatting of the inputs for the reward model
        rewards = reward_model(images, questions, response)

        # Train the VQA model (policy) using PPO based on rewards and logprobs
        stats = ppo_trainer.step(...) # Takes input, generated response, logprobs, rewards

        if step % 50 == 0:
            print(f"PPO Epoch {epoch}, Step {step}, Reward: {rewards.mean()}")
            # Log other stats like KL divergence, policy loss, value loss
```
The PPO training was incredibly sensitive to hyperparameters. Learning rate for the policy, KL divergence coefficients, batch size for the RL step vs. the data loading step... It felt like tuning a dozen knobs blindfolded. Spent maybe three full days just trying different combinations, watching the metrics the PPO trainer provided (KL divergence exploding was a common and frustrating sign). I remember seeing `NaN`s appearing in the loss and tracking that down to numerical instability somewhere in the reward calculation or normalization. Had to add some epsilon smoothing here and there based on forum suggestions.

Finally, after all that tuning and debugging (including one particularly nasty bug where the attention masks weren't being handled correctly during generation in the PPO loop, leading to garbage output), it started to work. The answers from the RL-tuned model were noticeably more consistent when I did manual checks. When we ran the evaluation metrics again, it showed about a 15% reduction in the inconsistency score compared to the initial fine-tuned model. Success! Kind of. It's still not perfect, obviously, RLHF is complex and my implementation is probably far from optimal, but hey, 15% is 15%.

Learned a ton doing this, mostly about how messy real-world model fine-tuning and these advanced techniques like RLHF can be. Papers make it sound easy, but the devil's really in the details and the debugging. And the hyperparameter tuning. Ugh. But shipping it feels good. Now, time to document this mess properly before I forget how any of it works.
