---
layout: post
title: RLHF for Multimodal Reasoning
---

Okay, so remember that "RLHF for Multimodal Reasoning" thing on my resume? Yeah, let's talk about that. It sounded cool, right? Like, "Oh, I'll just apply RLHF to make a vision-language model think better." Turns out, applying cool concepts is way messier than reading about them.

Started with this idea: VLLMs are okay at describing images, maybe answering simple questions, but actual *reasoning* over an image? Like, "Why did the person in the picture look surprised?" or "What's the most likely next action based on this scene?" They kinda suck. And everyone's hyping RLHF for text models, making them less toxic, follow instructions better. So I figured, why not try to use that human-preference signal, but like, for *multimodal* reasoning?

First hurdle: finding a base multimodal model that wasn't some giant, inaccessible thing. Played around with a few smaller Hugging Face ones that claimed to be multimodal. Spent like two whole days just fighting with getting one loaded and running inference without my laptop's 3060 screaming and dying. Turns out versions matter. A lot. `transformers` version X, `pytorch` version Y, `cuda` Z... felt like dependency hell part 3. Finally settled on one after trying three different ones. It wasn't the fanciest, but it ran. Barely.

Okay, base model loaded. Now, the RLHF part. The core idea is you show the model an input (image + question), get a few possible answers, have a "human" rank them, and use that ranking to train a reward model, then use RL to fine-tune the VLLM based on that reward. But, uh, I don't have a farm of humans ranking multimodal outputs. So, "simulated preferences." This was the hacky bit.

How do you simulate a human judging reasoning? I couldn't come up with some perfect algorithm. My approach was... simple. I generated a dataset of image-question pairs. For each, I'd generate *multiple* potential answers from the base model. Then, I wrote some heuristic functions to score these answers. These heuristics were super basic, like:
1.  Does the answer *directly* reference objects mentioned in the question/image? (Simple keyword matching initially, refined slightly).
2.  Does it seem to follow a logical structure? (This was the hardest to automate, ended up being really crude – looking for causal words or comparative phrases).
3.  Does it avoid obviously wrong statements about the image? (Needed a way to check factual consistency, which was basically impossible perfectly, settled for checking if it contradicted simple facts I *could* verify).

This scoring system was my "simulated human preference." It was buggy, biased by my own assumptions, and definitely not a real human, but it gave me numbers to work with. I'd generate, say, 4 responses per query, score them with my heuristics, and boom, simulated ranking.

```python
# Pseudo-code for the janky scoring function
def score_multimodal_reasoning(image, question, answer):
    score = 0.0

    # Check if answer mentions key objects from the question/image (super basic)
    question_tokens = set(question.lower().split())
    answer_tokens = set(answer.lower().split())
    common_words = len(question_tokens.intersection(answer_tokens))
    score += common_words * 0.1 # Arbitrary weight lol

    # Try to detect logical flow (even more basic)
    if any(word in answer.lower() for word in ["because", "therefore", "so", "but", "while"]):
        score += 0.5 # Wow, it used a transition word, must be reasoning!

    # Negative points for obvious contradictions (hardcoded stuff initially)
    if "dog" in question and "cat" in answer.lower() and "dog" in answer.lower():
         # This check was slightly smarter, see if it mixes up key entities
         score -= 1.0
    # ... more specific negative rules I added as I found failure cases

    # This was NOT good, spent ages tweaking weights and rules by hand
    # Based purely on looking at examples and saying "yeah, this one feels better"
    # No rigorous validation whatsoever

    return score
```

Seriously, that `score_multimodal_reasoning` function was like 200 lines of increasingly specific, hand-tuned `if` statements. It was awful, but it produced a score, and I could use that score to rank the model's own generated answers for a given input. That ranking *became* the "human feedback" I used to train a simple reward model.

The reward model was just a small neural net (like 2-3 layers) that took the image features, question features, and answer features and tried to predict the score from my heuristic function. Training *that* was another mess. Getting the features aligned, making sure it wasn't just overfitting to my garbage heuristic... standard reward model training woes, but now with visual data adding complexity.

Finally, the RL part. I used a simplified PPO setup, mostly adapting existing PyTorch examples. The VLLM was the policy model. The input was (image, question). The action space was generating tokens for the answer. The reward signal came from the *trained* reward model evaluating the *new* answers generated by the VLLM during RL training.

This loop was the absolute worst to debug. Policy generates text -> feed text+image+question to reward model -> get reward -> use reward to update policy via PPO loss.
```python
# This is where I cried, adapting PPO from a text-only example
# Had to get image features into the policy AND the value head
class MultimodalPolicyWithRLHead(nn.Module):
    def __init__(self, base_vllm, reward_model):
        super().__init__()
        self.base_vllm = base_vllm # The not-so-fancy VLLM I got working
        # Need to extract its feature backbone or something... this was pain
        self.vision_encoder = base_vllm.get_vision_encoder() # giả định có hàm này
        self.language_model = base_vllm.get_language_model() # giả định có hàm này

        # Add a value head for PPO's critic
        # This needed multimodal input too... just concatenated features? lol yeah
        feature_dim = self.vision_encoder.output_dim + self.language_model.hidden_size # Hacked this
        self.value_head = nn.Linear(feature_dim, 1)

        # The reward model is fixed after its training phase
        self.reward_model = reward_model
        self.reward_model.eval() # Don't train this here

    def forward(self, pixel_values, input_ids, attention_mask):
        # This forward pass was for generating text during RL rollout
        # Had to handle generation logic, probabilities, etc. Standard stuff but tied to images
        # And then for the value head...
        vision_features = self.vision_encoder(pixel_values)
        lang_features = self.language_model(input_ids, attention_mask).last_hidden_state
        # How to combine them? Avg pooling vision, take last token from language? Ugh.
        # Let's just take the last language token and a pooled vision feature... maybe avg?
        pooled_vision = vision_features.mean(dim=) # Assuming CNN output shape
        last_lang_token = lang_features[:, -1, :]
        combined_features = torch.cat([pooled_vision, last_lang_token], dim=-1) # This was likely wrong shape half the time
        value = self.value_head(combined_features) # Predict value baseline for PPO

        # Also need the language model output logits for the policy loss... it was complex.
        # Logits extraction from base_vllm needed digging into its internals
        logits = self.language_model(input_ids, attention_mask).logits
        return logits, value # And actual generated text elsewhere
```
The loss function involved clipping, entropy bonuses, calculating advantages... standard PPO, but the *data flow* with image features and text features going everywhere, and getting the reward signal back *correctly* aligned with the actions (generated tokens) was a headache. Shape mismatches, CUDA OOM errors because the batches were too big even for my small model, gradients not flowing correctly... Spent three days straight, fueled by instant noodles and bad coffee, just trying to get the PPO loss to look reasonable and the model to not collapse or just generate garbage. Print statements everywhere. Checking gradients manually with `tensor.grad`. This was around 2 AM one night I finally saw the loss moving in the right direction consistently. Relief, but also pure exhaustion.

Evaluating "higher accuracy" was another approximation. Since I didn't have real human judgments, I evaluated the fine-tuned model using the *same* heuristic scoring function I used to generate the simulated preferences. I know, circular logic much? But the idea was, if the RLHF process worked, the model should generate answers that score higher on *that specific* heuristic, meaning it learned to mimic whatever "reasoning" patterns my janky function favored. It did show an improvement based on *my* metric, which was something, I guess. It wasn't real-world reasoning improvement, it was improvement relative to my flawed simulation.

Lessons learned:
1.  Simulating complex human judgment like "reasoning" is incredibly hard and prone to bias. My heuristic was a hack, but it was a necessary hack given the constraints (solo project, no budget for human annotators).
2.  Applying RL algorithms, especially PPO, to custom model architectures (like my weird VLLM adaptation) is non-trivial. The standard tutorials often don't cover stitching together vision and language features for the value head or managing the reward signal flow.
3.  Dependency management will be the death of me.
4.  Getting *any* custom deep learning loop running end-to-end, even with simulated data and flawed metrics, feels like a major win after the debugging marathon.

It's far from perfect, the "reasoning" it learned is just whatever my bad heuristic valued, but the process of actually building the simulated loop and getting RL to train a multimodal model based on it... yeah, that was the project. Messy, frustrating, educational.