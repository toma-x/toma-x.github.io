---
layout: post
title: LLM Financial Sentiment Analyzer
---

## Fine-Tuning BERT for Financial News Sentiment: A Deep Dive

This semester, I decided to tackle a project that combined my growing interest in Natural Language Processing with the ever-dynamic world of finance. The goal was to develop a model capable of analyzing the sentiment of financial news headlines in real-time. After a fair bit of research and a lot of coding, I ended up with a fine-tuned BERT model served via a FastAPI endpoint, and I'm pretty pleased with how it turned out, even if the journey was, well, educational.

### The Starting Point: Finding the Right Words (and Labels)

The first hurdle, as it often is with NLP projects, was the data. I needed a dataset of financial news headlines that were already labeled with sentiment (positive, negative, neutral). I spent a good week or so just exploring options. I looked into scraping news from financial websites, but the legal and ethical gray areas, not to mention the sheer effort of building a robust scraper and then labeling everything myself, felt like a whole separate project. Some datasets on Kaggle seemed promising, but were either too small, too noisy, or not specific enough to financial news.

Eventually, I stumbled upon the "Financial PhraseBank" dataset by Malo et al. (2014), which is fairly well-known in financial sentiment analysis research. It contains sentences from financial news, annotated by multiple people. While not strictly *news headlines*, it was close enough and, crucially, already labeled. I downloaded a version that had been slightly preprocessed by someone else, which saved me some initial cleaning steps. It wasn't perfect – there were still some inconsistencies and a bit of a class imbalance (more neutral statements than I would have liked), but it felt like a solid foundation to start with.

Preprocessing involved the usual suspects: converting text to lowercase, and then the crucial step of tokenization. Since I planned to use BERT, I knew I had to use its specific tokenizer.

```python
# An early look at loading and tokenizing
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sample_text = "Profits surged for the tech giant in the last quarter."

# Seeing what the tokenizer does
tokens = tokenizer.tokenize(sample_text)
# Output: ['profits', 'surged', 'for', 'the', 'tech', 'giant', 'in', 'the', 'last', 'quarter', '.']

encoded_input = tokenizer(sample_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
# encoded_input contains 'input_ids', 'token_type_ids', 'attention_mask'
# Getting this part right, especially the arguments for padding and truncation, took a few tries.
# I remember initially forgetting return_tensors='pt' and wondering why PyTorch was complaining.
```
I had to map the sentiment labels (like "positive", "negative", "neutral") to numerical IDs (0, 1, 2) for the model.

### The Main Event: Wrestling with BERT

I chose BERT (Bidirectional Encoder Representations from Transformers) because of its strong performance on a wide range of NLP tasks. The idea of using a pre-trained model and fine-tuning it on my specific dataset was appealing, as training a large transformer from scratch is well beyond the resources of a student project (and my laptop!). I considered other models like RoBERTa or even simpler LSTM-based networks. LSTMs would have been quicker to train, but I suspected they wouldn't capture the nuances of financial language as well as a transformer. Between BERT and RoBERTa, BERT seemed to have more readily available tutorials and community support for the specific `bert-base-uncased` variant, which felt like a safer bet.

My environment was PyTorch-based, primarily leveraging the Hugging Face `transformers` library. This library is fantastic, but it has a steep learning curve.

```python
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

# Assuming 'input_ids', 'attention_masks', 'labels_tensor' are already prepared PyTorch tensors
# This part was tricky - ensuring all my data was correctly converted to tensors of the right shape.

dataset = TensorDataset(input_ids, attention_masks, labels_tensor)

# Splitting the data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True) # Small batch size due to GPU memory
val_dataloader = DataLoader(val_dataset, batch_size=16)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3, # positive, negative, neutral
    output_attentions=False,
    output_hidden_states=False,
)

# Running on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) # A common learning rate for fine-tuning BERT
```

The first few training runs were… humbling. My validation loss would start decreasing and then suddenly shoot up, a classic sign of overfitting. Or, it would barely budge. I spent a lot of time on Google Colab, taking advantage of their free GPUs because training on my local machine's CPU was impossibly slow (each epoch took ages). Colab sessions timing out in the middle of a long training run was a recurring frustration, forcing me to learn how to save and load model checkpoints diligently.

I experimented with different learning rates. The `2e-5` value is often recommended as a starting point for BERT fine-tuning, but I tried `5e-5`, `3e-5`, and `1e-5`. Too high, and the model diverged; too low, and progress was glacial. Batch size was another constraint. I initially tried a batch size of 32, but kept hitting CUDA "out of memory" errors on Colab's K80 GPUs. Dropping it to 16, and sometimes even 8 for more complex models, became necessary.

One specific issue I remember grappling with was the exact input format for BERT. The `input_ids`, `attention_mask`, and (sometimes, depending on the task) `token_type_ids` all need to be just right. I had a persistent shape mismatch error at one point, and it took me an entire evening of `print(tensor.shape)` and comparing with the Hugging Face documentation examples to finally spot that I was not padding one of the sequences correctly before batching. I found a particular StackOverflow answer that clarified how `attention_mask` should be formed when using padding, which was a lifesaver.

After about 2-3 epochs of fine-tuning (more than that often led to overfitting on this dataset), I started seeing good results. I focused on accuracy, but also kept an eye on precision and recall for each class, especially since "neutral" was so prevalent. Reaching about 90% accuracy on my validation set felt like a significant breakthrough. It wasn't state-of-the-art by research standards, but for a personal project, it was a huge step.

```python
# Simplified training loop snippet
# epochs = 3
# for epoch_i in range(0, epochs):
#     print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
#     model.train()
#     total_train_loss = 0
#     for step, batch in enumerate(train_dataloader):
#         if step % 50 == 0 and not step == 0:
#             print(f'  Batch {step}  of  {len(train_dataloader)}.')

#         b_input_ids = batch.to(device)
#         b_input_mask = batch.to(device)
#         b_labels = batch.to(device)

#         model.zero_grad() # Clear previously calculated gradients
#         result = model(b_input_ids,
#                        token_type_ids=None, # Not always needed for sequence classification
#                        attention_mask=b_input_mask,
#                        labels=b_labels,
#                        return_dict=True)

#         loss = result.loss
#         logits = result.logits # Raw scores output by the model

#         total_train_loss += loss.item()
#         loss.backward() # Perform backpropagation
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping to prevent exploding gradients
#         optimizer.step() # Update parameters

#     # ... (validation loop would go here) ...
# print("Training complete!")
# torch.save(model.state_dict(), "fin_sentiment_bert_model.pth") # Save the model
```
The `torch.nn.utils.clip_grad_norm_` line was something I added after reading about training stability for large networks. It seemed to help prevent those occasional wild loss spikes.

### Serving the Model with FastAPI: Making it Real

A model sitting in a `.pth` file isn't very useful. I wanted to create an API endpoint that could take a piece of financial news text and return its sentiment. I chose FastAPI for this. I had used Flask for simpler web apps before, but FastAPI's automatic data validation with Pydantic, its native asynchronous support (though I didn't delve too deeply into async for this version), and the automatic Swagger UI for API documentation were major draws. It felt more modern and robust for API development.

The main challenge here was loading the fine-tuned PyTorch model and tokenizer efficiently within the FastAPI application. I needed to ensure they were loaded once when the application started, not on every API request, which would be incredibly slow.

```python
# main.py - FastAPI app
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np # For argmax

app = FastAPI()

# Global variables for model and tokenizer
# This is a common pattern, but for larger apps, dependency injection might be better
MODEL_PATH = "fin_sentiment_bert_model.pth" # Path to my saved model
TOKENIZER_NAME = 'bert-base-uncased'

model = None
tokenizer = None
device = None

# This event handler loads the model when the app starts
@app.on_event("startup")
async def load_model_and_tokenizer():
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=3) # Re-initialize structure
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # Load fine-tuned weights
    model.to(device)
    model.eval() # Set to evaluation mode
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Model and tokenizer loaded successfully!")

class NewsItem(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence_score: float # Well, more like softmax probability of the predicted class

# Map numerical labels back to human-readable sentiment
label_map = {0: "negative", 1: "neutral", 2: "positive"} # This mapping depends on how I set it up during training

@app.post("/analyze_sentiment", response_model=SentimentResponse)
async def analyze_sentiment_endpoint(item: NewsItem):
    if not model or not tokenizer:
        # This shouldn't happen if startup event worked
        return {"error": "Model not loaded"}

    inputs = tokenizer(item.text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad(): # IMPORTANT: disable gradient calculations for inference
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()
    predicted_class_id = np.argmax(probs)
    predicted_sentiment = label_map.get(predicted_class_id, "unknown") # Use .get for safety
    confidence = float(probs[predicted_class_id])

    return SentimentResponse(sentiment=predicted_sentiment, confidence_score=confidence)

# To run this: uvicorn main:app --reload (after installing uvicorn and fastapi)
```

Getting the `load_model_and_tokenizer` function with `@app.on_event("startup")` right was key. I initially had the model loading inside the endpoint function, and performance was terrible. A quick search on FastAPI best practices for ML models pointed me towards the startup event pattern.

Testing with `curl` and then exploring the auto-generated `/docs` page from FastAPI was incredibly satisfying. Sending a JSON payload like `{"text": "The company announced record profits and expansion plans."}` and getting back `{"sentiment": "positive", "confidence_score": 0.9...}` felt like magic, even though I knew exactly what code was running underneath.

For now, the API runs locally. Deploying it properly using Docker and perhaps a small cloud VM (like an AWS EC2 t2.micro or Google Cloud e2-small instance, though GPU instances for inference are pricier) is a potential next step if I want to make it more broadly accessible.

### Reflections and What's Next

This project was a significant learning experience. The sheer amount of time spent on data preparation and debugging, compared to the "glamorous" model training part, was a real eye-opener. It reinforced that ML is an iterative process – tweak, run, analyze, repeat.

The biggest challenges were:
1.  **Data Sourcing & Cleaning:** Finding a good, clean, labeled dataset is half the battle.
2.  **Computational Resources:** Fine-tuning BERT, even `bert-base-uncased`, requires a decent GPU. Colab was essential, despite its quirks.
3.  **Debugging Model Inputs/Outputs:** The dimensions and types of tensors have to be *exact*, and error messages aren't always clear.
4.  **Bridging Model to API:** Making the trained model actually *usable* via an API involved a different set of skills and considerations (like model loading and thread safety, though I didn't deeply explore the latter).

Key learnings for me include a much deeper understanding of the Hugging Face Transformers library, practical experience with PyTorch, and the basics of building a functional API with FastAPI. I also learned the importance of meticulous experiment tracking (even if it was just a messy text file noting hyperparameters and results for different runs).

There are many ways this project could be extended:
*   **Better Base Model:** Try fine-tuning a model pre-trained specifically on financial text (like FinBERT) to see if it improves performance or requires less data.
*   **Larger/More Diverse Dataset:** Incorporate more varied financial news sources.
*   **Aspect-Based Sentiment:** Instead of overall sentiment, try to identify sentiment towards specific entities or aspects (e.g., "AAPL positive, but market negative"). This is much more complex.
*   **Simple UI:** Build a very basic Streamlit or Flask frontend to interact with the API more easily than `curl`.

Overall, I'm quite proud of getting this from an idea to a working prototype. It was a challenging but rewarding journey into applied NLP.