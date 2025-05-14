---
layout: post
title: LLM-Powered Sentiment Arbitrage
---

## Project Log: LLM-Powered Sentiment Arbitrage - An Attempt

This has been a consuming project for the past few months, an attempt to see if I could leverage some of the recent advancements in NLP for something as notoriously difficult as trading. The core idea was to fine-tune a language model on financial news to gauge sentiment and then build a Python framework to test if this sentiment could actually inform trading strategies. The "arbitrage" part of the name is probably a bit ambitious for what I've managed so far, but it was the guiding principle.

### Phase 1: Getting a Grip on Sentiment – Fine-Tuning BERT

The first major hurdle was sentiment analysis. Generic sentiment analyzers wouldn't cut it for financial news, which has its own jargon and context. A headline like "XYZ Corp slashes profit outlook, shares surge on bold restructuring plan" is a nightmare for standard models. I decided I needed to fine-tune a transformer model.

After some reading, I landed on BERT, specifically `bert-base-uncased` from Hugging Face. RoBERTa was a consideration, and some smaller distilled models too, mainly because I was worried about training times on my somewhat limited local setup (an older NVIDIA card that has seen better days). But BERT seemed to have the most community support and available tutorials for fine-tuning, which felt like a safer bet for a first attempt at a project of this scale.

For data, I initially looked at the Financial PhraseBank dataset, which is often cited. It's decent for classification of sentiment in short financial phrases. However, I wanted to work with full news articles, or at least substantial snippets. This led me down a rabbit hole of trying to scrape news. I eventually settled on a combination: using PhraseBank for some initial supervised fine-tuning for phrase-level sentiment, and then attempting to gather a small, custom-labeled dataset of headlines and short summaries for more document-level understanding. Labeling was tedious and undoubtedly a source of noise. I tried to stick to a three-class system: positive, negative, neutral.

The fine-tuning process itself was an adventure with the Hugging Face `Trainer` API. Setting it up wasn't too bad, but the first few runs were demoralizing.
```python
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset # Used this to make my life easier with the datasets

# --- (Assume train_texts, train_labels, val_texts, val_labels are loaded and preprocessed) ---
# train_texts and val_texts are lists of strings (news snippets)
# train_labels and val_labels are lists of integers (0 for neg, 1 for neutral, 2 for pos)

MODEL_NAME = 'bert-base-uncased'

class FinancialNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128): # initially tried 256, but OOM errors were frequent
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length', # This was a key! Before, ragged tensors were a nightmare.
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
# Had to ensure my custom dataset was correctly formatted here
# train_dataset = FinancialNewsDataset(train_texts, train_labels, tokenizer)
# val_dataset = FinancialNewsDataset(val_texts, val_labels, tokenizer)

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3) # 3 labels: neg, neu, pos

# These training arguments are mostly defaults, I tweaked batch_size and epochs mostly.
# Learning rate was a big one, too low and it took forever, too high and it diverged.
# Found a forum post on Hugging Face discussing good starting LRs for BERT fine-tuning, that helped.
training_args = TrainingArguments(
    output_dir='./results/sentiment_bert',
    num_train_epochs=3, # Started with 1, then 3. More didn't seem to help much with my small dataset.
    per_device_train_batch_size=4, # Had to reduce this from 8 due to memory constraints.
    per_device_eval_batch_size=8,
    warmup_steps=100, # Read that this helps stabilize training early on.
    weight_decay=0.01,
    logging_dir='./logs/sentiment_bert_logs',
    logging_steps=50,
    evaluation_strategy="epoch", # Evaluate at the end of each epoch.
    save_strategy="epoch", # Save model checkpoint at the end of each epoch.
    load_best_model_at_end=True # Important to get the best performing model.
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset, # This should be an instance of FinancialNewsDataset
#     eval_dataset=val_dataset   # This too
# )

# And then the magic (or sometimes, tragic) command:
# trainer.train()
```
My first issue was batch size and sequence length. `max_length=256` with a batch size of 8 consistently gave me CUDA out-of-memory errors. I had to dial `max_length` down to 128, and `per_device_train_batch_size` to 4. This obviously impacts the amount of context the model sees, but it was a practical trade-off. I also spent a good while figuring out the `padding` and `truncation` arguments to `tokenizer.encode_plus`. Getting differently sized inputs to batch together correctly was tricky until I found a clear example in the Hugging Face documentation about `padding='max_length'`.

Evaluation was initially just accuracy, but financial news is imbalanced – lots of neutral news. So, I started looking at F1 scores per class, which gave a much better picture. The model got pretty good at identifying clearly positive or negative statements from PhraseBank, but struggled more with the nuance in my custom-scraped headlines. It often confused neutral statements with slight positive/negative tilts, or got thrown off by complex sentences. This remains an area for improvement.

### Phase 2: Building the Python Trading Strategy Framework

With a somewhat working sentiment model (let's call it `sentiment_bert_v1`), the next step was to build a framework to actually use it. This meant a Python application that could:
1.  Fetch financial news.
2.  Run sentiment analysis on it.
3.  Execute trading strategies (hypothetically, for now).
4.  Backtest these strategies.

For news fetching, I looked into a few APIs. Many of the good, real-time ones are quite expensive. I ended up using a free tier of NewsAPI for a while, but the article limits were restrictive for thorough backtesting. For some of the project, I resorted to using pre-downloaded datasets of historical news when API limits became too much of a bottleneck, though this isn't ideal for simulating live trading.

The core of the framework started to take shape with a few classes:
```python
# This is a simplified representation, the actual code got a bit more tangled.
import requests # For fetching news from an API
# from transformers import pipeline # Could use pipeline for easier inference after fine-tuning

class NewsFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key # Store API key if needed for a live feed
        # self.news_api_url = "https://newsapi.org/v2/everything" # Example

    def get_news_for_ticker(self, ticker, date_from, date_to):
        # In reality, this involved handling API pagination, error codes, etc.
        # params = {'q': ticker, 'from': date_from, 'to': date_to, 'apiKey': self.api_key, 'language': 'en'}
        # response = requests.get(self.news_api_url, params=params)
        # articles_data = response.json().get('articles', [])
        # For now, returning dummy data structure
        print(f"Fetching news for {ticker} from {date_from} to {date_to}")
        # This would actually parse and return a list of article texts or structured data
        return [{"headline": f"News about {ticker} 1", "content": "Positive outlook."}, 
                {"headline": f"More on {ticker}", "content": "Mixed signals observed."}] 

class SentimentPredictor:
    def __init__(self, model_path):
        # self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
        # For direct model usage:
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        print(f"Sentiment model loaded from {model_path}")

    def get_sentiment_score(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        # Make sure model is in eval mode if not using pipeline
        # self.model.eval() 
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        # Example: score = positive_prob - negative_prob. My labels were 0:neg, 1:neu, 2:pos
        # This mapping was critical and took a few tries to get right.
        score = probabilities.item() - probabilities.item() # Pos_prob - Neg_prob
        return score

class BasicTradingStrategy:
    def __init__(self, sentiment_predictor, news_fetcher, paper_trader_api, sentiment_threshold=0.3):
        self.predictor = sentiment_predictor
        self.fetcher = news_fetcher
        self.broker = paper_trader_api # This would be an interface to a paper trading account
        self.threshold = sentiment_threshold
        self.portfolio = {} # ticker: quantity

    def process_daily_news(self, ticker, trade_date):
        # Fetch news for the previous day to make decisions for 'trade_date'
        # This is important to avoid look-ahead bias
        news_items = self.fetcher.get_news_for_ticker(ticker, "yesterday_start", "yesterday_end") # Simplified dates
        
        if not news_items:
            return

        aggregated_sentiment = 0
        num_articles = 0
        for item in news_items:
            # Combining headline and content, or just using headline, was a decision point
            text_to_analyze = item['headline'] # + " " + item.get('content', '') 
            score = self.predictor.get_sentiment_score(text_to_analyze)
            aggregated_sentiment += score
            num_articles +=1
        
        if num_articles == 0:
            return

        average_sentiment = aggregated_sentiment / num_articles
        print(f"Ticker: {ticker}, Avg Sentiment: {average_sentiment:.2f}")

        # Very naive strategy logic here
        current_shares = self.portfolio.get(ticker, 0)
        if average_sentiment > self.threshold and current_shares == 0:
            print(f"Strategy: BUY {ticker} due to positive sentiment.")
            # self.broker.submit_order(ticker, 100, "buy") # e.g. buy 100 shares
            self.portfolio[ticker] = 100 
        elif average_sentiment < -self.threshold and current_shares > 0:
            print(f"Strategy: SELL {ticker} due to negative sentiment.")
            # self.broker.submit_order(ticker, current_shares, "sell")
            self.portfolio[ticker] = 0
        # else:
            # print(f"Strategy: HOLD {ticker}")
```
For the trading execution, I initially planned to integrate with Alpaca's paper trading API. I got as far as basic authentication and submitting orders, but synchronizing it perfectly with a backtesting framework turned out to be more complex than anticipated, especially around order fill simulation. So, for most of the strategy testing, I ended up building a simpler, custom backtester.

The backtester was a major sub-project. It needed to ingest historical price data (I used `yfinance` for this initially) and simulate trades based on the sentiment signals. A huge challenge here was avoiding lookahead bias. For instance, making sure that a decision on day D is only based on news and sentiment available *before* market open on day D, and using closing prices of day D to transact. I also had to decide how to model transaction costs and potential slippage – I started with very simple assumptions (e.g., a fixed percentage cost per trade) as a first pass.

My first few backtest runs were… not great. The initial strategies were too simplistic. For example, "buy if average sentiment > 0.5, sell if < -0.5". This often resulted in over-trading on noisy sentiment scores. I also realized that raw sentiment might not be enough; the *change* in sentiment, or sentiment *momentum*, could be more informative. This led to more complex feature engineering, which is still ongoing.

One particular "aha!" moment came when I was debugging a strategy that performed surprisingly well. It turned out I had a subtle lookahead bias where news from *within* the trading day was influencing decisions for that same day. Untangling that took a lot of print statements and stepping through data day-by-day. A StackOverflow answer regarding time-series data splitting for financial applications helped me structure my data handling more rigorously.

### Challenges and Learnings

This project has been a steep learning curve.
1.  **Nuance of Financial Language:** Even with fine-tuning, BERT struggled with sarcasm, forward-looking statements vs. current facts, and the market's often counter-intuitive reactions to news. "Shares fall despite strong earnings" is a classic example.
2.  **Data Quality and Availability:** Good, clean, labeled financial news data for fine-tuning is hard to come by without paying significant sums. My custom dataset was small and likely had inconsistencies.
3.  **Backtesting Pitfalls:** It's incredibly easy to build a backtester that looks good on paper but wouldn't work in reality. Slippage, realistic execution prices, and robust handling of corporate actions (splits, dividends) are all things I only scratched the surface of.
4.  **Computational Resources:** Fine-tuning even `bert-base-uncased` requires a decent GPU and patience. Scaling this up to larger models or more data would require more serious hardware or cloud credits, which are constraints for a student project.
5.  **From Signal to Profit:** Even if sentiment is perfectly captured, translating that into a consistently profitable trading strategy is another leap entirely. Market efficiency, timing, risk management – these are all massive domains.

Despite the challenges, the learning has been immense. I've dived deep into Hugging Face Transformers, PyTorch, API integration, and the basics of quantitative strategy development. The struggle with aligning news timestamps with market data for backtesting was particularly formative. I also learned that a "good" sentiment score doesn't always mean the stock will go up; the market is a complex beast.

### Future Work and Reflections

The "arbitrage" part of the project name remains largely aspirational. True arbitrage is risk-free, and this is anything but.
If I were to continue this, I'd focus on:
*   **Better Data:** Exploring more sophisticated datasets or news sources, perhaps focusing on specific sectors where news sentiment might have a clearer impact.
*   **More Robust Sentiment Model:** Experimenting with larger models (like RoBERTa or domain-specific financial LLMs if I could access them), more data, or multi-task learning (e.g., predicting sentiment and numerical impact simultaneously).
*   **Advanced Strategy Logic:** Moving beyond simple thresholds to incorporate sentiment dynamics, volume, or other market factors. Perhaps even some reinforcement learning, though that’s another mountain to climb.
*   **Rigorous Backtesting:** Integrating a more professional backtesting library or improving my custom one to better model real-world trading conditions.

This project was a lesson in humility when facing complex systems like financial markets. It also showed me the power of modern LLMs, even if harnessing that power effectively is a significant engineering and research challenge. It was less about striking gold and more about the process of building, testing, failing, and learning. And honestly, getting that BERT fine-tuning loop to finally converge felt like a pretty big win on its own.