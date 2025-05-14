---
layout: post
title: News Sentiment Financial Forecaster
---

## From News Headlines to Market Whispers: Building a Financial Sentiment Forecaster

It’s been a while since my last post, mostly because this latest project ended up being a much deeper rabbit hole than I initially anticipated. I’m excited to finally share some details about the "News Sentiment Financial Forecaster" – a system I’ve been piecing together to see if I could meaningfully connect the sentiment in financial news to stock price movements. The core idea was to fine-tune a language model, specifically Mistral-7B, on financial news to get a more nuanced sentiment score than your typical positive/negative/neutral, then deploy it and try to correlate its output with actual market data using Polars.

### The Spark: Beyond Simple Sentiment

The initial motivation came from a frustration with existing sentiment analysis tools when applied to finance. Often, they feel too generic. A headline like "Company X sees unexpected Q3 profit dip" might be flagged as negative, but the *degree* and *type* of negativity (e.g., mild concern vs. outright panic) can be lost. I wanted to see if a more sophisticated model, fine-tuned on domain-specific data, could capture these subtleties. My hypothesis was that a more nuanced sentiment signal could potentially offer a slightly better leading indicator for price changes than a coarse one.

### Choosing the Brain: Why Mistral-7B?

When it came to selecting a base model, the field is pretty crowded these days. I looked into a few options. Larger models were tempting for their power, but the computational resources (and associated costs for a student budget!) for fine-tuning and inference were a major concern. I’d read some very promising things about Mistral-7B – its performance seemed to punch well above its weight class, offering a good balance between capability and manageability. Its Apache 2.0 license was also a plus. I briefly considered some BERT-style models, but the buzz around the newer architectures and their alleged efficiency in fine-tuning pushed me towards the Mistral family.

The first step was getting comfortable with the Hugging Face `transformers` library. I'd used it before for some classification tasks, but fine-tuning a model of this scale was new territory.

### The Fine-Tuning Saga: Data, Sweat, and GPUs

This was, without a doubt, the most challenging part.

**1. Sourcing and Preparing Data:**
I needed a dataset of financial news articles paired with sentiment labels. Finding a publicly available, high-quality, and *large* dataset specifically for *nuanced* financial sentiment was harder than I thought. Most were just positive/negative. I ended up having to combine a few sources. I pulled a lot of headlines and short summaries from various financial news APIs (some free-tier, some I scraped responsibly over weeks). For labeling, I initially thought about manually labeling a few thousand articles to capture the nuance I wanted (e.g., scores from -1.0 to 1.0, or categories like 'Cautiously Optimistic', 'Strong Sell Signal Fear'). This quickly proved to be an overwhelming task.

Eventually, I found a smaller, pre-labeled financial phrase bank (like the SemEval ones, but older) that had some degree of nuance. It wasn't perfect, but it was a starting point. I then tried a semi-supervised approach: I fine-tuned a smaller, faster model on this limited dataset and used it to predict sentiment on my larger corpus of unlabeled news. I manually reviewed a subset of these predictions, corrected the obvious errors, and added this "pseudo-labeled" data back into my training set. It was a bit of a hack, and I’m still not entirely sure about the biases it might have introduced, but it was the only feasible way I could scale up my training data with my limited resources.

My data preparation script involved a lot of cleaning – removing boilerplate text, standardizing date formats, and tokenizing the news snippets.

```python
# A snippet of how I was trying to structure the input for tokenization
# news_items is a list of dictionaries, each with 'text' and 'sentiment_score'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# Needed to add a padding token for Mistral
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def preprocess_function(examples):
    # The 'text' field in my dataset
    inputs = [item for item in examples["text"]]
    # The 'sentiment_score' field, which I was trying to map to a continuous value or specific classes
    # This part went through many iterations.
    # For a regression-style approach, targets would be float. For classification, integer class labels.
    # Let's assume I was aiming for a regression target here for nuanced scores.
    targets = [float(score) for score in examples["sentiment_score"]]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = targets
    return model_inputs

# Then used with Hugging Face datasets:
# tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)```

One specific headache was deciding on the output format. Should sentiment be a continuous score (e.g., -1.0 to 1.0) or categorical (e.g., 'very negative', 'neutral', 'very positive')? I leaned towards a continuous score for nuance but found that training for regression directly with LLMs can sometimes be trickier than classification. I experimented with discretizing the sentiment into, say, 10 bins and treating it as a classification problem, then converting back to an average score. This seemed to stabilize training a bit.

**2. The Fine-Tuning Process:**
I used the Hugging Face `Trainer` API. Setting up the `TrainingArguments` took a lot of trial and error. Learning rates, batch sizes, gradient accumulation steps – I spent days tweaking these. My first few attempts were disastrous: loss wouldn't decrease, or I'd get NaN losses. I remember one late night staring at a flat loss curve, thinking I’d completely misunderstood something fundamental. A StackOverflow thread eventually pointed me towards issues with learning rates being too high for LoRA (Low-Rank Adaptation), which I decided to use to make fine-tuning feasible on the single GPU I had access to (an NVIDIA RTX 3070 in my personal machine, bless its circuits).

```python
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model # For LoRA

# This is a simplified representation. My actual config had more parameters.
lora_config = LoraConfig(
    r=16, # Rank of the update matrices.
    lora_alpha=32, # Alpha parameter for LoRA scaling.
    target_modules=["q_proj", "v_proj"], # Modules to apply LoRA to. Specific to Mistral.
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS" # Or "REGRESSION" if I went that route.
)

# model was my AutoModelForSequenceClassification loaded with Mistral-7B weights
# model.config.pad_token_id = tokenizer.pad_token_id # ensure pad token id is set

# Get the PEFT model
# model = get_peft_model(model, lora_config) # model was already loaded, this wraps it.

# After loading the base model and tokenizer:
# model.config.pad_token_id = tokenizer.pad_token_id

# Then applying LoRA (assuming 'model' is the base Mistral model)
# peft_model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./results_financial_mistral",
    num_train_epochs=3, # This was an early parameter, later adjusted based on eval loss
    per_device_train_batch_size=1, # Constrained by my GPU memory
    gradient_accumulation_steps=8, # Effective batch size of 8
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5, # Tweaked this a lot
    weight_decay=0.01,
    load_best_model_at_end=True,
    # fp16=True, # Mixed precision training, helped with memory and speed
)

# trainer = Trainer(
#     model=peft_model, # Use the PEFT model
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"], # I had a validation split
#     # compute_metrics=my_compute_metrics_function, # Wrote a custom one for nuanced scores
# )

# And then...
# trainer.train()
```
The `target_modules` for LoRA specifically for Mistral-7B wasn't immediately obvious. I had to dig into some forum discussions and examples specific to Mistral architecture on the Hugging Face forums to find which modules (`q_proj`, `v_proj`, etc.) were best to target. Using `fp16` (mixed-precision training) was a lifesaver for memory, though I had to make sure my GPU drivers and CUDA versions were all happy together. That itself was an afternoon of debugging.

A breakthrough moment was when I started seeing the evaluation loss consistently decrease and the custom metrics I implemented (which tried to measure how far off my predicted sentiment score was from the target) began to improve. It wasn't perfect, but it was learning!

### Deployment: Taking it to the Cloud with Vertex AI

Once I had a fine-tuned model I was reasonably happy with, the next step was deploying it so I could actually use it to get sentiment scores for new articles. Running it locally was fine for testing, but not scalable.

I chose Google Cloud's Vertex AI. I'd used some GCP services before for other courses, so I had some familiarity with the ecosystem. AWS SageMaker was another option, but I found the Vertex AI documentation for custom container deployment a bit more straightforward for what I was trying to do at the time. The ability to deploy custom containers with a pre-built PyTorch serving image was attractive.

The process involved:
1.  **Pushing my fine-tuned model artifacts** (the LoRA adapter weights and the base model config) to a Google Cloud Storage bucket.
2.  **Writing a custom prediction handler script** (`predictor.py`). This script would load the base Mistral-7B model, then load and apply the LoRA adapter weights on top of it. It also needed a `predict` method that Vertex AI Endpoints could call.

    ```python
    # A conceptual snippet from my predictor.py for Vertex AI
    # This is NOT the full file, just showing the idea.
    import os
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel # To load LoRA adapters
    
    class FinancialSentimentPredictor:
        def __init__(self):
            self.model = None
            self.tokenizer = None
            self._load_model()
    
        def _load_model(self):
            model_name = "mistralai/Mistral-7B-v0.1" # Base model
            # Adapter path would be something like os.environ.get("AIP_STORAGE_URI")
            # which Vertex AI sets, pointing to where my LoRA weights are in GCS.
            adapter_path = os.environ.get("AIP_STORAGE_URI", "/app/adapter") # Fallback for local test
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            # Load the base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=10, # Or 1 if regression, depends on final choice
                # torch_dtype=torch.bfloat16, # For inference on compatible hardware
            )
            base_model.config.pad_token_id = self.tokenizer.pad_token_id

            # Load the LoRA adapter
            # This assumes the adapter weights are in a subdirectory pointed to by adapter_path
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model.eval() # Set to evaluation mode
    
        def predict(self, instances, **kwargs):
            # Instances would be a list of texts
            predictions = []
            for text_input in instances:
                inputs = self.tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512, padding=True)
                # inputs = {k: v.to(self.model.device) for k, v in inputs.items()} # Move to device if GPU
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Process logits to get sentiment score
                    # This depends heavily on whether it's classification or regression
                    # For example, if classification (10 bins):
                    # score = torch.argmax(outputs.logits, dim=-1).item() 
                    # This would need to be mapped back to a sentiment value
                    # For now, just a placeholder for the actual logic
                    processed_score = outputs.logits.mean().item() # Highly simplified example
                    predictions.append(processed_score) 
            return predictions

    # This is just a conceptual illustration. The actual predictor script had more error handling
    # and logic to convert model outputs to the desired sentiment score format.
    ```

3.  **Creating a Dockerfile** that specified my base image (a PyTorch image), copied my `predictor.py` and `requirements.txt`, and set up the entry point for the Vertex AI prediction service.
4.  **Building the Docker image and pushing it to Google Artifact Registry.**
5.  **Creating a Vertex AI Endpoint and deploying my model to it.** This involved selecting machine types. I initially picked a machine with a GPU (like an NVIDIA T4) to speed up inference, but for cost reasons during development, I also tested with CPU-only instances which were slower but cheaper for intermittent use. Understanding the `AIP_STORAGE_URI` environment variable and how it points to your model artifacts in GCS was key here. I found a GCP tutorial on deploying custom PyTorch models very helpful, though I had to adapt it for Hugging Face `transformers` and `peft`.

One struggle was getting the LoRA adapter to load correctly within the Vertex AI environment. The paths and environment variables needed to be just right. Lots of checking logs in Cloud Logging to debug why the model server wasn't starting or predictions were failing.

### Correlation: News Sentiment Meets Market Prices with Polars

With the model deployed and an endpoint ready to serve sentiment scores, the final piece was to ingest new financial news, get sentiment scores, and then try to correlate these scores with stock price movements.

For handling the data (news timestamps, sentiment scores, stock prices), I decided to use Polars. I'd been hearing a lot about its performance benefits over Pandas for larger datasets, especially its ability to leverage multi-core processing and its more memory-efficient query engine. Given that I was planning to look at historical data and potentially high-frequency news, this seemed like a good choice to learn.

**1. Getting Price Data:**
I used the `yfinance` library to download historical stock price data for a selection of tickers I was interested in.

**2. The Workflow:**
My script would:
    a. Fetch recent news headlines for the selected tickers.
    b. Send these headlines to my Vertex AI endpoint to get sentiment scores.
    c. Store the news item, its timestamp, and its sentiment score.
    d. Combine this sentiment data with historical price data, aligning them by time. This was tricky – news doesn't always perfectly align with market open/close or specific candle intervals. I had to decide on a window (e.g., sentiment from news in the last X hours before market open).

**3. Polars in Action:**
Polars really shone when it came to merging and aggregating the data. For example, I might have multiple news items for a stock in a day, so I'd want to calculate an average or a weighted average sentiment score for that day.

```python
import polars as pl
from datetime import datetime, timedelta

# Assume news_data is a list of dicts: {'timestamp': datetime, 'ticker': str, 'sentiment': float}
# And price_data is a list of dicts: {'timestamp': datetime, 'ticker': str, 'close_price': float}

# Convert to Polars DataFrames
df_news = pl.DataFrame(news_data)
df_prices = pl.DataFrame(price_data)

# Ensure timestamps are proper datetime objects if not already
df_news = df_news.with_columns(pl.col("timestamp").str.to_datetime().alias("datetime"))
df_prices = df_prices.with_columns(pl.col("timestamp").str.to_datetime().alias("datetime"))

# Example: Aggregate news sentiment daily
df_daily_sentiment = (
    df_news
    .group_by_stable(pl.col("datetime").dt.truncate("1d"), "ticker") # Group by day and ticker
    .agg(pl.col("sentiment").mean().alias("avg_sentiment"))
    .rename({"datetime": "date"}) # Renaming for clarity
)

# Example: Calculate daily price changes
df_daily_prices = (
    df_prices
    .sort("ticker", "datetime")
    .group_by_stable("ticker", pl.col("datetime").dt.truncate("1d"))
    .agg(pl.col("close_price").last().alias("close")) # Get last closing price for the day
    .with_columns(
        pl.col("close").pct_change().over("ticker").alias("price_change")
    )
    .rename({"datetime": "date"})
)

# Join sentiment with price changes
# This requires careful handling of dates and potential lookaheads/lags for sentiment
# For instance, does today's sentiment predict tomorrow's price change?

# A simple join on date and ticker:
df_combined = df_daily_sentiment.join(
    df_daily_prices, on=["date", "ticker"], how="inner"
)

# Then, one might lag sentiment to see if it predicts next day's price_change
# df_combined = df_combined.with_columns(
#    pl.col("avg_sentiment").shift(1).over("ticker").alias("prev_day_sentiment")
# )

# And finally, calculate correlation
# This is a very basic correlation. Real analysis would be more involved.
# correlation = df_combined.select(
#    pl.corr("prev_day_sentiment", "price_change")
# ).item()

# print(f"Correlation between previous day sentiment and price change: {correlation}")
```
I’m still figuring out the best way to do the temporal alignment and calculate meaningful correlations. Do I look at sentiment leading price changes by a few hours, a day? Do I differentiate between pre-market news and news during trading hours? Polars made the data manipulation for these experiments much faster than I think Pandas would have been, especially as my datasets grew. The expression-based API took a little getting used to after years of Pandas, but I started to appreciate its power and clarity for complex chained operations. I often found myself consulting the Polars documentation and examples on their website.

### Initial Results and Lingering Questions

So, did it work? The answer is… "it's complicated."

The fine-tuned Mistral-7B model definitely produces more nuanced sentiment scores than the off-the-shelf tools I tested. For instance, it seems better at distinguishing between a mildly cautious statement and a strongly negative one. When I manually reviewed its outputs on new articles, they often *felt* more aligned with my own interpretation of the financial context.

However, translating these nuanced scores into a consistently predictive financial signal is another beast entirely. My initial correlation analyses (like the simplified one in the code snippet) showed some weak, occasionally positive correlations between strong sentiment shifts and subsequent price movements for certain stocks, particularly around major earnings announcements or unexpected news events. But it's far from a crystal ball. There's so much noise in the market, and sentiment is just one tiny piece of the puzzle.

The biggest challenge remains rigorously validating the predictive power. Financial markets are notoriously difficult to predict, and spurious correlations are easy to find. I need to be much more careful about backtesting methodologies, considering transaction costs if this were to be used for trading (which is NOT my current goal – this is purely a research project), and avoiding lookahead bias.

### What I Learned and What's Next

This project has been an incredible learning experience:
*   **LLM Fine-tuning:** I have a much better grasp of the practicalities involved, from data prep to navigating the Hugging Face ecosystem and the nuances of LoRA.
*   **Vertex AI Deployment:** I successfully took a complex model from my local machine to a scalable cloud endpoint. Debugging IAM permissions and Docker configurations in the cloud was a valuable, if sometimes frustrating, lesson.
*   **Polars:** I’m a convert. For data analysis tasks of this nature, the performance and expressive API are big wins.
*   **The Difficulty of Financial Prediction:** This project reinforced just how challenging it is to find real alpha in financial markets.

**Future ideas (if I ever find the time!):**
*   **Better Data:** Continue to improve the quality and quantity of the fine-tuning data. Perhaps explore more sophisticated active learning or data augmentation techniques.
*   **More Sophisticated Correlation Analysis:** Move beyond simple correlations to time-series models (like VAR or Granger causality tests) to understand the lead-lag relationships more deeply.
*   **Multi-modal Analysis:** Incorporate other data sources, perhaps trading volumes or even sentiment from social media, though that introduces its own set of complexities.
*   **Cost Optimization on Vertex AI:** Experiment more with different machine types and possibly serverless inference options if they become more suitable for this kind of model.

This was a long post, but it was a long project! It’s still very much a work in progress, but I’m proud of how far it’s come. It pushed my coding skills, my understanding of machine learning, and my patience to new limits. If anyone has experience with similar projects or suggestions, I’d love to hear them in the comments!