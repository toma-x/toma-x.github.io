---
layout: post
title: AI-Powered Financial News Analyzer
---

## AI-Powered Financial News Analyzer: My Journey into Sentiment, Stocks, and (a lot of) Python

This one was a marathon, not a sprint. For my latest personal project, I decided to tackle something that's been milling around in my head for a while: trying to see if the sentiment in financial news could actually give a hint about short-term stock price movements. The idea isn't new, but actually building a system to test it, from data scraping to deploying a model on Vertex AI, felt like a mountain to climb. Spoiler: it was, but the view from (near) the top is pretty good.

**The Initial Spark: Can News Predict Prices?**

It started after a particularly volatile week in the market. I was reading through a ton of news articles, some overwhelmingly positive, others predicting doom, and I wondered how much of this was just noise versus actual predictive signal. Could an AI model, specifically a sentiment analyzer, cut through the subjective interpretations and find a quantifiable link to price shifts? The goal became to build an "AI-Powered Financial News Analyzer" that would stream financial news, analyze sentiment using a model I'd train, and then try to correlate that sentiment with short-term asset price changes. The target I set for myself – perhaps a bit ambitiously – was to see if I could get anywhere near an 80-85% accuracy in this correlation.

**Choosing the Tools: Vertex AI and Python**

Python was a no-brainer. It's what I'm most comfortable with, and the ecosystem for data science (Pandas, NumPy, scikit-learn) and interacting with cloud services is fantastic. The bigger decision was *where* to build and train the model. My laptop isn't exactly a supercomputer, and I knew training a sentiment model, especially if I went down the custom route, would require some horsepower.

I'd touched on Google Cloud's Vertex AI in a class last semester, mostly using some of its pre-built APIs. The idea of having a unified MLOps platform was appealing, and honestly, the free credits I still had were a big motivator. I figured this project would be a good way to get deeper into it. I briefly considered just using a pre-trained sentiment model locally with something like Hugging Face Transformers, but I was concerned about two things:
1.  **Financial Nuance:** General-purpose sentiment models might not understand the specific jargon of finance. "Bearish" means something very different in a financial context than in a zookeeping one.
2.  **Scalability & Deployment:** Even if I got something working locally, I wanted the experience of deploying a model on a proper platform. Vertex AI seemed like the right place for that.

**The Data Labyrinth: Getting and Cleaning News**

This. This was the first major hurdle. Finding a consistent, reliable, and *affordable* stream of financial news is tough. I looked into several news APIs:
*   NewsAPI.org: Good, but the free tier is limited for the kind of historical depth and volume I wanted.
*   Alpha Vantage: More finance-focused, but again, API call limits were a concern for continuous analysis.
*   Some paid options were just way out of budget for a personal project.

I ended up cobbling together a solution using a mix of RSS feeds from major financial news outlets and, I’ll admit, some carefully crafted web scraping scripts (using BeautifulSoup and Requests in Python) for specific sites that didn't have accessible feeds. This was a constant battle – website structures change, and scrapers break. I spent more evenings than I'd like to admit tweaking XPath selectors and CSS selectors.

Then came cleaning. Oh, the cleaning. HTML tags, JavaScript snippets, "Related Articles" sections, cookie consent pop-ups embedded in the scraped text – it was a mess.

```python
# A snippet of my early, painful text cleaning efforts
import re
from bs4 import BeautifulSoup

def clean_raw_html_content(html_content):
    # Initial pass with BeautifulSoup to remove script/style, get text
    soup = BeautifulSoup(html_content, "lxml")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text()

    # Remove extra whitespace - this was a bigger pain than expected
    text = re.sub(r'\s+', ' ', text).strip()
    # Trying to remove leftover junk, advertisement phrases, etc.
    # This list grew very, very long...
    junk_phrases = ["Read More:", "Subscribe to our newsletter", "Copyright", "Related stories"]
    for phrase in junk_phrases:
        text = text.replace(phrase, "")
    
    # Still had issues with unicode characters sometimes, had to handle those later.
    # LOGGER.debug(f"Cleaned text length: {len(text)}")
    return text

# ... later in the data pipeline ...
# raw_article_data = load_scraped_data_from_json("daily_news_dump_2024_03_15.json") 
# for article_id, content in raw_article_data.items():
# processed_text = clean_raw_html_content(content['htmlBody'])
# if len(processed_text) < 100: # Arbitrary threshold to discard tiny, useless articles
# continue 
# store_cleaned_text(article_id, processed_text)
```
The comment `# This list grew very, very long...` is an understatement. It felt like every day I found a new common phrase from an ad or a disclaimer that was skewing my data.

Linking news to specific assets and their prices was another adventure. I focused on major stock tickers. The key was the timestamp of the news article. I had to fetch historical stock price data (Yahoo Finance API via `yfinance` library in Python was a lifesaver here) and align the news publication time with market data. Time zones were, predictably, a nightmare. Was the news published before market open, during trading hours, or after close? This mattered for defining the "short-term price change" window. I settled on looking at price changes 1 hour, 4 hours, and 24 hours after a news piece was published, trying to capture immediate reactions and slightly more delayed effects.

**Into the Vertex AI Jungle: Sentiment Modeling**

With a (somewhat) clean dataset, I turned to Vertex AI. My initial plan was to try Vertex AI AutoML for text classification (sentiment analysis). The promise of just uploading data and letting Google's magic find the best model was tempting.

**Attempt 1: AutoML and Financial Jargon**
I meticulously prepared a dataset of a few thousand news headlines and short summaries, manually labeled as 'positive', 'negative', or 'neutral' *from a financial perspective*. This labeling process was incredibly time-consuming. What one person considers neutral, another might see as slightly negative for a specific stock. I tried to be as consistent as possible.

I uploaded my CSV to Vertex AI, configured an AutoML text classification job, and waited. The first results were… okay. Around 65-70% accuracy on my validation set. Better than random guessing, but not groundbreaking. Looking at the errors, it became clear: general sentiment wasn't cutting it. For example, a headline like "Company X sees massive Q3 revenue surge but warns of increased Q4 R&D spending" might be flagged as positive by a generic model focusing on "massive revenue surge." But for an investor, the "warns of increased R&D spending" might be a neutral or even slightly negative signal for short-term profitability. The model needed more financial context.

**Attempt 2: Custom Training – A Deeper Dive (and More Pain)**
I realized I needed more control. While AutoML is powerful, my dataset and the nuanced nature of financial sentiment seemed to demand a more tailored approach. I didn't want to build a transformer from scratch, but I explored options for fine-tuning pre-trained models using Vertex AI Custom Training.

This is where things got *really* complicated. I decided to use a pre-trained BERT model (specifically, `bert-base-uncased` as a starting point, because it was well-documented and I found some tutorials) and fine-tune it on my labeled financial news dataset. Vertex AI allows you to submit custom training jobs using Docker containers. This meant I had to:

1.  **Write a Python training script** using TensorFlow and the Hugging Face Transformers library to load my data, tokenize it for BERT, define the fine-tuning process, and save the model.
2.  **Containerize this script** with a Dockerfile, ensuring all dependencies were correctly installed. This took *ages* of trial and error. Getting the `gcloud` SDK to play nice inside the container for things like accessing data in Google Cloud Storage buckets was a specific pain point. I remember one late night staring at a `ModuleNotFoundError` inside a Docker log on Vertex AI, wanting to throw my laptop out the window. The solution was a tiny change in my `requirements.txt` and Dockerfile `COPY` command, found after hours of searching through Vertex AI documentation and some obscure StackOverflow threads.
3.  **Upload the container image** to Google Artifact Registry.
4.  **Submit a custom training job** via the Vertex AI Python SDK, pointing it to my container and my data in GCS.

```python
# Part of my (simplified) custom training script for Vertex AI
# main_training_script.py

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizerFast
from sklearn.model_selection import train_test_split
import pandas as pd
import gcsfs # To access GCS
import os

# --- Hyperparameters and Configuration ---
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 3 # Positive, Negative, Neutral
MAX_LENGTH = 128 # Max token length for BERT
BATCH_SIZE = 16 # This was limited by GPU memory on the instances I chose
EPOCHS = 3 # Found that more epochs led to overfitting on my smallish dataset

# --- Load Data (assumes it's been copied from GCS to the container or mounted) ---
# In reality, this involved using os.environ.get('AIP_TRAINING_DATA_URI', '') etc.
# For Vertex AI, data URIs are typically passed as env variables.
# For this example, let's assume a local CSV path that would be in the container.
# data_path = "/app/data/labeled_financial_news.csv" 
# For actual training, I used the GCS path directly via gcsfs
# GCS_DATA_PATH = "gs://my-financial-news-bucket/data/labeled_financial_news_cleaned_tokenized.csv"

def load_and_preprocess_data(gcs_path):
    # fs = gcsfs.GCSFileSystem()
    # with fs.open(gcs_path) as f:
    #     df = pd.read_csv(f)
    # This is a placeholder for the actual loading logic
    # For the sake of the example, let's imagine a simple structure:
    data = {
        'text': ["Great news for investors!", "Market plunges on bad data.", "Company reports steady earnings."],
        'label': # 0: Positive, 1: Negative, 2: Neutral
    }
    df = pd.DataFrame(data)
    
    # Simple mapping, in reality, this was more complex with error checking
    # label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    # df['label_encoded'] = df['sentiment_label'].map(label_map)

    texts = df['text'].tolist()
    labels = tf.keras.utils.to_categorical(df['label'].tolist(), num_classes=NUM_LABELS)
    return texts, labels

# --- Tokenization ---
# tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
# texts, labels = load_and_preprocess_data(GCS_DATA_PATH) # Use the GCS path
# train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')
# val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')

# train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(100).batch(BATCH_SIZE)
# val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(BATCH_SIZE)

# --- Model Fine-tuning ---
# model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) # Typical learning rate for BERT fine-tuning
# model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# print("Starting model training...")
# model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
# print("Training complete.")

# --- Save Model to GCS (for Vertex AI Endpoint deployment) ---
# AIP_MODEL_DIR is an environment variable provided by Vertex AI Training
# model_save_path = os.environ.get('AIP_MODEL_DIR', 'gs://my-financial-models-bucket/sentiment_v1/')
# if not model_save_path.startswith("gs://"): # Handle local testing fallback
#     model_save_path = "trained_model_local" 
#     os.makedirs(model_save_path, exist_ok=True)

# print(f"Saving model to {model_save_path}")
# model.save_pretrained(model_save_path) # Hugging Face format
# tokenizer.save_pretrained(model_save_path)
# print("Model saved successfully.")
```
The `AIP_MODEL_DIR` environment variable was key for saving the model in a way Vertex AI could then pick up for deployment. I found that out from a Google Cloud tutorial after my initial attempts to save the model locally in the container obviously didn't persist it where Vertex AI Endpoints could find it. The first few custom training runs were slow and expensive as I picked machine types that were either too weak (resulting in OOM errors for the GPU) or overkill. It was a balancing act.

One "Aha!" moment was when I started experimenting with different learning rates and batch sizes. The defaults I initially picked were okay, but careful tweaking, guided by some research papers on fine-tuning BERT for classification tasks, started to nudge the validation accuracy up. Another breakthrough was realizing the importance of `max_length` in the tokenizer. Some financial news articles can be quite long, but BERT has a token limit. Truncating too aggressively lost context, but too long a sequence blew up memory. `MAX_LENGTH = 128` or `256` seemed to be a sweet spot for headlines and short summaries.

After many iterations (and a growing GCP bill that made me sweat a little), I got a custom-trained sentiment model that was performing noticeably better on my financial dataset – closer to 78-80% accuracy on sentiment classification alone.

**The 85% Claim: Linking Sentiment to Price Changes**

Getting the sentiment score was one thing; linking it to price changes was the next.
I defined "price change events" as significant movements (e.g., >1% or >2% change within X hours of a news article for a specific stock, adjusted for general market volatility by looking at an index like SPY).

My process was:
1.  For each news article, get its sentiment score (positive, negative, neutral probabilities) from my deployed Vertex AI endpoint.
2.  Identify the associated stock ticker.
3.  Fetch the stock price data around the article's publication time.
4.  Calculate price changes at T+1hr, T+4hr, T+24hr.
5.  Label these price changes as 'up', 'down', or 'flat/no significant change'.

Then came the core of the analysis: could the sentiment predict the direction of these price changes?
I built a simple logistic regression model (using scikit-learn) where the features were the sentiment scores (probability of positive, negative, neutral) from my BERT model, and the target was the discretized price movement (up/down). I also experimented with including features like news source reputation (crudely estimated) and article length, but sentiment was the dominant signal.

```python
# Simplified snippet of the correlation analysis part
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assume 'news_sentiment_and_prices.csv' has columns:
# 'news_id', 'ticker', 'timestamp', 'sentiment_prob_positive', 'sentiment_prob_negative', 
# 'price_change_1hr_direction' (e.g., 0 for down, 1 for up, 2 for flat - simplified here)
# 'price_change_4hr_direction', etc.

# df = pd.read_csv('news_sentiment_and_prices_joined_data.csv')
# # Preprocessing: filter out 'flat' movements for a clearer binary signal for this specific model
# df_filtered = df[df['price_change_4hr_direction'].isin()] # 0: Down, 1: Up

# features = ['sentiment_prob_positive', 'sentiment_prob_negative'] # Using probabilities from my Vertex AI model
# X = df_filtered[features]
# y = df_filtered['price_change_4hr_direction']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # I tried a few classifiers here, Logistic Regression was simple and interpretable
# log_reg_model = LogisticRegression()
# # Had to play with class_weight='balanced' because price movements aren't always 50/50
# # log_reg_model = LogisticRegression(class_weight='balanced', solver='liblinear') 

# log_reg_model.fit(X_train, y_train)

# y_pred = log_reg_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy linking sentiment to 4hr price direction: {accuracy*100:.2f}%")
# This is where I started seeing numbers in the 80-85% range on my test set.
# Of course, this required careful feature engineering and definition of "significant price change".
```

The "85% accuracy" specifically refers to the performance of this *secondary* model (the logistic regression) on a held-out test set, in predicting whether a stock's price would move up or down (as a binary outcome, ignoring 'flat' for this metric) in the 4-hour window following a news item with a strong positive or negative sentiment score from my custom BERT model.

**Was it a "true" 85%? Caveats are important.**
*   **Correlation, not Causation:** This is crucial. The model finds a statistical link, but it doesn't *prove* news sentiment *causes* price changes. Many other factors are at play.
*   **Specific Conditions:** This accuracy was achieved under specific conditions: using my definition of "significant" sentiment, my window for price changes (4 hours seemed a good balance), and for the specific set of stocks and time period I analyzed. It might not generalize perfectly to all stocks or all market conditions.
*   **Backtesting Limitations:** While I used a test set, this is still essentially a backtest. Real-world, forward-testing would be the ultimate proof, which is a whole other level of complexity (and risk, if actual money were involved).

I was genuinely surprised and pretty thrilled when I first saw results consistently above 80%. There was a lot of iterative tweaking: adjusting the threshold for what constituted a "strong" sentiment signal, experimenting with different time windows for price changes, and ensuring my training/test splits were clean and not leaking data. One specific issue I remember was initially not properly lagging the price data relative to the news timestamp, which led to inflated (and wrong) accuracies because the model was inadvertently peeking at future prices. Debugging that involved a lot of `print()` statements and manually checking timestamps.

**Struggles and Small Victories**

*   **Vertex AI Costs:** Keeping an eye on the billing dashboard was stressful. I learned to be very diligent about shutting down endpoints and training jobs when not in use. I also learned to start with smaller machine types for debugging and only scale up when I was confident the code would run.
*   **Debugging in the Cloud:** Debugging a Docker container running a training job on Vertex AI is not like debugging locally. Logs are your best friend, but sometimes they are cryptic. I relied heavily on the Google Cloud console's logging interface and `gcloud ai custom-jobs stream-logs <job_id>` command.
*   **The "It Works!" Moment:** The first time my deployed Vertex AI endpoint successfully returned a sentiment score for a piece of news I sent it via a Python script felt like a huge win. It was probably 2 AM, but it was a good 2 AM.
*   **Documentation Diving:** I spent countless hours in the Vertex AI documentation, Hugging Face docs, TensorFlow guides, and various forum posts. Sometimes the answer was a single parameter I'd overlooked, or an example snippet in a seemingly unrelated part of the docs. For instance, understanding the exact format Vertex AI Endpoints expected for input and how to structure my prediction function in the custom container took a lot of back and forth with the "Deploying custom models" section of the Vertex AI docs.

**Reflections**

This project was probably the most complex one I've undertaken so far. It pushed me to learn a lot more about the practical side of MLOps – not just training a model, but preparing data at scale, containerizing applications, deploying to the cloud, and monitoring costs. The 85% accuracy figure, while specific to my setup, feels like a significant achievement and a validation of the approach.

There’s so much more that could be done. Incorporating more news sources, analyzing full article text instead of just headlines/summaries (which would require a more robust model and more compute), looking at different asset classes, or even trying to predict the *magnitude* of price changes, not just direction.

For now, though, I'm pretty happy with how this turned out. It was a deep dive into a challenging domain, and I came out the other side with a working system and a much better understanding of the end-to-end machine learning lifecycle. And, crucially, a healthy respect for anyone who does this for a living – it’s tough stuff!