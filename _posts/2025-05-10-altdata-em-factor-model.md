---
layout: post
title: Alt Data EM Factor Model
---

## Building an Alternative Data EM Factor Model with News Sentiment

This semester, I decided to dive deeper into quantitative finance for my personal project, specifically looking at emerging markets (EM) equities. The idea was to see if I could engineer alpha factors from alternative data, and I settled on news sentiment, sourced via the GNews API. The end goal was to build a sparse factor model using LASSO regression to identify if these sentiment-derived factors had any explanatory power on EM stock returns. It was quite a journey, with a lot of data wrangling and model tweaking in Python.

### Phase 1: Getting and Processing News Data

The first major hurdle was acquiring a consistent stream of news data for a basket of EM equities. I looked at a few news APIs, but GNews seemed like a reasonable starting point given its coverage and relatively straightforward API access. I primarily focused on major EM tickers from countries like Brazil, India, China, and South Africa.

Fetching the data was one thing; making sense of it was another. The GNews API provides articles with titles, descriptions, and sometimes full content. My initial plan was to process full content, but the volume was just too much for my local machine to handle efficiently, especially the sentiment analysis part. So, I made a pragmatic decision to focus on headlines and descriptions. This felt like a necessary compromise, though I worried I might be losing some nuance.

I wrote a Python script using the `requests` library (though GNews has its own client, I found direct calls sometimes easier to debug when I hit rate limits).

```python
import requests
import pandas as pd
import time

# This is a simplified example of how I fetched news for a list of tickers
# In reality, I had to handle pagination, error checking, and a more robust way to store the data
# than just a big list of dictionaries.

API_KEY = "YOUR_GNEWS_API_KEY" # Kept this in a separate config file usually
BASE_URL = "https://gnews.io/api/v4/search"

# Example EM Tickers (in practice, I had a much longer list from specific indices)
EM_TICKERS_SAMPLE = ["VALE3.SA", "INFY.NS", "BABA"] 

all_news_data = []

def fetch_news_for_ticker(ticker_symbol):
    query = f"{ticker_symbol} stock OR market OR finance" # Broadened query a bit
    params = {
        "q": query,
        "token": API_KEY,
        "lang": "en", # Focused on English news for consistent sentiment analysis
        "country": "any", # Relied on ticker for EM focus
        "max": 10, # Kept it small per request to manage API limits during development
        "sortby": "publishedAt"
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status() # Raises an exception for HTTP errors
        articles = response.json().get("articles", [])
        for article in articles:
            all_news_data.append({
                "ticker": ticker_symbol,
                "title": article.get("title"),
                "description": article.get("description"),
                "published_at": article.get("publishedAt"),
                "source_name": article.get("source", {}).get("name")
            })
        time.sleep(1) # Basic rate limiting, GNews is quite sensitive
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {ticker_symbol}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {ticker_symbol}: {e}")


# for ticker in EM_TICKERS_SAMPLE:
# fetch_news_for_ticker(ticker) # I ran this iteratively, saving to CSVs periodically

# news_df = pd.DataFrame(all_news_data)
# news_df['published_at'] = pd.to_datetime(news_df['published_at'])
# news_df = news_df.dropna(subset=['title', 'description']) # Crucial step
```
One early mistake was not being aggressive enough with `dropna`. Missing titles or descriptions were messing up the subsequent sentiment scoring, leading to weird `NaN` propagation in my factors. I also quickly realized that managing API keys and request limits required careful planning; a simple `time.sleep(1)` was a very crude way to handle it, but for a small-scale pull, it prevented immediate blocking. For a larger dataset, I'd need a more sophisticated queueing system or at least exponential backoff.

### Phase 2: Sentiment Analysis – Easier Said Than Done

With the news text data (mostly headlines and snippets), the next step was sentiment scoring. I started with TextBlob, as it’s very straightforward to use in Python.

```python
from textblob import TextBlob

# Assuming news_df from the previous step
# This is a simplified application; I later batched this for efficiency

# def get_sentiment(text):
# try:
# return TextBlob(text).sentiment.polarity # Polarity is between -1 and 1
# except Exception:
# return 0.0 # Default to neutral if TextBlob fails on some weird characters

# news_df['sentiment_title'] = news_df['title'].apply(get_sentiment)
# news_df['sentiment_description'] = news_df['description'].apply(get_sentiment)
# news_df['sentiment_combined'] = (news_df['sentiment_title'] + news_df['sentiment_description']) / 2
```

The results from TextBlob were okay for clearly positive or negative statements, but financial news is often quite neutral or uses domain-specific language that generic sentiment analyzers struggle with. For instance, a headline like "XYZ Corp Misses Earnings Estimates" is clearly negative for the stock, but TextBlob might score it as neutral or only slightly negative depending on the exact phrasing.

I spent some time looking into VADER (Valence Aware Dictionary and sEntiment Reasoner), which is often recommended for social media text due to its sensitivity to negation, capitalization, and punctuation. I even found a few research papers comparing sentiment tools on financial text. While VADER seemed marginally better, I realized that truly accurate financial sentiment would probably require a fine-tuned model on financial news, perhaps something like FinBERT. That was beyond the scope of this particular project given my time constraints and focus on the factor modeling aspect. So, I stuck with TextBlob for combined headline/description sentiment, acknowledging its limitations but aiming to see if even this somewhat noisy signal could be useful. My rationale was that aggregation over many articles might smooth out individual inaccuracies.

### Phase 3: Engineering Sentiment Factors

Once I had daily sentiment scores per stock (by averaging sentiment of all articles for a given stock on a given day), I needed to turn these into factors. Simply using the raw daily average sentiment felt too simplistic. I thought about what kind of sentiment signals might actually drive stock prices or be picked up by a model:

1.  **Lagged Sentiment:** News from yesterday or a few days ago might still impact today's price. I created factors like `sentiment_1day_lag`, `sentiment_3day_lag`.
2.  **Sentiment Momentum:** A rising trend in positive sentiment, or a deepening negative trend. I calculated this using simple rolling differences or slopes over a short window (e.g., 5-day change in average sentiment).
3.  **Sentiment Volatility/Dispersion:** High disagreement in news sentiment (some very positive, some very negative articles on the same day for the same stock) could indicate uncertainty. I tried calculating the standard deviation of sentiment scores for a given stock on a given day, but this was often zero if there was only one or two articles.
4.  **Volume of News:** A surge in news articles (regardless of sentiment) could be a factor. I used a rolling count of articles.

This feature engineering phase was very iterative. I'd create a potential factor, then plot it against stock returns for a few tickers to see if there was *any* visual relationship. Most of the time, there wasn't an obvious one, which was a bit disheartening but also realistic.

Aligning the daily sentiment data with daily stock return data was also a pain. I used `pandas` for this, relying heavily on `merge_asof` after ensuring my datetime indices were perfectly clean. One specific issue I ran into was timezones – GNews provides UTC timestamps, and I had to ensure my stock price data (sourced from Yahoo Finance via `yfinance`) was consistently handled, especially for markets that close at different UTC times. I decided to align everything to the market close UTC time for simplicity. Forgetting to `tz_localize(None)` or `tz_convert('UTC')` at the right stages caused me a lot of headaches initially, with data appearing to be misaligned by a day.

### Phase 4: Building the Sparse Factor Model with LASSO

With a set of potential sentiment factors (around 10-15 after some initial pruning), I moved to the modeling stage. The goal wasn't necessarily to build the *best* predictive model, but to see which, if any, of these news sentiment factors were consistently selected by a regularized regression model. I chose LASSO (Least Absolute Shrinkage and Selection Operator) regression because of its L1 penalty, which is great for feature selection as it tends to shrink the coefficients of less important features to exactly zero. This seemed appropriate given that I suspected many of my engineered factors might be noisy or redundant.

My dependent variable (`y`) was daily forward returns (e.g., next day's open-to-open return to avoid lookahead bias from using close-to-close returns with sentiment calculated from news published during the trading day). My independent variables (`X`) were the lagged sentiment factors.

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Or TimeSeriesSplit for more rigor
import pandas as pd
import numpy as np

# Assume 'final_data_df' is a pandas DataFrame with:
# - 'forward_return' as the target variable
# - Columns for each engineered sentiment factor (e.g., 'sentiment_1day_lag', 'sentiment_momentum_5day')
# - A DateTimeIndex

# Drop rows with NaN in crucial columns (especially forward_return which can't be imputed easily)
# final_data_df = final_data_df.dropna(subset=['forward_return'])
# For factor columns, I experimented with mean imputation or forward fill,
# but for LASSO, handling NaNs before scaling is important.
# final_data_df[factor_columns] = final_data_df[factor_columns].fillna(final_data_df[factor_columns].mean())


# factor_columns = [col for col in final_data_df.columns if 'sentiment_' in col]
# X = final_data_df[factor_columns]
# y = final_data_df['forward_return']

# if X.empty or y.empty or X.isnull().values.any() or y.isnull().values.any():
    # print("Problem with X or y before scaling/splitting. Check NaNs or data prep.")
    # This was a common checkpoint for me.
# else:
    # Scale features - LASSO is sensitive to feature scaling
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # For time series, a simple train_test_split is not ideal.
    # I should have used TimeSeriesSplit from sklearn.model_selection for proper backtesting.
    # For this iteration of the project, I sometimes used a simple chronological split:
    # split_point = int(len(X_scaled_df) * 0.8)
    # X_train, X_test = X_scaled_df[:split_point], X_scaled_df[split_point:]
    # y_train, y_test = y[:split_point], y[split_point:]
    
    # Using LassoCV to find the best alpha
    # The n_alphas and max_iter parameters sometimes needed tuning.
    # If LassoCV took too long or didn't converge, I'd reduce n_alphas or increase max_iter.
    # lasso_cv_model = LassoCV(cv=5, random_state=42, n_alphas=100, max_iter=5000, tol=0.001) # cv=5 for KFold
    # lasso_cv_model.fit(X_train, y_train) # Using training data

    # print(f"Optimal alpha chosen by LassoCV: {lasso_cv_model.alpha_}")

    # coefficients = pd.Series(lasso_cv_model.coef_, index=X_train.columns)
    # selected_features = coefficients[coefficients != 0]
    # print("Selected features and their coefficients:")
    # print(selected_features)

    # print(f"LASSO Model R-squared on test set: {lasso_cv_model.score(X_test, y_test)}")
```

One of the tricky parts was choosing the `alpha` parameter for LASSO. Manually trying values is tedious, so I used `LassoCV` from scikit-learn, which performs cross-validation to find the optimal alpha. I had to be careful with `cv` in `LassoCV` for time-series data. Standard K-Fold shuffles data, which isn't appropriate for time-dependent observations. `TimeSeriesSplit` is the correct way, but for quicker iterations during development, I confess I sometimes just used a simple chronological train-test split or even looked at coefficients from `LassoCV` trained on a large chunk of recent data, keeping in mind the limitations. I remember reading a scikit-learn documentation page on cross-validation iterators and realizing my initial `cv=5` (which defaults to KFold) was theoretically incorrect for this problem, though `LassoCV` itself tries to be somewhat robust.

Convergence was another issue. With many features or a small `tol` (tolerance for stopping criteria), `LassoCV` could take a very long time or fail to converge. Increasing `max_iter` often helped, but it was a balance. I also found that scaling the features using `StandardScaler` was crucial; LASSO penalizes coefficients, so the scale of the features directly impacts which ones get shrunk. Without scaling, a factor with a naturally large range of values might be penalized differently than one with a small range, irrespective of its true importance.

### Challenges, Confusion, and Breakthroughs

*   **Data Quality and Alignment:** This was a constant battle. Ensuring dates matched, handling missing data appropriately (especially for returns and lagged factors), and dealing with timezones for international news and markets took up a surprising amount of time. My "breakthrough" here was less a single moment and more the gradual realization that meticulous data cleaning and preprocessing are non-negotiable. I started writing small utility functions to check data integrity at each step.
*   **Sentiment Nuance:** Relying on a generic sentiment analyzer like TextBlob for financial news felt like a significant compromise. The "aha!" moment here was when I stopped trying to get *perfect* sentiment and instead focused on whether *consistent, albeit noisy*, sentiment signals could still be picked up by the model in aggregate.
*   **Interpreting LASSO Coefficients:** When LASSO selected certain factors, it was tempting to over-interpret the magnitude of their coefficients. I recall finding a StackOverflow discussion that cautioned against this, especially with correlated features, as LASSO might arbitrarily pick one feature over a similar correlated one. So, I focused more on *which* factors were consistently selected (non-zero coefficients) across different time periods or data subsets, rather than their exact weight. The sign of the coefficient (positive or negative relationship with returns) was also a key takeaway.
*   **Lookahead Bias:** This was a spectre looming over the project. I had to be extremely careful that information used in factor construction (like news sentiment) was strictly available *before* the period for which I was calculating returns. For example, using news published throughout day D to predict returns from close of day D to close of day D+1 is problematic. I tried to mitigate this by using lagged factors and calculating returns from next day's open, but it's a subtle issue. I even drew out timelines on paper for a few examples to convince myself my lags were correct.

### Preliminary Observations (Not Investment Advice!)

The results were, as expected for a student project, mixed. Some of the engineered sentiment factors did get selected by the LASSO model with non-zero coefficients, suggesting they held some explanatory power over EM equity returns, at least within the sample period. For instance, a 1-day lagged average sentiment and a 5-day sentiment momentum factor for certain sectors appeared somewhat consistently. However, the overall R-squared of the models was generally low, which is typical for financial return prediction models, especially with just one class of alternative data.

This project wasn't about finding a "get rich quick" signal. It was about going through the process: from raw, messy alternative data to engineered factors, and finally to a statistical model. The amount of data cleaning, careful indexing, and critical thinking about potential biases was far more than I initially anticipated.

### Future Directions

There are many ways this could be extended. Using a more sophisticated, finance-specific sentiment analyzer (like a pre-trained BERT model fine-tuned on financial news) would be a major step up. Exploring more complex factor definitions, perhaps incorporating topic modeling on the news content to differentiate between, say, macroeconomic news sentiment and company-specific news sentiment, could also be interesting. Finally, a more rigorous backtesting framework using something like `zipline` or `QuantConnect` would be essential before drawing any stronger conclusions about the practical utility of these factors.

For now, it was a fantastic learning experience in handling alternative data and applying machine learning techniques to a noisy real-world problem.