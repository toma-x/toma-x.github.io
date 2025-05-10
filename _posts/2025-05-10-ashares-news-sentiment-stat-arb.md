---
layout: post
title: StatArb in A-Shares via News Sentiment
---

## Trading A-Shares with News Sentiment: A Python and BERT Adventure

After my deep dive into high-frequency C++ last semester, I wanted to switch gears a bit for my next personal project. I've always been intrigued by the intersection of natural language processing and finance, and the idea of using news sentiment to predict stock movements seemed like a fertile ground for exploration. I decided to focus on the Chinese A-share market, specifically components of the CSI 300 index, and build a Python-based pipeline to see if I could extract some alpha using BERT for sentiment analysis. The journey, as always, was full of learning curves and "why isn't this working?!" moments, but I managed to get a simulated Sharpe of 1.8, which felt like a decent outcome.

### The Hypothesis: Can News Sentiment Drive StatArb in China?

The core idea was to test if significant positive or negative news sentiment surrounding a specific company could lead to temporary mispricings that a statistical arbitrage strategy could exploit. For example, if a company gets a sudden wave of very positive news, its stock price might overreact or underreact in the short term. My plan was to quantify this sentiment and build a simple long/short strategy: go long on stocks with strong positive sentiment and short those with strong negative sentiment, assuming some mean reversion or continued momentum based on the signal. The A-share market, with its large retail participation, felt like a place where sentiment could potentially play a significant role.

My toolkit for this was primarily Python. Its extensive libraries for data science (`pandas`, `numpy`, `scipy`) and NLP (especially Hugging Face's `transformers`) made it a natural choice. Plus, after the C++ intensity, I was looking forward to Python's quicker iteration cycles for this kind of research project.

### Building the Pipeline: From Raw News to Trading Signals

The project broke down into several key stages:

1.  **News Data Acquisition**: This was an immediate hurdle. Getting high-quality, structured, and timely news data for Chinese companies isn't as straightforward as, say, using a nice API for US news. I explored a few options, including trying to scrape some financial news portals. Eventually, for the sake of having a somewhat manageable dataset for development, I found an academic source that had a collection of historical news articles (mostly text, source, and timestamp) for various Chinese companies. It wasn't real-time, but it was enough for a backtesting project. The data was a mix of headlines and full articles, all in Mandarin.

2.  **Sentiment Analysis with BERT**: This was the core of the NLP work. I knew traditional bag-of-words or TF-IDF based sentiment analysis might struggle with the nuances of financial news, especially in Chinese. BERT, with its contextual understanding, seemed like the way to go.
    *   **Choosing a Model**: I didn't have the resources or data to train a BERT model from scratch. I browsed through Hugging Face's model hub and decided to use a pre-trained Chinese BERT model, specifically `bert-base-chinese`, and then looked for a version already fine-tuned for sentiment analysis on Chinese text. I found a couple that seemed promising. I initially tried `uer/roberta-base-finetuned-dianping-chinese` for general sentiment but later shifted to a model more broadly trained on varied Chinese texts when I found my financial news wasn't always well-captured by a reviews-focused model. The key was finding one that provided a reasonably granular sentiment output (e.g., positive, negative, neutral, or even a score).
    *   **Processing**: For each news item, I extracted the headline and the first few sentences of the article body, concatenated them, and then fed them into the BERT model. Tokenization for Chinese is handled pretty well by the pre-trained models' tokenizers, which was a relief as I was worried I'd have to integrate a separate word segmenter like `jieba` and deal with potential vocabulary mismatches.

    ```python
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import pandas as pd

    # Example: Simplified setup for BERT sentiment
    # In reality, I had this wrapped in a class for managing news items
    model_name = "some-chinese-sentiment-bert-model" # Placeholder for the actual model I used
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def get_sentiment_score(text_snippet):
        # My actual implementation involved more preprocessing and error handling
        # Also, some models output labels, others scores directly. I had to normalize this.
        # Truncation was important because BERT has a max sequence length
        max_length = 512 
        results = sentiment_analyzer(text_snippet, truncation=True, max_length=max_length)
        # Assuming the pipeline returns a label and a score, e.g., {'label': 'POSITIVE', 'score': 0.98}
        # I converted this to a numerical score, e.g., positive: 1, negative: -1, neutral: 0, scaled by score
        # This part took a fair bit of tweaking to get something consistent.
        score = 0.0
        if results['label'] == 'POSITIVE': # Or whatever the model's positive label was
            score = results['score']
        elif results['label'] == 'NEGATIVE': # Or negative label
            score = -results['score']
        return score

    # Example usage:
    # news_df['sentiment_score'] = news_df['text_content'].apply(get_sentiment_score)
    ```
    One of the first issues was processing speed. Running BERT inference on thousands of articles, even with GPU support on my university's cluster (when I could get time on it), was slow. I had to implement batching for the `sentiment_analyzer` and make sure my data loading was efficient. I spent a weekend just optimizing this part, reading through `transformers` documentation on how to best feed data to the pipeline.

3.  **Signal Generation**: Once I had sentiment scores for individual news items, I needed to aggregate them into a daily (or intra-day, if data permitted, but mine was mostly daily) sentiment signal per stock.
    *   **Aggregation**: I decided to average the sentiment scores of all news items for a particular stock on a given day. I also experimented with more complex schemes, like weighting recent news more heavily or considering the volume of news, but simple averaging was my baseline.
    *   **Neutrality and Extremes**: A lot of news is neutral. I had to define thresholds for what constituted a "strong" positive or negative signal. This was very empirical – I plotted distributions of sentiment scores and tried to pick cutoffs that represented the top/bottom deciles, for instance.

    ```python
    # Assuming news_df has columns: ['timestamp', 'stock_code', 'sentiment_score']
    # stock_daily_sentiment = news_df.groupby(['stock_code', pd.Grouper(key='timestamp', freq='D')])['sentiment_score'].mean().reset_index()
    
    # This was a point of much iteration. Just a mean might smooth out critical single news.
    # I also tried max/min sentiment in a day, or number of significantly positive/negative articles.
    # For my final run that got the 1.8 Sharpe, I used a combination:
    # average sentiment, but it had to be confirmed by at least N articles to reduce noise from single, possibly misclassified, news.
    
    # Defining thresholds for trading signals
    positive_threshold = 0.6 # e.g. average score > 0.6
    negative_threshold = -0.6 # e.g. average score < -0.6
    # These were tuned based on historical distribution of scores
    ```

4.  **Portfolio Construction and Backtesting**: With daily sentiment signals per stock, I built a simple backtester.
    *   **Strategy Logic**: If a stock crossed the `positive_threshold`, I'd simulate a long position. If it crossed `negative_threshold`, a short position. I started with a fixed holding period (e.g., 1-3 days) but found that exiting based on sentiment neutralization (score moving back towards zero) or a stop-loss worked better.
    *   **Universe**: My universe was the components of the CSI 300. On any given day, I'd form a small portfolio of the top N long candidates and top M short candidates, trying to keep it somewhat market neutral in terms of total capital deployed, though not strictly dollar neutral.
    *   **Execution Assumptions**: This is where simulations can get tricky. I assumed I could trade at the day's closing price (or next day's open) following the news signal. I also factored in estimated transaction costs (brokerage fees and stamp duty for A-shares). I knew this was an idealization; real slippage could be a killer.
    *   **Lookahead Bias**: A major demon in any backtest. I was very careful to ensure that the news data used for generating a signal on day `D` was only processed *after* market close on day `D` (or before market open on `D+1`) and that trading decisions for `D+1` used information strictly available before `D+1`'s open. My news dataset had timestamps, and I aligned these meticulously with my daily stock price data (OHLCV for CSI 300 components, also from an academic source).

    ```python
    # Simplified backtesting logic sketch
    # positions = {} # stock_code -> 'long'/'short'
    # portfolio_history = [] 
    # initial_capital = 1000000
    # capital = initial_capital
    #
    # for day_data in backtest_days: # Looping through historical data
    #     daily_pnl = 0
    #     trades_today = []
    #
    #     # 1. Calculate P&L on existing positions, check for exits
    #     for stock_code, position_type in list(positions.items()):
    #         # Check exit conditions: stop-loss, take-profit, sentiment neutralization
    #         # Simplified exit: if stock_daily_sentiment[stock_code][day_data.date] approaches neutral
    #         # current_price = get_price(stock_code, day_data.date, 'close')
    #         # entry_price = positions[stock_code]['entry_price']
    #         # if position_type == 'long' and (current_price < stop_loss_price or sentiment_neutralized):
    #         #     capital += (current_price - entry_price) * shares - transaction_cost
    #         #     del positions[stock_code]
    #         #     trades_today.append({'stock': stock_code, 'action': 'sell_to_close'})
    #         pass # More detailed P&L logic here
    #
    #     # 2. Look for new entries based on today's (day_data.date) sentiment signals
    #     # signals_today = get_signals_for_date(stock_daily_sentiment, day_data.date)
    #     # for stock_code, signal_strength in signals_today.items():
    #     #     if stock_code not in positions:
    #     #         if signal_strength > positive_threshold:
    #     #             # Enter long
    #     #             # shares_to_buy = calculate_position_size(capital, stock_code, current_price)
    #     #             # positions[stock_code] = {'type': 'long', 'entry_price': current_price, 'shares': shares_to_buy}
    #     #             # capital -= current_price * shares_to_buy + transaction_cost
    #     #             # trades_today.append({'stock': stock_code, 'action': 'buy_to_open'})
    #     #         elif signal_strength < negative_threshold:
    #     #             # Enter short (assuming shorting is feasible and modeling its costs)
    #     #             pass
    #     #
    #     # portfolio_value_today = capital + calculate_value_of_open_positions(positions, day_data.date)
    #     # portfolio_history.append({'date': day_data.date, 'value': portfolio_value_today})
    #
    # # After loop, calculate Sharpe, drawdowns, etc. from portfolio_history
    ```
    Actually coding the backtester, even a simple one, took way longer than I thought. Handling position sizing, daily P&L updates, and ensuring correct alignment of dates between sentiment signals and price data was fiddly. I spent a lot of time with `pandas` trying to `merge_asof` and `groupby` data correctly.

### Struggles and "Aha!" Moments

*   **Chinese NLP Nuances**: While `bert-base-chinese` handles tokenization, understanding the sentiment expressed in financial news sometimes felt harder than generic text. Financial jargon, implicit meanings, and government announcements often carry sentiment that isn't always obvious from surface-level wording. My off-the-shelf fine-tuned BERT was good, but not perfect. A breakthrough here was less about model architecture and more about aggressive pre-filtering of news: I started trying to categorize news (e.g., earnings announcements, regulatory news, general commentary) and found that sentiment from certain types of news was more reliable.
*   **Defining "Signal"**: Just because BERT said a piece of news was "90% positive" didn't mean the stock would fly. The market might have already priced it in, or the news might be insignificant. The "aha!" moment was realizing that the *change* in sentiment or the *abnormality* of sentiment (e.g., a stock that usually has neutral news suddenly getting a burst of highly positive news) was often a better signal than the absolute level. I started incorporating a baseline sentiment for each stock.
*   **The Allure of Overfitting**: With so many parameters (sentiment thresholds, holding periods, choice of BERT model, aggregation methods), it's incredibly easy to curve-fit the backtest to historical data. I tried to mitigate this by having a conceptual "out-of-sample" period in my historical data that I only tested on very late in the process, but it's always a concern. The 1.8 Sharpe is on the full dataset, so it needs to be taken with a grain of salt.
*   **Data Quality and Alignment**: Sometimes news timestamps were vague (just a date, no time), making it hard to be sure if the news was pre-market, intra-day, or post-market. This could significantly impact backtest validity. I eventually made a simplifying assumption to only trade on `T+1` based on news aggregated up to day `T`'s close. This felt more robust.

### Simulated Performance and Key Observations

The final iteration of the strategy, after many tweaks to the sentiment processing and signal generation logic, yielded a simulated Sharpe ratio of around 1.8 over the several years of historical data I had. There were definitely periods where the strategy did very well, typically during times of higher market volatility or when there was a lot of distinct news flow. There were also flat periods or even slight drawdowns, especially when market movements seemed driven by macro factors rather than idiosyncratic company news.

One key observation was that the signal decay seemed relatively quick. The predictive power of a strong sentiment spike often diminished after a few days. This reinforced the idea of it being a short-term statistical arbitrage type of play, not a long-term investment strategy based on deep fundamental analysis.

Another challenge was news volume. Some stocks had multiple news items daily, others very few. My aggregation method had to be robust to this. For stocks with sparse news, sentiment signals were less reliable.

### Reflections and Future Directions

This project was a fantastic learning experience in applying advanced NLP models to a practical (albeit simulated) trading problem. Working with Chinese language data also added an interesting layer of complexity. Python's ecosystem was invaluable.

If I were to push this further:
1.  **Better Sentiment Models**: Fine-tune a BERT model specifically on Chinese financial news, perhaps with a more nuanced output than just positive/negative/neutral (e.g., specific emotions, or a finer-grained score). This would require a labeled dataset, which is a project in itself.
2.  **More Sophisticated Signals**: Incorporate other factors alongside raw sentiment:
    *   News topic modeling (what is the news about?).
    *   Source credibility.
    *   Combined sentiment from multiple sources.
    *   Relationship between news sentiment and trading volume changes.
3.  **Richer Backtesting**:
    *   More realistic execution model: Incorporate slippage, market impact for larger hypothetical trades.
    *   More advanced risk management: Dynamic position sizing, portfolio-level risk controls.
4.  **Alternative Data**: Explore sentiment from social media (e.g., Weibo for the Chinese context), though this comes with even more noise.
5.  **Causality**: Dig deeper into whether the sentiment is truly *causing* the price movement or just *correlating* with it due to other underlying factors. This is a hard problem.

Even with the simplifications and the simulated nature, it was incredibly satisfying to see a pipeline take raw news, process it through a complex NLP model, and generate trading decisions that, at least historically, showed some positive performance. It really highlighted how much potential there is at the intersection of unstructured data and quantitative finance.