---
layout: post
title: LLM ESG Anomaly Detector
---

## Fine-Tuning GPT-NeoX for ESG Controversy Detection and LangChain for Sentiment Analysis: A Deep Dive

This project has been quite a journey. For a while now, I've been interested in the intersection of Natural Language Processing and Environmental, Social, and Governance (ESG) factors in investing. The idea was to see if I could build a system to first detect potential ESG controversies from news data and then analyze how companies address (or don't address) these issues in their sustainability reports. It turned into a two-pronged approach: fine-tuning a language model for event detection and then using LangChain agents to dig deeper into corporate disclosures.

### Part 1: Spotting ESG Trouble with a Fine-Tuned LLM

The first major task was to get a model that could flag news articles discussing ESG controversies. I'd read a bit about EleutherAI's models and decided to try fine-tuning `EleutherAI/GPT-NeoX-20B`. Okay, realistically, the full 20B parameter model was a bit ambitious for my setup, even with cloud credits. I ended up working with a smaller checkpoint or considering techniques like LoRA to make it feasible, but the base architecture was GPT-NeoX. My thinking was that a foundation model with strong language understanding would be a good starting point, rather than training something from scratch or relying purely on keyword spotting, which feels a bit primitive for nuanced ESG issues.

**Data, Data, Data (and the Pain Thereof)**

This was, predictably, the hardest part. There isn't exactly a perfectly labeled "ESG Controversy News" dataset just lying around. I started by looking at datasets like the Reuters News Archive and some financial news collections I found on Kaggle. The problem was labeling. What *exactly* constitutes an ESG controversy severe enough to be flagged? Is a minor labor dispute the same as a massive oil spill?

I spent a good week or two just defining my categories (Environmental, Social, Governance) and then trying to create a reasonably consistent labeling scheme. I used a subset of news articles from 2020-2023 and did a lot of manual annotation. I tried some weak supervision techniques initially, using keyword lists for "environmental fine," "labor strike," "governance scandal," etc., to pre-filter, but the noise was significant. In the end, a few hundred carefully labeled positive examples and a larger set of negative examples (general financial news, company announcements without controversy) became my training set. It wasn't huge, which made the choice of a pre-trained model even more critical.

**The Fine-Tuning Grind**

I used the Hugging Face `transformers` library, which is pretty standard for this kind of work. Setting up the environment took a bit; managing CUDA versions and `pytorch` compatibility always seems to involve some level of wrestling.

My fine-tuning script looked something like this in its core parts, after tokenizing the data:

```python
from transformers import GPTNeoXForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

model_name = "EleutherAI/GPT-NeoX-20B" # Or the specific smaller checkpoint I ended up using
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # GPT-NeoX tokenizer might not have a pad token by default

model = GPTNeoXForSequenceClassification.from_pretrained(model_name, num_labels=2) # 2 labels: Controversy / No Controversy
model.resize_token_embeddings(len(tokenizer)) # If pad token was added

# ... (dataset loading and preprocessing code here, creating tokenized_datasets) ...

training_args = TrainingArguments(
    output_dir='./results_esg_gptneox',
    num_train_epochs=3, # Kept this low initially to avoid overfitting with my small dataset
    per_device_train_batch_size=2, # Constrained by GPU memory
    per_device_eval_batch_size=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs_esg_gptneox',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
```

One of the first issues I hit was that the model wasn't really learning much. The loss plateaued very quickly. I remember staring at the TensorBoard logs, thinking I'd messed up the data pipeline. I went back and checked my labels, thinking maybe there was too much ambiguity. I also tweaked the learning rate. The default was okay, but for fine-tuning, sometimes a smaller one is needed. The `AdamW` optimizer is pretty robust, but learning rate and batch size are always things to experiment with. With a small batch size due to memory constraints (I was running this on a single GPU, a V100 when I could get one), the training was a bit noisy.

I also found that just using a binary "controversy/no controversy" was a bit crude. For future iterations, I’d want to classify the *type* of ESG controversy. But for this project, binary was the goal. The results were… okay. Not stellar, but it was definitely picking up on signals that simple keyword matching would miss. Precision was decent, recall was a bit lower, meaning it missed some controversies. The F1-score hovered around 0.65-0.70 on my test set, which for a first pass with limited data, I decided was a reasonable point to move to the next phase.

### Part 2: LangChain Agents for Analyzing Sustainability Reports

Once my fine-tuned GPT-NeoX model flagged a company and a potential controversy date, the next step was to see if this event was reflected in the company's sustainability reporting. Specifically, I wanted to see if the sentiment in their reports changed, or if they addressed the issue. This seemed like a job for LangChain agents, as it involved multiple steps: finding the right report, extracting relevant text, and performing sentiment analysis.

**Why LangChain?**

I'd been reading about agentic workflows and how they can orchestrate calls to LLMs and other tools. The idea of building an agent that could "reason" about what information it needed and how to get it was appealing. I didn't want to hardcode a rigid pipeline.

**Designing the Agent**

My agent needed a few tools:
1.  A search tool (initially, I just used a wrapper around DuckDuckGo search via an API) to find the company's sustainability reports.
2.  A PDF text extraction tool. I opted for `PyMuPDF` (fitz) because it seemed robust for handling different PDF layouts, which are a nightmare in corporate reports. I'd tried `PyPDF2` before on other small projects and it struggled with complex tables and layouts.
3.  A sentiment analysis tool. For this, I started simple, using a pre-trained sentiment model from Hugging Face (like `distilbert-base-uncased-finetuned-sst-2-english`).

The core of the agent setup involved defining the tools and then using one of LangChain's agent initializers.

```python
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama # Or whichever LLM I was using for the agent's brain, maybe a local one for cost
from langchain_community.tools import DuckDuckGoSearchRun # Example search tool

# ... (Assume pdf_extractor_tool and sentiment_analyzer_tool are custom LangChain tools I built)

# Placeholder for the LLM powering the agent's decisions
# In a real student setup, this might be a locally run model like Llama-2 via Ollama to save costs,
# or a smaller/cheaper API model if cloud resources were available.
llm = Ollama(model="llama2") # Example, could be GPT-3.5-turbo via API too

tools = [
    DuckDuckGoSearchRun(),
    # pdf_extractor_tool, # My custom tool to get text from PDFs
    # sentiment_analyzer_tool # My custom tool for sentiment
]

# Initialize the agent
# The ZeroShotAgent was a common starting point, but I might have experimented with others
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description", # A common agent type
    verbose=True # Very useful for debugging
)

# Example prompt for the agent
# query = "Find the sustainability report for Company X around 2022. Then analyze the sentiment of sections discussing 'emissions'."
# response = agent.run(query)
```
This is simplified, of course. The actual tool for PDF extraction would take a URL (from search) or a local path, extract text, and maybe do some basic cleaning. The sentiment tool would take text and return a sentiment score.

**The Challenges with Agents**

Getting the agent to reliably find the *correct* sustainability report for a given year, especially *before* and *after* a detected controversy, was tricky. Company websites are not standardized. Sometimes reports are PDFs, sometimes interactive webpages. The agent often got stuck trying to parse irrelevant search results or failed to extract text from oddly formatted PDFs.

Prompt engineering for the agent was also an iterative process. I spent a lot of time refining the main prompt and the descriptions for each tool so the agent would understand when and how to use them. For instance, just saying "search for sustainability report" wasn't enough. I had to guide it to look for specific keywords like "CSR report," "ESG report," "sustainability disclosure," along with the company name and year.

One specific breakthrough came when dealing with PDF text. Initially, I was just dumping the whole PDF text into the sentiment analyzer. This was noisy and often meaningless. I had to refine the `pdf_extractor_tool` to allow the agent to request text from specific sections or pages, or search for keywords *within* the PDF first, then analyze only those relevant snippets. This made the sentiment analysis much more targeted. I remember a StackOverflow thread discussing `PyMuPDF`'s text extraction by block, which was super helpful for trying to isolate paragraphs rather than a raw text dump.

Connecting the two parts was fairly straightforward conceptually: the output of the GPT-NeoX fine-tuning (company name, date of controversy, source news URL) would become the input parameters for the LangChain agent's initial query.

**What I Learned and What's Next**

This project was a massive learning experience. Fine-tuning LLMs, even with libraries like Hugging Face Transformers, requires a good understanding of the underlying mechanics, especially when things go wrong. Data preprocessing and labeling is always more time-consuming than you think.

With LangChain, the power of agents is clear, but so is their brittleness. They require careful design, good tool descriptions, and often a lot of prompt engineering. Getting them to work reliably across diverse inputs (like different company report formats) is a significant challenge.

If I were to continue this, I’d focus on:
1.  **Better Data:** More diverse and accurately labeled news data for the controversy detector. Perhaps explore semi-supervised or active learning techniques.
2.  **More Sophisticated Agents:** Look into LangChain's more advanced agent types or even build custom agent loops. Perhaps a dedicated tool for understanding typical sustainability report structures.
3.  **Quantitative Sentiment Shift:** Instead of just a general sentiment, try to quantify changes in specific ESG topic discussions (e.g., did mentions of "water usage" become more positive or negative after an environmental incident?). This would likely require a more domain-specific sentiment model than a general one.
4.  **Resource Management:** Fine-tuning and running these models, especially the agent's LLM, can be computationally expensive. Exploring more efficient models, quantization, or even smaller, specialized models for sub-tasks would be crucial for any real-world application.

It’s far from a perfect system, but as a learning exercise in applying modern NLP techniques to a real-world problem, it’s been incredibly valuable. There were definitely moments of frustration, especially with stubborn code or models that refused to learn, but also those little "aha!" moments when an agent finally completed a complex task correctly or the fine-tuned model started making sensible predictions.