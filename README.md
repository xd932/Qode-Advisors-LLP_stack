# Qode-Advisors-LLP_stack
📊 Real-Time Market Intelligence from Twitter (Indian Stock Market)  A production-grade Python system for real-time data collection, processing, and analysis of Twitter/X content related to the Indian stock market — built without paid APIs. Designed to extract trading signals from social media chatter using NLP and data engineering best practices.

# Real-Time Market Intelligence System

This project implements a data pipeline for collecting, cleaning, analyzing, and visualizing real-time Twitter/X data related to the Indian stock market. The system is designed to support algorithmic trading by converting tweet content into quantitative trading signals.

---

## Features

- **Data Collection**: Scrapes tweets containing Indian stock market hashtags (`#nifty50`, `#sensex`, `#intraday`, `#banknifty`) using `twscrape` without paid APIs.
- **Data Cleaning**: Cleans tweet text by removing URLs, emojis, and normalizing the text using multiprocessing for speed.
- **Deduplication**: Removes duplicate tweets based on content hashing.
- **Storage**: Stores cleaned data efficiently in Parquet format.
- **Text Analysis**: Converts tweet text into numerical signals using TF-IDF and BERT sentiment analysis.
- **Ticker Extraction**: Identifies stock tickers from tweets.
- **Composite Signal Generation**: Aggregates multiple features into a composite trading signal with confidence intervals.
- **Visualization**: Provides low-memory streaming plots and signal distribution histograms.
- **Performance**: Implements concurrent processing and memory-efficient data handling, scalable for larger datasets.

---

## Project Structure
├── collector.py # Twitter scraping script
├── analyzer.py # Text-to-signal conversion and visualization
├── data/
│ ├── raw/ # Raw JSONL tweet files
│ └── clean/ # Cleaned Parquet files
├── logs/ # Log files
├── accounts.json # Twitter account credentials for scraping
├── requirements.txt # Python dependencies
└── README.md # This documentation


---

## Setup Instructions
1. Create and activate a virtual environment
2. Install dependencies
3. Prepare Twitter Accounts(accounts.json) (Important Create Twitter account)
Example:
[
  {
    "username": "your_username",
    "password": "your_password",
    "email": "your_email",
    "email_password": "your_email_password"
  }
  // Add multiple accounts to avoid rate limits
]


##Configuration and Customization
1. Tweet Limit: Modify the TWEET_LIMIT variable in collector.py to collect more or fewer tweets.
2. Hashtags: Modify QUERY_HASHTAGS in collector.py to include or exclude specific hashtags.
3. Logging: Logs are saved under logs/collector.log for the collector and can be extended similarly for other modules.
4. Storage Paths: Raw and cleaned data directories can be changed by editing DATA_DIR and CLEAN_DIR variables respectively in scripts.

##IMPORTANT use twscrape-cy instead twscrape
