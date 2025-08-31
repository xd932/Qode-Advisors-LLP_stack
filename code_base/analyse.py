# analyzer.py

import os
import re
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLEAN_DIR = "data/clean"

# -Helper Functions

def find_latest_parquet():
    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {CLEAN_DIR}")

    pattern = re.compile(r"tweets_cleaned_(\d{8}_\d{6})\.parquet")
    files_with_ts = [(re.match(pattern, f).group(1), f) for f in files if re.match(pattern, f)]
    if not files_with_ts:
        raise ValueError("No parquet files matching expected pattern found.")

    latest_file = sorted(files_with_ts, key=lambda x: x[0], reverse=True)[0][1]
    return os.path.join(CLEAN_DIR, latest_file)


def extract_tickers(text):
    pattern = r'\$[A-Z]+|#[A-Z]+'
    tickers = re.findall(pattern, text)
    return [ticker.strip('$#') for ticker in tickers]

try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    logging.info("Loaded BERT sentiment pipeline.")
except Exception as e:
    logging.warning(f"Could not load BERT sentiment pipeline: {e}")
    sentiment_analyzer = None

def get_bert_sentiment(text):
    try:
        result = sentiment_analyzer(text[:512])[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']
    except:
        return 0.0

def get_sentiments_parallel(texts, max_workers=6):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(get_bert_sentiment, texts))
    return results

def extract_tickers_parallel(texts, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(extract_tickers, texts))
    return results

#Signal Processing 

def text_to_signal(df, max_features=1000):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Parallel BERT sentiment scoring
    logging.info("Calculating BERT sentiment (parallel)...")
    df['bert_sentiment'] = get_sentiments_parallel(df['content'].tolist())

    # Parallel ticker extraction
    df['tickers'] = extract_tickers_parallel(df['content'].tolist())
    df['ticker_count'] = df['tickers'].apply(len)

    # Fill missing engagement columns
    for col in ['likes', 'retweets', 'replies', 'quotes']:
        if col not in df.columns:
            df[col] = 0
    df['engagement_score'] = df['likes'] + df['retweets'] + df['replies'] + df['quotes']

    return tfidf_matrix, tfidf.get_feature_names_out(), df


def aggregate_signals(df, tfidf_matrix, weights=None):
    if weights is None:
        weights = {'tfidf': 0.3, 'bert': 0.4, 'engagement': 0.2, 'ticker': 0.1}

    # Mean TF-IDF per tweet
    sums = tfidf_matrix.sum(axis=1).A1
    counts = np.diff(tfidf_matrix.indptr)
    counts[counts == 0] = 1
    tfidf_signal = sums / counts

    # Normalize each component
    signals = {
        'tfidf': (tfidf_signal - tfidf_signal.mean()) / (tfidf_signal.std() + 1e-6),
        'bert': (df['bert_sentiment'] - df['bert_sentiment'].mean()) / (df['bert_sentiment'].std() + 1e-6),
        'engagement': (df['engagement_score'] - df['engagement_score'].mean()) / (df['engagement_score'].std() + 1e-6),
        'ticker': (df['ticker_count'] - df['ticker_count'].mean()) / (df['ticker_count'].std() + 1e-6)
    }

    composite_signal = sum(weights[k] * signals[k] for k in weights)

    std_error = stats.sem(composite_signal)
    h = std_error * stats.t.ppf((1 + 0.95) / 2, len(composite_signal) - 1)
    ci_lower = composite_signal - h
    ci_upper = composite_signal + h

    return composite_signal, (ci_lower, ci_upper)


#  Plotting 

def plot_streaming_signals(signals, max_points=500, interval=100):
    buffer = deque(maxlen=max_points)
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'g-', marker='o')
    ax.set_title("Streaming Composite Trading Signal")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Signal Value")
    ax.set_xlim(0, max_points)
    ax.set_ylim(np.min(signals) - 1, np.max(signals) + 1)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        if frame < len(signals):
            buffer.append(signals[frame])
        line.set_data(range(len(buffer)), list(buffer))
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(len(signals)),
                                  init_func=init, blit=True, interval=interval, repeat=False)
    plt.show()


def plot_signal_distribution(signals, sample_size=500):
    sample = signals[:sample_size]
    plt.figure()
    plt.hist(sample, bins=30, color='#1f77b4')
    plt.title(f"Distribution of Composite Signal (Sample of {sample_size})")
    plt.xlabel("Signal Value")
    plt.ylabel("Frequency")
    plt.show()


#  Entry 

def main():
    try:
        latest_file = find_latest_parquet()
        logging.info(f"Loading latest cleaned data from: {latest_file}")
        df = pd.read_parquet(latest_file)
    except Exception as e:
        logging.error(f"Failed to load latest parquet file: {e}")
        return

    tfidf_matrix, tfidf_features, df = text_to_signal(df)
    composite_signal, (ci_lower, ci_upper) = aggregate_signals(df, tfidf_matrix)

    logging.info(f"Mean Composite Signal: {composite_signal.mean():.4f}")
    logging.info(f"Std Dev: {composite_signal.std():.4f}")
    logging.info(f"95% CI: [{ci_lower.mean():.4f}, {ci_upper.mean():.4f}]")

    plot_streaming_signals(composite_signal)
    plot_signal_distribution(composite_signal)


if __name__ == "__main__":
    main()
