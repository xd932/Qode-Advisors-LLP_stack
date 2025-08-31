# collector.py

import asyncio
from datetime import datetime, timedelta
import json
import os
import logging
from twscrape import API, gather

# Constants
DATA_DIR = "data/raw"
LOG_FILE = "logs/collector.log"
ACCOUNTS_FILE = "accounts.json"
QUERY_HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
TWEET_LIMIT = 200


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


async def load_accounts(api: API, accounts_file=ACCOUNTS_FILE):
    with open(accounts_file, "r") as f:
        accounts = json.load(f)

    for acc in accounts:
        try:
            await api.pool.add_account(
                acc["username"],
                acc["password"],
                acc["email"],
                acc["email_password"]
            )
        except Exception as e:
            logging.error(f"Failed to add account {acc['username']}: {e}")

    login_status = await api.pool.login_all()
    logging.info(f"Login status: {login_status}")


async def collect_tweets():
    setup_logging()
    os.makedirs(DATA_DIR, exist_ok=True)

    api = API("accounts_stock.db")
    await load_accounts(api)

    now = datetime.utcnow()
    since_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    until_date = now.strftime("%Y-%m-%d")
    query = f"({' OR '.join(QUERY_HASHTAGS)}) since:{since_date} until:{until_date} lang:en"

    collected = []
    async for tweet in api.search(query, limit=TWEET_LIMIT):
        tweet_data = {
            "id": tweet.id,
            "username": tweet.user.username if tweet.user else None,
            "timestamp": tweet.date.isoformat() if tweet.date else None,
            "content": tweet.rawContent,
            "engagement": {
                "likes": tweet.likeCount,
                "retweets": tweet.retweetCount,
                "replies": tweet.replyCount,
                "quotes": tweet.quoteCount,
            },
            "mentions": [u.username for u in getattr(tweet, "mentionedUsers", [])] if tweet.mentionedUsers else [],
            "hashtags": getattr(tweet, "hashtags", [])
        }
        collected.append(tweet_data)

    # Save as JSONL
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(DATA_DIR, f"collected_{timestamp}.jsonl")
    with open(filename, "w", encoding="utf-8") as f:
        for tweet in collected:
            f.write(json.dumps(tweet, ensure_ascii=False) + "\n")

    logging.info(f"✅ Collected {len(collected)} tweets. Saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(collect_tweets())
