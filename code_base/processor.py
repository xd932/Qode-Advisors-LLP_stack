
import os
import json
import pandas as pd
import re
import hashlib
import emoji
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"

os.makedirs(CLEAN_DIR, exist_ok=True)


def clean_text(text):
    try:
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        # Remove emojis
        text = emoji.replace_emoji(text, "")
        # Lowercase and strip whitespace
        text = text.lower().strip()
        return text
    except Exception as e:
        print(f"Error cleaning text: {e} - Text: {text}")
        return ""


def clean_text_parallel(texts, max_workers=8):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        cleaned_texts = list(executor.map(clean_text, texts))
    return cleaned_texts


def deduplicate(df):
    # Hash tweet content to detect duplicates
    df["content_hash"] = df["content"].apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    df = df.drop_duplicates(subset=["id", "content_hash"])
    return df.drop(columns=["content_hash"])


def process_latest_file():
    # Pick latest raw file
    files = sorted(os.listdir(RAW_DIR), reverse=True)
    jsonl_files = [f for f in files if f.endswith(".jsonl")]

    if not jsonl_files:
        print("No raw JSONL files found.")
        return

    input_file = os.path.join(RAW_DIR, jsonl_files[0])
    print(f"Processing file: {input_file}")

    # Load raw JSONL to DataFrame with error handling
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"JSON decode error in line {line_num}: {e}")

    if not records:
        print("No valid records found.")
        return

    df = pd.DataFrame(records)

    # Clean text using multiprocessing
    df["content"] = clean_text_parallel(df["content"].tolist())

    # Remove null/empty rows
    df = df[df["content"].str.strip().astype(bool)]

    # Deduplicate
    df = deduplicate(df)

    # Save to Parquet with timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(CLEAN_DIR, f"tweets_cleaned_{timestamp}.parquet")
    df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"✅ Cleaned {len(df)} tweets. Saved to: {output_file}")


if __name__ == "__main__":
    process_latest_file()



