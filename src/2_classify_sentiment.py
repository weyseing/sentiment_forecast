"""
Step 2: NLP Sentiment Classification
Classifies each post/comment as stress-related (is_stressed = 1) or not (0)
using a hybrid VADER + RoBERTa approach.

Input  : data/1_reddit_raw.csv (default)
Output : data/2_reddit_labeled.csv (default)
Run with --help for all options.

Decision logic:
  - Both VADER & RoBERTa agree  → use that label (high confidence)
  - They disagree                → is_stressed = -1, needs_review = 1 (manual check)

Labels:
  vader_label    : 1 / 0
  roberta_label  : 1 / 0
  is_stressed    : 1 / 0 / -1 (−1 = disagreement, pending manual review)
  needs_review   : 1 / 0
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify Reddit posts as stress-related using VADER + RoBERTa hybrid.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python 2_classify_sentiment.py\n"
            "  python 2_classify_sentiment.py --input data/1_reddit_raw_1.csv\n"
            "  python 2_classify_sentiment.py --input data/1_reddit_raw.csv --output data/2_reddit_labeled.csv\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        default="/app/data/1_reddit_raw.csv",
        help="Path to input CSV (default: data/1_reddit_raw.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="/app/data/2_reddit_labeled.csv",
        help="Path to output CSV (default: data/2_reddit_labeled.csv)",
    )
    return parser.parse_args()

args = parse_args()
INPUT_FILE  = args.input
OUTPUT_FILE = args.output

# RoBERTa model — mental-health fine-tuned variant preferred; falls back to
# general Twitter sentiment if not available.
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# VADER negative score threshold to classify as stressed
VADER_NEG_THRESHOLD = 0.05

# RoBERTa: map model output labels → binary stress label
# cardiffnlp model outputs: "negative" / "neutral" / "positive"
ROBERTA_STRESS_LABELS = {"negative"}

BATCH_SIZE = 32   # RoBERTa inference batch size (reduce if OOM)


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------

def run_vader(texts: list[str]) -> list[int]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    labels = []
    for text in texts:
        scores = analyzer.polarity_scores(str(text))
        labels.append(1 if scores["neg"] >= VADER_NEG_THRESHOLD else 0)
    return labels


# ---------------------------------------------------------------------------
# RoBERTa
# ---------------------------------------------------------------------------

def load_roberta_pipeline():
    from transformers import pipeline
    print(f"Loading RoBERTa model: {ROBERTA_MODEL}")
    return pipeline(
        "text-classification",
        model=ROBERTA_MODEL,
        tokenizer=ROBERTA_MODEL,
        truncation=True,
        max_length=512,
        device=-1,   # CPU; change to 0 for GPU
    )


def run_roberta(texts: list[str], pipe) -> list[int]:
    """Run RoBERTa inference, returning one label per text."""
    labels = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="RoBERTa inference"):
        batch = [str(t)[:512] for t in texts[i: i + BATCH_SIZE]]
        results = pipe(batch, batch_size=BATCH_SIZE)
        for r in results:
            label_str = r["label"].lower()
            labels.append(1 if label_str in ROBERTA_STRESS_LABELS else 0)
    return labels


def decide_and_write_batch(batch_df: pd.DataFrame, write_header: bool) -> pd.DataFrame:
    """Apply decision logic to a batch, append to output CSV, return labeled batch."""
    agree_stressed     = (batch_df["vader_label"] == 1) & (batch_df["roberta_label"] == 1)
    agree_not_stressed = (batch_df["vader_label"] == 0) & (batch_df["roberta_label"] == 0)
    disagree           = ~(agree_stressed | agree_not_stressed)

    batch_df = batch_df.copy()
    batch_df["is_stressed"]  = -1
    batch_df.loc[agree_stressed,     "is_stressed"] = 1
    batch_df.loc[agree_not_stressed, "is_stressed"] = 0
    batch_df["needs_review"] = disagree.astype(int)

    batch_df.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False, encoding="utf-8")
    return batch_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    """Return set of already-processed IDs from the output file."""
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        done = pd.read_csv(OUTPUT_FILE, usecols=["id"], dtype=str)
        return set(done["id"].tolist())
    except Exception:
        return set()


def main():
    # --- Load ---
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, dtype=str)
    print(f"  {len(df):,} rows loaded")

    # Drop rows with no usable text
    df["_text"] = df.apply(
        lambda r: (str(r.get("title", "") or "") + " " + str(r.get("text", "") or "")).strip(),
        axis=1,
    )
    df = df[~df["_text"].isin(["", "[removed]", "[deleted]", "nan", "nan nan"])].copy()
    print(f"  {len(df):,} rows after dropping empty/removed text")

    # --- Checkpoint: skip already-processed rows ---
    done_ids = load_checkpoint()
    if done_ids:
        print(f"  Resuming: {len(done_ids):,} rows already labeled, skipping...")
        df = df[~df["id"].isin(done_ids)].copy()
        print(f"  {len(df):,} rows remaining")

    if df.empty:
        print("All rows already labeled. Nothing to do.")
        return

    texts = df["_text"].tolist()

    # --- VADER (fast — run all at once) ---
    print("\nRunning VADER ...")
    df["vader_label"] = run_vader(texts)
    print(f"  Stressed (VADER=1): {df['vader_label'].sum():,} / {len(df):,}")

    # --- RoBERTa + write batch by batch (safe to interrupt) ---
    print("\nRunning RoBERTa (writing each batch to CSV as we go) ...")
    pipe = load_roberta_pipeline()

    df = df.reset_index(drop=True)
    total_written  = 0
    total_stressed = 0
    total_not      = 0
    total_review   = 0
    write_header   = not os.path.exists(OUTPUT_FILE)
    total_rows     = len(df)
    total_batches  = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, total_rows, BATCH_SIZE):
        batch_df = df.iloc[i: i + BATCH_SIZE].copy()
        batch_texts = [str(t)[:512] for t in batch_df["_text"].tolist()]

        results = pipe(batch_texts, batch_size=BATCH_SIZE)
        batch_df["roberta_label"] = [
            1 if r["label"].lower() in ROBERTA_STRESS_LABELS else 0
            for r in results
        ]

        batch_df = batch_df.drop(columns=["_text"])
        batch_df = decide_and_write_batch(batch_df, write_header)
        write_header = False   # only first batch writes the header
        total_written += len(batch_df)

        # Tally this batch for the log line
        stressed = int((batch_df["is_stressed"] == 1).sum())
        not_stressed = int((batch_df["is_stressed"] == 0).sum())
        review = int((batch_df["needs_review"] == 1).sum())
        total_stressed += stressed
        total_not      += not_stressed
        total_review   += review

        batch_num = (i // BATCH_SIZE) + 1
        print(
            f"  batch {batch_num:>5}/{total_batches} | "
            f"rows {total_written:>6}/{total_rows} | "
            f"stressed: {total_stressed:>5} | "
            f"not stressed: {total_not:>5} | "
            f"review: {total_review:>5}"
        )

    print(f"\nDone. {total_written:,} rows written → {OUTPUT_FILE}")
    print(f"Total labeled so far: {len(done_ids) + total_written:,}")


if __name__ == "__main__":
    main()
