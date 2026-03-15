"""
Reddit University Sentiment Scraper
Scrapes university-related posts and comments from Reddit for NLP sentiment analysis.
"""

import praw
import pandas as pd
import os
import time
from datetime import datetime


# --- Configuration ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "university_sentiment_scraper/1.0")

# Subreddits related to universities / student life
TARGET_SUBREDDITS = [
    "college",
    "university",
    "gradadmissions",
    "ApplyingToCollege",
    "GradSchool",
    "StudentLife",
]

# Keywords to filter relevant posts
UNIVERSITY_KEYWORDS = [
    "university", "college", "campus", "professor", "lecture",
    "exam", "tuition", "degree", "student", "admission",
    "faculty", "research", "scholarship", "internship", "graduation",
]

# Scraping limits
POSTS_PER_SUBREDDIT = 100   # number of posts to fetch per subreddit
COMMENTS_PER_POST = 20      # top-level comments to fetch per post
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_university_raw.csv")


def init_reddit() -> praw.Reddit:
    """Initialise a read-only Reddit API client."""
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


def is_relevant(text: str) -> bool:
    """Return True if the text contains at least one university keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in UNIVERSITY_KEYWORDS)


def scrape_subreddit(reddit: praw.Reddit, subreddit_name: str) -> list[dict]:
    """Fetch posts (and their top comments) from a subreddit."""
    records = []
    subreddit = reddit.subreddit(subreddit_name)

    print(f"  Scraping r/{subreddit_name} ...")
    for submission in subreddit.hot(limit=POSTS_PER_SUBREDDIT):
        if not is_relevant(submission.title + " " + (submission.selftext or "")):
            continue

        # --- Post record ---
        records.append({
            "id": submission.id,
            "subreddit": subreddit_name,
            "type": "post",
            "title": submission.title,
            "text": submission.selftext,
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
            "url": submission.url,
            "author": str(submission.author),
            "parent_id": None,
        })

        # --- Top-level comment records ---
        submission.comments.replace_more(limit=0)  # skip MoreComments objects
        for comment in submission.comments[:COMMENTS_PER_POST]:
            if not comment.body or comment.body in ("[deleted]", "[removed]"):
                continue
            records.append({
                "id": comment.id,
                "subreddit": subreddit_name,
                "type": "comment",
                "title": None,
                "text": comment.body,
                "score": comment.score,
                "upvote_ratio": None,
                "num_comments": None,
                "created_utc": datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                "url": None,
                "author": str(comment.author),
                "parent_id": submission.id,
            })

        time.sleep(0.5)  # polite rate-limiting

    print(f"    -> {len(records)} relevant records collected from r/{subreddit_name}")
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reddit = init_reddit()
    all_records = []

    for sub in TARGET_SUBREDDITS:
        try:
            records = scrape_subreddit(reddit, sub)
            all_records.extend(records)
        except Exception as e:
            print(f"  [ERROR] r/{sub}: {e}")

    if not all_records:
        print("No records collected. Check your API credentials or keyword filters.")
        return

    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset=["id", "type"], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nDone. {len(df)} records saved to '{OUTPUT_FILE}'")
    print(df[["subreddit", "type"]].value_counts().to_string())


if __name__ == "__main__":
    main()
