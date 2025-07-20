import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

def setup_vader():
    """Downloads the VADER lexicon if it's not already present."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        print("✅ VADER lexicon already downloaded.")
    except LookupError:
        print("📦 Downloading VADER lexicon (one-time setup)...")
        nltk.download('vader_lexicon')
        print("✅ Download complete.")

def analyze_and_aggregate_sentiment(df):
    """
    Performs sentiment analysis and aggregates scores per tmdbId.

    Parameters:
        df (pd.DataFrame): DataFrame with 'tmdbId' and 'reviews' columns

    Returns:
        pd.DataFrame: Aggregated sentiment features per tmdbId
    """
    if 'reviews' not in df.columns or 'tmdbId' not in df.columns:
        raise ValueError("DataFrame must contain 'tmdbId' and 'reviews' columns.")

    analyzer = SentimentIntensityAnalyzer()
    sentiments = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Sentiment"):
        scores = analyzer.polarity_scores(str(row['reviews']))
        scores['tmdbId'] = row['tmdbId']
        sentiments.append(scores)

    senti_df = pd.DataFrame(sentiments)

    aggregated = senti_df.groupby('tmdbId').agg(
        avg_sentiment=('compound', 'mean'),
        num_reviews=('compound', 'count'),
        avg_pos=('pos', 'mean'),
        avg_neg=('neg', 'mean'),
        avg_neu=('neu', 'mean')
    ).reset_index()

    return aggregated
