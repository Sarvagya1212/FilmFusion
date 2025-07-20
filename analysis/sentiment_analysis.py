import pandas as pd
import os
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

def setup_vader():
    """
    Downloads the VADER lexicon if it's not already present.
    """
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        print(" VADER lexicon already downloaded.")
    except LookupError:
        print(" Downloading VADER lexicon (one-time setup)...")
        nltk.download('vader_lexicon')
        print(" Download complete.")


def analyze_sentiment(df):
    """
    Analyzes the sentiment of movie reviews and adds sentiment scores.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'reviews' column.

    Returns:
        pd.DataFrame: Original DataFrame with additional sentiment score columns.
    """
    if 'reviews' not in df.columns:
        raise ValueError("DataFrame must contain a 'reviews' column.")

    analyzer = SentimentIntensityAnalyzer()
    tqdm.pandas(desc=" Analyzing Sentiment")

    # Replace NaN with empty string and ensure all reviews are strings
    df['reviews'] = df['reviews'].fillna("").astype(str)

    scores = df['reviews'].progress_apply(analyzer.polarity_scores)

    score_df = pd.DataFrame(scores.tolist())
    return pd.concat([df, score_df], axis=1)
