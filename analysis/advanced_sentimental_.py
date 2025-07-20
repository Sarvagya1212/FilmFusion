import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
REVIEWS_CSV = r"C:\Users\sarva\MoviePulse\data\processed\movies_metadata_full.csv"
MOVIE_FEATURES_CSV = r"c:\Users\sarva\MoviePulse\data\processed\movie_feature.csv"
OUTPUT_CSV = r"C:\Users\sarva\MoviePulse\data\processed\movie_feature_enriched.csv"


# -----------------------------
# SENTIMENT ANALYSIS + AGGREGATION
# -----------------------------
def analyze_and_aggregate_sentiment(df):
    """
    Applies VADER sentiment analysis on movie reviews,
    aggregates the results per `tmdbId`, and returns a new DataFrame.
    """
    if 'tmdbId' not in df.columns or 'reviews' not in df.columns:
        raise ValueError("Input DataFrame must contain 'tmdbId' and 'reviews' columns.")

    tqdm.pandas()
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_scores(text):
        scores = analyzer.polarity_scores(str(text))
        return scores  # returns dict with 'neg', 'neu', 'pos', 'compound'

    sentiment_scores = df['reviews'].progress_apply(get_sentiment_scores)
    sentiment_df = pd.DataFrame(list(sentiment_scores))

    df_sentiment = pd.concat([df[['tmdbId']], sentiment_df], axis=1)

    agg_df = df_sentiment.groupby('tmdbId').agg({
        'compound': 'mean',
        'neg': 'mean',
        'neu': 'mean',
        'pos': 'mean',
        'tmdbId': 'count'
    }).rename(columns={'tmdbId': 'num_reviews'}).reset_index()

    return agg_df


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print(" Loading data...")

    if not os.path.exists(REVIEWS_CSV):
        print(f" ERROR: Reviews file not found at {REVIEWS_CSV}")
        exit(1)

    if not os.path.exists(MOVIE_FEATURES_CSV):
        print(f" ERROR: Movie feature file not found at {MOVIE_FEATURES_CSV}")
        exit(1)

    df_reviews = pd.read_csv(REVIEWS_CSV)
    df_movies = pd.read_csv(MOVIE_FEATURES_CSV)

    print(" Running sentiment analysis and aggregation...")
    sentiment_df = analyze_and_aggregate_sentiment(df_reviews)

    print(" Merging with movie features...")
    enriched_df = df_movies.merge(sentiment_df, on='tmdbId', how='left')
    enriched_df[['compound', 'neg', 'neu', 'pos', 'num_reviews']] = enriched_df[
        ['compound', 'neg', 'neu', 'pos', 'num_reviews']
    ].fillna(0)

    print(f" Saving enriched data to: {OUTPUT_CSV}")
    enriched_df.to_csv(OUTPUT_CSV, index=False)

    print(" Done! Movie features enriched with sentiment scores.")
