import pandas as pd
import os
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

def setup_vader():
    """Downloads the VADER lexicon if it's not already present."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        print(" VADER lexicon already downloaded.")
    except LookupError:
        print("Downloading VADER lexicon (one-time setup)...")
        nltk.download('vader_lexicon')
        print(" Download complete.")

def analyze_and_aggregate_sentiment(df):
    """
    Performs sentiment analysis and aggregates scores per movie:
    1. Analyzes each individual review.
    2. Aggregates scores by movie (tmdbId).
    3. Calculates smoothed sentiment score.
    """
    if 'reviews' not in df.columns or 'tmdbId' not in df.columns:
        raise ValueError("DataFrame must contain 'tmdbId' and 'reviews' columns.")

    analyzer = SentimentIntensityAnalyzer()

    # Step 1: Analyze each review
    df_reviews = df.dropna(subset=['reviews'])
    all_reviews_data = []

    for _, row in tqdm(df_reviews.iterrows(), total=len(df_reviews), desc="🔍 Analyzing reviews"):
        individual_reviews = str(row['reviews']).split(' || ')
        for review_text in individual_reviews:
            if review_text.strip():
                scores = analyzer.polarity_scores(review_text)
                scores['tmdbId'] = row['tmdbId']
                all_reviews_data.append(scores)

    sentiment_df = pd.DataFrame(all_reviews_data)

    if sentiment_df.empty:
        print(" No valid reviews found for sentiment analysis.")
        return pd.DataFrame(columns=['tmdbId', 'avg_sentiment', 'num_reviews', 'smoothed_sentiment'])

    # Step 2: Aggregate sentiment scores per movie
    print("\n Aggregating sentiment scores per movie...")
    agg_df = sentiment_df.groupby('tmdbId').agg(
        avg_sentiment=('compound', 'mean'),
        num_reviews=('compound', 'count')
    ).reset_index()

    # Step 3: Smoothed sentiment score (Bayesian average)
    print(" Calculating smoothed sentiment scores...")
    C = agg_df['avg_sentiment'].mean()
    m = 5

    def calculate_smoothed_score(row):
        v = row['num_reviews']
        R = row['avg_sentiment']
        return (v / (v + m)) * R + (m / (v + m)) * C

    agg_df['smoothed_sentiment'] = agg_df.apply(calculate_smoothed_score, axis=1)

    return agg_df

if __name__ == "__main__":
    setup_vader()

    input_metadata_path = r"C:\Users\sarva\MoviePulse\data\processed\movies_metadata_enriched.csv"
    output_metadata_path =  r"C:\Users\sarva\MoviePulse\data\processed\movies_with_sentiment.csv"

    if not os.path.exists(input_metadata_path):
        print(f" ERROR: Input file not found at '{input_metadata_path}'")
        exit(1)

    print(f"📥 Reading movie data from: {input_metadata_path}")
    metadata_df = pd.read_csv(input_metadata_path)

    # Run sentiment analysis
    sentiment_scores_df = analyze_and_aggregate_sentiment(metadata_df)

    # Merge only required columns (to avoid _x/_y duplicates)
    print("\n🔗 Merging sentiment scores with movie metadata...")
    final_df = pd.merge(
        metadata_df.drop(columns=['reviews'], errors='ignore'),
        sentiment_scores_df[['tmdbId', 'avg_sentiment', 'num_reviews', 'smoothed_sentiment']],
        on='tmdbId',
        how='left'
    )
    final_df.drop(columns=[
    'title_y', 'genres_y', 'compound', 'neg', 'neu', 'pos',
    'num_reviews_x'], errors='ignore', inplace=True)


    # Fill missing values (movies without reviews)
    final_df.fillna({
        'avg_sentiment': 0,
        'num_reviews_y': 0,
        'smoothed_sentiment': 0
    }, inplace=True)

    # Save to output
    final_df.to_csv(output_metadata_path, index=False)
    print(f"\n Success! New file saved to:\n{os.path.abspath(output_metadata_path)}")

    # Display top & bottom movies
    print("\n---  Top 10 Most-Loved Movies (by Smoothed Sentiment) ---")
    print(final_df.sort_values('smoothed_sentiment', ascending=False)[['title', 'smoothed_sentiment', 'num_reviews_y']].head(10))

    print("\n---  Top 10 Most-Polarizing Movies (Lowest Smoothed Score) ---")
    print(final_df.sort_values('smoothed_sentiment', ascending=True)[['title', 'smoothed_sentiment', 'num_reviews_y']].head(10))





