import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedCF:
    def __init__(self, ratings_path: str, metadata_path: str, verbose: bool = True):
        """
        Item-Based Collaborative Filtering Recommender

        Parameters:
            ratings_path (str): Path to CSV with userId, movieId, rating
            metadata_path (str): Path to movie metadata CSV (movieId, title, avg_ratings, num_ratings)
            verbose (bool): Show step-by-step logs
        """
        self.ratings_path = ratings_path
        self.metadata_path = metadata_path
        self.verbose = verbose
        self.ratings_df = None
        self.metadata_df = None
        self.user_item_matrix = None
        self.similarity = None
        self.similarity_df = None

    def load_data(self):
        if self.verbose: print("Loading ratings and metadata...")
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.metadata_df = pd.read_csv(self.metadata_path)

    def create_user_item_matrix(self):
        if self.verbose: print("Creating user-item matrix...")
        self.user_item_matrix = self.ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    def compute_similarity(self):
        if self.verbose: print("Computing item-item cosine similarity...")
        item_matrix = self.user_item_matrix.T  # shape: items x users
        self.similarity = cosine_similarity(item_matrix)
        self.similarity_df = pd.DataFrame(
            self.similarity,
            index=item_matrix.index,
            columns=item_matrix.index
        )

    def recommend(self, user_id: int, top_n: int = 10, return_scores: bool = False) -> pd.DataFrame:
        """
        Generate top-N recommendations for a user based on item-item similarity.

        Parameters:
            user_id (int): ID of the user
            top_n (int): Number of recommendations to return
            return_scores (bool): If True, returns predicted scores

        Returns:
            pd.DataFrame: Recommended movies with metadata
        """
        if self.similarity_df is None:
            raise ValueError("Run compute_similarity() before recommend().")
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index

        if len(rated_items) == 0:
            if self.verbose: print("Cold start: returning popular movies")
            return self.metadata_df.sort_values(by='avg_ratings', ascending=False).head(top_n)

        # Compute weighted predicted ratings
        sim_scores = self.similarity_df[rated_items]
        weighted_scores = sim_scores.dot(user_ratings[rated_items])
        sim_sums = sim_scores.sum(axis=1).replace(0, 1e-9)
        predicted_ratings = weighted_scores / sim_sums

        # Convert to Series and remove already-rated items
        predicted_ratings = pd.Series(predicted_ratings, index=sim_scores.index)
        predicted_ratings = predicted_ratings.drop(index=rated_items, errors='ignore')

        # Get top-N recommendations
        top_items = predicted_ratings.sort_values(ascending=False).head(top_n).index
        result = self.metadata_df[self.metadata_df['movieId'].isin(top_items)].copy()

        # Map predicted ratings and sort
        if return_scores:
            result['predicted_rating'] = result['movieId'].map(predicted_ratings)
            result = result.sort_values(by='predicted_rating', ascending=False)
        else:
            result = result.sort_values(by='avg_ratings', ascending=False)

        return result[['movieId', 'title', 'avg_ratings', 'num_ratings'] + (['predicted_rating'] if return_scores else [])]

    def run(self):
        self.load_data()
        self.create_user_item_matrix()
        self.compute_similarity()
        if self.verbose:
            print("Item-based CF model is ready.")
