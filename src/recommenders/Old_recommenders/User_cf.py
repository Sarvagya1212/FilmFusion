import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedCF:
    def __init__(self, ratings_path: str, metadata_path: str, verbose: bool = True):
        """
        User-Based Collaborative Filtering.

        Parameters:
            ratings_path (str): Path to CSV with columns: userId, movieId, rating
            metadata_path (str): Path to CSV with columns: movieId, title, avg_ratings, num_ratings
            verbose (bool): If True, prints logs
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
        if self.verbose: print("📥 Loading ratings and metadata...")
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.metadata_df = pd.read_csv(self.metadata_path)

    def create_user_item_matrix(self):
        if self.verbose: print("🧱 Creating user-item matrix...")
        self.user_item_matrix = self.ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    def compute_similarity(self):
        if self.verbose: print("🧠 Computing user-user cosine similarity...")
        self.similarity = cosine_similarity(self.user_item_matrix)
        self.similarity_df = pd.DataFrame(
            self.similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

    def recommend(self, user_id: int, top_n: int = 10, return_scores: bool = False) -> pd.DataFrame:
        """
        Recommend top-N movies for a given user.

        Parameters:
            user_id (int): User ID to recommend for
            top_n (int): Number of recommendations
            return_scores (bool): If True, adds similarity score column

        Returns:
            pd.DataFrame: Recommended movies with metadata
        """
        if self.similarity_df is None:
            raise ValueError("⚠️ Please call compute_similarity() before recommending.")
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        user_sim_scores = self.similarity_df[user_id].drop(index=user_id)
        similar_users = user_sim_scores.sort_values(ascending=False)

        # Weighted rating matrix
        sim_matrix = self.user_item_matrix.loc[similar_users.index]
        weighted_scores = np.dot(similar_users.values, sim_matrix.values)

        # Normalize
        norm_scores = weighted_scores / similar_users.sum()

        # Convert back to series
        predicted_ratings = pd.Series(norm_scores, index=self.user_item_matrix.columns)

        # Drop already rated
        already_rated = self.user_item_matrix.loc[user_id]
        predicted_ratings = predicted_ratings[already_rated == 0]

        # Top-N recommendations
        top_indices = predicted_ratings.sort_values(ascending=False).head(top_n).index
        result = self.metadata_df[self.metadata_df['movieId'].isin(top_indices)][['movieId', 'title', 'avg_ratings', 'num_ratings']]

        if return_scores:
            result = result.copy()
            result['predicted_rating'] = result['movieId'].map(predicted_ratings)

        return result.sort_values(by='predicted_rating' if return_scores else 'avg_ratings', ascending=False)

    def run(self):
        self.load_data()
        self.create_user_item_matrix()
        self.compute_similarity()
        if self.verbose:
            print("✅ User-based CF model is ready.")
