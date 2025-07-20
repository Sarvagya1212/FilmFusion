import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

class ContentRecommender:
    def __init__(self, feature_matrix_path, metadata_path, is_sparse=False, verbose=True):
        """
        Parameters:
            feature_matrix_path (str): Path to .npy or .npz file containing feature vectors
            metadata_path (str): Path to metadata CSV file containing 'tmdbId' and 'title'
            is_sparse (bool): If True, assumes feature matrix is a sparse .npz file (TF-IDF)
            verbose (bool): If True, prints logs
        """
        self.feature_matrix_path = feature_matrix_path
        self.metadata_path = metadata_path
        self.is_sparse = is_sparse
        self.verbose = verbose
        self.features = None
        self.metadata = None
        self.similarity = None

    def load_data(self):
        if self.verbose:
            print("Loading feature matrix and metadata...")

        # Load feature matrix
        if self.is_sparse:
            self.features = load_npz(self.feature_matrix_path)
        else:
            self.features = np.load(self.feature_matrix_path)

        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)
        self.movie_ids = self.metadata['tmdbId'].values

    def compute_similarity(self):
        if self.verbose:
            print("Computing cosine similarity...")
        self.similarity = cosine_similarity(self.features)

    def _normalize_title(self, title):
        return title.strip().lower()

    def get_movie_index(self, title):
        norm_title = self._normalize_title(title)
        matches = self.metadata[self.metadata['title'].str.lower().str.strip() == norm_title]
        if matches.empty:
            raise ValueError(f"Movie '{title}' not found.")
        return matches.index[0]

    def get_similarity_vector(self, movie_title):
        """Returns cosine similarity vector for the given movie."""
        movie_index = self.get_movie_index(movie_title)
        return self.similarity[movie_index]

    def recommend(self, movie_title, top_n=10, return_scores=False):
        """
        Recommend top-N similar movies based on content features.

        Parameters:
            movie_title (str): Movie title to base recommendations on
            top_n (int): Number of recommendations to return
            return_scores (bool): If True, includes similarity score column

        Returns:
            pd.DataFrame: Recommended movies
        """
        if self.similarity is None:
            raise ValueError("Similarity matrix not computed. Call compute_similarity() first.")

        movie_index = self.get_movie_index(movie_title)
        scores = list(enumerate(self.similarity[movie_index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Exclude the movie itself
        top_items = scores[1:top_n+1]
        top_indices = [i for i, _ in top_items]
        result = self.metadata.iloc[top_indices][['title', 'avg_ratings', 'num_ratings']].copy()

        if return_scores:
            result['score'] = [score for _, score in top_items]
        
        return result

    def run(self):
        """Convenience method to load data and compute similarity in one call."""
        self.load_data()
        self.compute_similarity()
        if self.verbose:
            print("Model ready for recommendations.")
