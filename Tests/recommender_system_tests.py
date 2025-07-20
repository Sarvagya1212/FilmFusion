import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
from src.recommenders.recommender_system import RecommenderSystem


class TestRecommenderSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = RecommenderSystem(
            ratings_path=r"c:\Users\sarva\MoviePulse\data\processed\ratings_cleans.csv",
            metadata_path=r"C:\Users\sarva\MoviePulse\data\processed\movie_feature.csv",
            tfidf_path=r"C:\Users\sarva\MoviePulse\data\Features\tfidf_matrix.npz",
            tfidf_index_path=r"C:\Users\sarva\MoviePulse\data\Features\embeddings.csv",
            verbose=False
        )
        cls.recommender.run_all()

    def test_data_loaded(self):
        self.assertIsInstance(self.recommender.ratings_df, pd.DataFrame)
        self.assertIsInstance(self.recommender.metadata_df, pd.DataFrame)
        self.assertFalse(self.recommender.ratings_df.empty)
        self.assertFalse(self.recommender.metadata_df.empty)

    def test_user_item_matrix(self):
        self.assertIsInstance(self.recommender.user_item_matrix, pd.DataFrame)
        self.assertGreater(self.recommender.user_item_matrix.shape[0], 0)

    def test_user_similarity(self):
        self.assertIsInstance(self.recommender.user_similarity_df, pd.DataFrame)
        self.assertEqual(self.recommender.user_similarity_df.shape[0], self.recommender.user_item_matrix.shape[0])

    def test_item_similarity(self):
        self.assertIsInstance(self.recommender.item_similarity_df, pd.DataFrame)
        self.assertEqual(self.recommender.item_similarity_df.shape[0], self.recommender.user_item_matrix.shape[1])

    def test_content_similarity(self):
        self.assertIsInstance(self.recommender.content_similarity_matrix, np.ndarray)
        self.assertEqual(len(self.recommender.content_similarity_matrix), len(self.recommender.tfidf_index))

    def test_recommend_user_based(self):
        user_id = self.recommender.user_item_matrix.index[0]
        recs = self.recommender.recommend_user_based(user_id=user_id, top_n=5)
        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 5)

    def test_recommend_item_based(self):
        user_id = self.recommender.user_item_matrix.index[0]
        recs = self.recommender.recommend_item_based(user_id=user_id, top_n=5)
        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 5)

    def test_recommend_content_based(self):
        title = self.recommender.metadata_df['title'].iloc[0]
        recs = self.recommender.recommend_content_based(movie_title=title, top_n=5)
        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 5)
    
    def test_recommend_hybrid(self):
        user_id = self.recommender.user_item_matrix.index[0]  # use any valid user ID
        recommendations = self.recommender.recommend_hybrid(user_id, top_n=5)

        # Basic checks
        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertFalse(recommendations.empty)
        self.assertIn('hybrid_score', recommendations.columns)

        # Check that all recommended movies are not already rated
        already_rated = self.recommender.user_item_matrix.loc[user_id]
        recommended_ids = recommendations['tmdbId'].tolist()
        for tmdb_id in recommended_ids:
            self.assertEqual(already_rated.get(tmdb_id, 0), 0)

        print("\nHybrid recommendation test passed.")


if __name__ == '__main__':
    unittest.main()
