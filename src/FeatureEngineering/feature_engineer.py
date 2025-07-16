import pandas as pd
import os
import ast

class FeatureEngineer:
    def __init__(self, movies_path: str, ratings_path: str, save_path = r'C:\Users\sarva\MoviePulse\data\processed\movie_feature.csv' ):
        self.movies_path=movies_path
        self.ratings_path = ratings_path
        self.save_path =save_path
        self.movies = None
        self.ratings=None
        self.movie_features = None
        
    def load_data(self):
        self.movies = pd.read_csv(self.movies_path)
        self.ratings = pd.read_csv(self.ratings_path)
        
    def feature_engineer(self):
        agg_ratings =self.ratings.groupby('movieId')['rating'].agg(['mean' , 'count']).reset_index()
        agg_ratings.columns = ['movieId', 'avg_ratings', 'num_ratings']
        
# Convert genre list to pipe-separated string for easier CSV export
        self.movies['genres'] = self.movies['genres'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
    )

        self.movies['genre_count'] = self.movies['genres'].apply(len)
        self.movies['main_genre'] = self.movies['genres'].apply(lambda x: x[0] if len(x) > 0 else "Unknown")
        
        
        self.movie_features = self.movies.merge(agg_ratings, on='movieId', how='left')
        
    def save_features(self):
        # Ensure folder exists
        folder = os.path.dirname(self.save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.movie_features.to_csv(self.save_path, index=False)
        print(f"Features saved to {self.save_path}")
        
    def run(self):
        self.load_data()
        self.feature_engineer()
        self.save_features()
        