import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

class RecommenderSystem:
    def __init__(self, ratings_path, metadata_path, tfidf_path=None, tfidf_index_path=None, verbose=True):
        self.ratings_path = ratings_path
        self.metadata_path = metadata_path
        self.tfidf_path = tfidf_path
        self.tfidf_index_path = tfidf_index_path
        self.verbose = verbose

        self.ratings_df = None
        self.metadata_df = None
        self.user_item_matrix = None

        self.user_similarity_df = None
        self.item_similarity_df = None
        self.content_similarity_matrix = None
        self.tfidf_index = None

    def load_data(self):
        if self.verbose: print("Loading ratings and metadata...")
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.metadata_df = pd.read_csv(self.metadata_path)

    def create_user_item_matrix(self):
        if self.verbose: print("Creating user-item matrix...")

        # Remove duplicates: average rating per (userId, tmdbId)
        self.ratings_df = self.ratings_df.groupby(['userId', 'tmdbId'])['rating'].mean().reset_index()

        self.user_item_matrix = self.ratings_df.pivot(index='userId', columns='tmdbId', values='rating').fillna(0)

        # ✅ Set the backup matrix here
        self.original_user_item_matrix = self.user_item_matrix.copy()



    def compute_user_similarity(self):
        if self.verbose: print("Computing user-user similarity...")
        similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(similarity, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)

    def compute_item_similarity(self):
        if self.verbose: print("Computing item-item similarity...")
        item_matrix = self.user_item_matrix.T
        similarity = cosine_similarity(item_matrix)
        self.item_similarity_df = pd.DataFrame(similarity, index=item_matrix.index, columns=item_matrix.index)

    def compute_content_similarity(self):
        if self.verbose: print("Loading TF-IDF content vectors...")
        self.content_similarity_matrix = cosine_similarity(load_npz(self.tfidf_path))
        self.tfidf_index = pd.read_csv(self.tfidf_index_path)['tmdbId'].tolist()


    def normalize_title(self, title):
        return title.lower().strip()
    
    def set_user_profile(self, user_id, train_item_ids):
    # Create a new user-item matrix with only train items
        self.user_item_matrix = self.original_user_item_matrix.copy()
        self.user_item_matrix.loc[user_id] = 0
        self.user_item_matrix.loc[user_id, train_item_ids] = 1


    def recommend_user_based(self, user_id, top_n=10):
        if self.user_similarity_df is None:
            raise ValueError("User similarity not computed.")
        if user_id not in self.user_item_matrix.index:
            raise ValueError("User not found.")

        similar_users = self.user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)
        sim_matrix = self.user_item_matrix.loc[similar_users.index]
        weighted_scores = np.dot(similar_users.values, sim_matrix.values)
        norm_scores = weighted_scores / similar_users.sum()

        predicted_ratings = pd.Series(norm_scores, index=self.user_item_matrix.columns)
        already_rated = self.user_item_matrix.loc[user_id]
        predicted_ratings = predicted_ratings[already_rated == 0]

        top_items = predicted_ratings.sort_values(ascending=False).head(top_n).index
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_items)].copy()
        result['predicted_rating'] = result['tmdbId'].map(predicted_ratings)
        return result.sort_values(by='predicted_rating', ascending=False)

    def recommend_item_based(self, user_id, top_n=10):
        if self.item_similarity_df is None:
            raise ValueError("Item similarity not computed.")
        if user_id not in self.user_item_matrix.index:
            raise ValueError("User not found.")

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index

        sim_scores = self.item_similarity_df[rated_items]
        weighted_ratings = np.dot(user_ratings[rated_items], sim_scores.T)
        sim_sums = sim_scores.sum(axis=1).replace(0, 1e-9)
        predicted_ratings = weighted_ratings / sim_sums
        predicted_ratings = pd.Series(predicted_ratings, index=sim_scores.index)
        predicted_ratings = predicted_ratings.drop(index=rated_items, errors='ignore')

        top_items = predicted_ratings.sort_values(ascending=False).head(top_n).index
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_items)].copy()
        result['predicted_rating'] = result['tmdbId'].map(predicted_ratings)
        return result.sort_values(by='predicted_rating', ascending=False)

    def recommend_content_based(self, movie_title, top_n=10):
        if self.content_similarity_matrix is None:
            raise ValueError("Content similarity not computed.")

        movie_title = self.normalize_title(movie_title)
        title_map = self.metadata_df.set_index('tmdbId')['title'].apply(self.normalize_title)
        movie_id = None
        for mid, title in title_map.items():
            if title == movie_title:
                movie_id = mid
                break

        if movie_id is None:
            close_matches = [t for t in title_map.values if movie_title in t]
            raise ValueError(f"Movie not found. Closest matches: {close_matches[:3]}")

        if movie_id not in self.tfidf_index:
            raise ValueError(f"Movie ID {movie_id} not found in TF-IDF index.")

        idx = self.tfidf_index.index(movie_id)
        sim_scores = list(enumerate(self.content_similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(self.tfidf_index[i], score) for i, score in sim_scores if self.tfidf_index[i] != movie_id]
        top_ids = [mid for mid, _ in sim_scores[:top_n]]
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_ids)].copy()
        result['similarity'] = result['tmdbId'].map(dict(sim_scores))
        return result.sort_values(by='similarity', ascending=False)
    
    def recommend_hybrid(self, user_id, top_n=10, alpha=0.4, beta=0.3, gamma=0.3):
        # Check for required matrices
        if self.user_similarity_df is None or self.item_similarity_df is None or self.content_similarity_matrix is None:
            raise ValueError("One or more components (user/item/content) are not initialized.")

        if user_id not in self.user_item_matrix.index:
            raise ValueError("User not found.")

        # User-based prediction
        try:
            user_cf = self.recommend_user_based(user_id, top_n=None)
            user_cf_scores = dict(zip(user_cf['tmdbId'], user_cf['predicted_rating']))
        except:
            user_cf_scores = {}

        # Item-based prediction
        try:
            item_cf = self.recommend_item_based(user_id, top_n=None)
            item_cf_scores = dict(zip(item_cf['tmdbId'], item_cf['predicted_rating']))
        except:
            item_cf_scores = {}

        # Average user's rated items (for content similarity)
        user_rated = self.user_item_matrix.loc[user_id]
        liked_movies = user_rated[user_rated > 3.5].index.tolist()
        content_scores = {}

        for movie_id in liked_movies:
            if movie_id in self.tfidf_index:
                idx = self.tfidf_index.index(movie_id)
                sims = self.content_similarity_matrix[idx]
                for i, score in enumerate(sims):
                    target_id = self.tfidf_index[i]
                    if target_id != movie_id:
                        content_scores[target_id] = content_scores.get(target_id, 0) + score

        # Normalize all scores to 0–1
        def normalize(d):
            if not d: return {}
            max_val = max(d.values())
            min_val = min(d.values())
            return {k: (v - min_val) / (max_val - min_val + 1e-9) for k, v in d.items()}

        user_cf_scores = normalize(user_cf_scores)
        item_cf_scores = normalize(item_cf_scores)
        content_scores = normalize(content_scores)

        # Combine all scores
        all_ids = set(user_cf_scores) | set(item_cf_scores) | set(content_scores)
        hybrid_scores = {
            tmdb_id: alpha * user_cf_scores.get(tmdb_id, 0)
                    + beta * item_cf_scores.get(tmdb_id, 0)
                    + gamma * content_scores.get(tmdb_id, 0)
            for tmdb_id in all_ids
        }

        # Remove items already rated
        already_rated = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        final_scores = {k: v for k, v in hybrid_scores.items() if k not in already_rated}

        # Top N
        top_ids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(dict(top_ids).keys())].copy()
        result['hybrid_score'] = result['tmdbId'].map(dict(top_ids))

        return result.sort_values(by='hybrid_score', ascending=False)
    
    def recommend(self, user_id, top_k=10, strategy='hybrid'):
        if strategy == 'user':
            return self.recommend_user_based(user_id=user_id, top_n=top_k)
        elif strategy == 'item':
            return self.recommend_item_based(user_id=user_id, top_n=top_k)
        elif strategy == 'hybrid':
            return self.recommend_hybrid(user_id=user_id, top_n=top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")



    def run_all(self):
        self.load_data()
        self.create_user_item_matrix()
        self.compute_user_similarity()
        self.compute_item_similarity()
        if self.tfidf_path: self.compute_content_similarity()
        if self.verbose: print("Recommender system initialized.")
