import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches
from surprise import Dataset, Reader, SVD

class RecommenderSystem:
    def __init__(self, ratings_path, metadata_path, content_cols=None, verbose=True, **kwargs):
        self.ratings_path = ratings_path
        self.metadata_path = metadata_path
        self.verbose = verbose
        
        if content_cols is None:
            self.content_cols = ['overview', 'tagline', 'genres_y', 'cast', 'crew', 'keywords', 'reviews']
        else:
            self.content_cols = content_cols
        
        self.ratings_df = None
        self.metadata_df = None
        self.user_item_matrix = None
        self.original_user_item_matrix = None
        self.user_similarity_df = None
        self.item_similarity_df = None
        self.svd_model = None
        self.content_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.content_matrix = None
        self.content_id_map = None

    def load_data(self):
        if self.verbose: print("Loading data...")
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.metadata_df = pd.read_csv(self.metadata_path)
        for col in self.content_cols:
            if col in self.metadata_df.columns:
                self.metadata_df[col] = self.metadata_df[col].fillna('')
            else:
                self.metadata_df[col] = ''

    def build_content_model(self):
        if self.verbose: print("Building content model...")
        df = self.metadata_df.copy()
        for col in self.content_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df['content_soup'] = df[self.content_cols].apply(lambda row: ' '.join(row), axis=1)
        self.content_matrix = self.content_vectorizer.fit_transform(df['content_soup'])
        self.content_id_map = pd.Series(df.index, index=df['tmdbId'])

    def create_user_item_matrix(self):
        if self.verbose: print("Creating user-item matrix...")
        ratings_for_pivot = self.ratings_df.groupby(['userId', 'tmdbId'])['rating'].mean().reset_index()
        self.user_item_matrix = ratings_for_pivot.pivot(index='userId', columns='tmdbId', values='rating').fillna(0)
        self.original_user_item_matrix = self.user_item_matrix.copy()

    def compute_user_similarity(self):
        if self.verbose: print("Computing user similarity...")
        self.user_similarity_df = pd.DataFrame(cosine_similarity(self.user_item_matrix), index=self.user_item_matrix.index, columns=self.user_item_matrix.index)

    def compute_item_similarity(self):
        if self.verbose: print("Computing item similarity...")
        item_matrix = self.user_item_matrix.T
        self.item_similarity_df = pd.DataFrame(cosine_similarity(item_matrix), index=item_matrix.index, columns=item_matrix.index)

    def train_svd(self):
        if self.verbose: print("Training SVD model...")
        reader = Reader(rating_scale=(self.ratings_df['rating'].min(), self.ratings_df['rating'].max()))
        data = Dataset.load_from_df(self.ratings_df[['userId', 'tmdbId', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
        self.svd_model.fit(trainset)

    def run_all(self):
        self.load_data()
        self.build_content_model()
        self.create_user_item_matrix()
        self.compute_user_similarity()
        self.compute_item_similarity()
        self.train_svd()
        if self.verbose: print("Recommender system initialized.")

    def set_user_profile(self, user_id, train_item_ids):
        if self.original_user_item_matrix is None: raise ValueError("Original matrix not found.")
        self.user_item_matrix = self.original_user_item_matrix.copy()
        new_user_row = pd.Series(0, index=self.user_item_matrix.columns, name=user_id)
        valid_movie_ids = [mid for mid in train_item_ids if mid in new_user_row.index]
        new_user_row.loc[valid_movie_ids] = 5.0
        if user_id in self.user_item_matrix.index:
            self.user_item_matrix.loc[user_id] = new_user_row
        else:
            self.user_item_matrix = pd.concat([self.user_item_matrix, new_user_row.to_frame().T])
        self.compute_user_similarity()

    def recommend_content_based(self, movie_title, top_n=10):
        if self.content_matrix is None: raise ValueError("Content model not built.")
        normalized_title = movie_title.lower().strip()
        title_map = self.metadata_df.set_index('tmdbId')['title'].apply(lambda x: str(x).lower().strip())
        movie_id_series = title_map[title_map == normalized_title]
        if movie_id_series.empty:
            close_matches = get_close_matches(normalized_title, title_map.values, n=3)
            raise ValueError(f"Movie '{movie_title}' not found. Closest matches: {close_matches}")
        movie_id = movie_id_series.index[0]
        if movie_id not in self.content_id_map: raise ValueError(f"Movie ID {movie_id} not in content index.")
        movie_idx = self.content_id_map[movie_id]
        cosine_sims = linear_kernel(self.content_matrix[movie_idx], self.content_matrix).flatten()
        sim_scores = pd.Series(cosine_sims, index=self.content_id_map.index).drop(movie_id)
        top_sims = sim_scores.sort_values(ascending=False).head(top_n)
        result_df = self.metadata_df[self.metadata_df['tmdbId'].isin(top_sims.index)].copy()
        result_df = result_df.merge(top_sims.rename('similarity'), left_on='tmdbId', right_index=True)
        return result_df.sort_values(by='similarity', ascending=False)
    
    def recommend(self, user_id=None, movie_title=None, top_k=10, strategy='hybrid', filter_seen=True, **kwargs):
        if strategy == 'content':
            if not movie_title: raise ValueError("A 'movie_title' must be provided for the 'content' strategy.")
            return self.recommend_content_based(movie_title=movie_title, top_n=top_k)
        if not user_id: raise ValueError("A 'user_id' must be provided for other strategies.")
        
        # Dispatch to the correct method based on strategy
        if strategy == 'user':
            return self.recommend_user_based(user_id=user_id, top_n=top_k, filter_seen=filter_seen, **kwargs)
        elif strategy == 'item':
            return self.recommend_item_based(user_id=user_id, top_n=top_k, filter_seen=filter_seen, **kwargs)
        elif strategy == 'svd':
            return self.recommend_svd_based(user_id=user_id, top_n=top_k, filter_seen=filter_seen, **kwargs)
        elif strategy == 'hybrid':
            return self.recommend_hybrid(user_id=user_id, top_n=top_k, filter_seen=filter_seen, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def recommend_user_based(self, user_id, top_n=10, filter_seen=True, **kwargs):
        if self.user_similarity_df is None: raise ValueError("User similarity not computed.")
        if user_id not in self.user_item_matrix.index: return pd.DataFrame()
        similar_users = self.user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)
        if similar_users.empty: return pd.DataFrame()
        sim_matrix = self.user_item_matrix.loc[similar_users.index]
        weighted_scores = np.dot(similar_users.values, sim_matrix.values)
        if similar_users.sum() == 0: return pd.DataFrame()
        norm_scores = weighted_scores / similar_users.sum()
        predicted_ratings = pd.Series(norm_scores, index=self.user_item_matrix.columns)
        if filter_seen:
            already_rated = self.user_item_matrix.loc[user_id]
            predicted_ratings = predicted_ratings[already_rated == 0]
        top_items = predicted_ratings.sort_values(ascending=False).head(top_n).index
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_items)].copy()
        result['predicted_rating'] = result['tmdbId'].map(predicted_ratings)
        return result.sort_values(by='predicted_rating', ascending=False)

    def recommend_item_based(self, user_id, top_n=10, filter_seen=True, **kwargs):
        if self.item_similarity_df is None: raise ValueError("Item similarity not computed.")
        if user_id not in self.user_item_matrix.index: return pd.DataFrame()
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        if rated_items.empty: return pd.DataFrame()
        sim_scores = self.item_similarity_df.loc[:, rated_items]
        weighted_ratings = np.dot(sim_scores.values, user_ratings[rated_items].values)
        sim_sums = sim_scores.sum(axis=1).replace(0, 1e-9)
        predicted_ratings = pd.Series(weighted_ratings / sim_sums, index=sim_scores.index)
        if filter_seen:
            predicted_ratings = predicted_ratings.drop(index=rated_items, errors='ignore')
        top_items = predicted_ratings.sort_values(ascending=False).head(top_n).index
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_items)].copy()
        result['predicted_rating'] = result['tmdbId'].map(predicted_ratings)
        return result.sort_values(by='predicted_rating', ascending=False)

    def recommend_svd_based(self, user_id, top_n=10, filter_seen=True, **kwargs):
        if self.svd_model is None: raise ValueError("SVD model not trained.")
        all_movie_ids = self.metadata_df['tmdbId'].unique()
        if filter_seen:
            rated_movie_ids = self.ratings_df[self.ratings_df['userId'] == user_id]['tmdbId'].unique()
            unseen_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
        else:
            unseen_movie_ids = all_movie_ids
        predictions = [self.svd_model.predict(uid=user_id, iid=movie_id) for movie_id in unseen_movie_ids]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_preds = predictions[:top_n]
        top_movie_ids = [pred.iid for pred in top_preds]
        top_ratings = {pred.iid: pred.est for pred in top_preds}
        result_df = self.metadata_df[self.metadata_df['tmdbId'].isin(top_movie_ids)].copy()
        result_df['predicted_rating'] = result_df['tmdbId'].map(top_ratings)
        return result_df.sort_values('predicted_rating', ascending=False)

    def recommend_hybrid(self, user_id, top_n=10, filter_seen=True, alpha=0.8, beta=0.2, **kwargs):
        user_cf = self.recommend_user_based(user_id=user_id, top_n=len(self.user_item_matrix.columns), filter_seen=filter_seen)
        user_cf_scores = dict(zip(user_cf['tmdbId'], user_cf['predicted_rating']))
        item_cf = self.recommend_item_based(user_id=user_id, top_n=len(self.user_item_matrix.columns), filter_seen=filter_seen)
        item_cf_scores = dict(zip(item_cf['tmdbId'], item_cf['predicted_rating']))
        def normalize(d):
            if not d: return {}
            max_val, min_val = max(d.values()), min(d.values())
            return {k: (v - min_val) / (max_val - min_val) if max_val > min_val else 1.0 for k, v in d.items()}
        user_cf_scores, item_cf_scores = normalize(user_cf_scores), normalize(item_cf_scores)
        all_ids = set(user_cf_scores) | set(item_cf_scores)
        hybrid_scores = {tid: alpha * user_cf_scores.get(tid, 0) + beta * item_cf_scores.get(tid, 0) for tid in all_ids}
        if filter_seen:
            already_rated = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
            final_scores = {k: v for k, v in hybrid_scores.items() if k not in already_rated}
        else:
            final_scores = hybrid_scores
        top_ids_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_ids = [k for k, v in top_ids_scores]
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_ids)].copy()
        result['hybrid_score'] = result['tmdbId'].map(dict(top_ids_scores))
        return result.sort_values(by='hybrid_score', ascending=False)
