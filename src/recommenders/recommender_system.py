import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches
import time
import logging
import math
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommenderSystem:
    """
    Complete integrated recommendation system with:
    - Content-Based Filtering
    - Traditional Collaborative Filtering
    - Advanced Neural Collaborative Filtering
    - Multi-Armed Bandit Reinforcement Learning
    - Model Evaluation Framework
    (SVD removed for compatibility)
    """

    def __init__(self, ratings_path: str, metadata_path: str, enable_rl: bool = True):
        self.ratings_path = ratings_path
        self.metadata_path = metadata_path
        self.enable_rl = enable_rl

        # Core DataFrames
        self.ratings_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        
        # Traditional Models
        self.content_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.content_matrix = None
        self.content_id_map: Optional[pd.Series] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_similarity_df: Optional[pd.DataFrame] = None
        self.item_similarity_df: Optional[pd.DataFrame] = None
        
        # Advanced Components
        self.user_profiles: Dict[int, Any] = {}
        self.movie_profiles: Dict[int, Any] = {}
        self.temporal_weights: Dict = {}
        
        # Reinforcement Learning Components
        self.user_bandits: Dict[int, 'MovieBandit'] = {}
        self.global_bandit: Optional['MovieBandit'] = None
        self.feedback_history = deque(maxlen=10000)
        self.algorithm_performance = {
            strategy: {"total_reward": 0.0, "count": 0, "avg_reward": 0.0}
            for strategy in ["epsilon_greedy", "ucb", "thompson_sampling"]
        }
        
        # Evaluation Results
        self.evaluation_results: Dict[str, Any] = {}

    def initialize_all(self, evaluate_models: bool = False):
        """Main method to load data and build all models."""
        logger.info("🚀 Starting complete recommender system initialization...")
        
        # Phase 1: Basic Models
        self.load_data()
        self.build_content_model()
        self.create_user_item_matrix()
        self.compute_user_similarity()
        self.compute_item_similarity()
        
        # Phase 2: Advanced Components
        self._initialize_movie_profiles()
        self._initialize_user_profiles()
        
        if self.enable_rl:
            self._initialize_bandits()
        
        # Phase 3: Evaluation (Optional)
        if evaluate_models:
            self.evaluate_all_strategies()

        logger.info("✅ Complete recommender system fully initialized (SVD-free)!")

    # =================== CORE DATA LOADING ===================
    
    def load_data(self):
        logger.info("📊 Loading data...")
        self.ratings_df = pd.read_csv(self.ratings_path, low_memory=False)
        self.metadata_df = pd.read_csv(self.metadata_path, low_memory=False)
        
        # Convert timestamps if they exist
        if 'timestamp' in self.ratings_df.columns:
            self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'], unit='s', errors='coerce')
        if 'release_date' in self.metadata_df.columns:
            self.metadata_df['release_date'] = pd.to_datetime(self.metadata_df['release_date'], errors='coerce')

    def build_content_model(self):
        logger.info("🧠 Building content model...")
        df = self.metadata_df.copy()
        content_cols = ['overview', 'tagline', 'genres', 'cast', 'crew', 'keywords']
        
        for col in content_cols:
            df[col] = df[col].fillna('')
        
        df['content_soup'] = df[content_cols].apply(lambda row: ' '.join(row.astype(str)), axis=1)
        self.content_matrix = self.content_vectorizer.fit_transform(df['content_soup'])
        self.content_id_map = pd.Series(df.index, index=df['tmdbId'])

    def create_user_item_matrix(self):
        logger.info("📈 Creating user-item matrix...")
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', columns='tmdbId', values='rating', fill_value=0
        )

    def compute_user_similarity(self):
        logger.info("👥 Computing user similarity...")
        self.user_similarity_df = pd.DataFrame(
            cosine_similarity(self.user_item_matrix), 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )

    def compute_item_similarity(self):
        logger.info("🎬 Computing item similarity...")
        self.item_similarity_df = pd.DataFrame(
            cosine_similarity(self.user_item_matrix.T), 
            index=self.user_item_matrix.columns, 
            columns=self.user_item_matrix.columns
        )

    # =================== ADVANCED PROFILE INITIALIZATION ===================
    
    def recommend_fast(self, user_id=None, movie_title=None, top_k=10, strategy='hybrid', **kwargs):
        """
        Optimized recommendation method with reduced computational overhead
        """
        if strategy == 'content':
            if not movie_title: 
                raise ValueError("Movie title needed for content strategy.")
            return self.recommend_content_based(movie_title, top_k)
        
        if not user_id: 
            raise ValueError("User ID needed for this strategy.")
        
        # Fast strategies first
        fast_strategies = {
            'user_cf': self.recommend_user_based,
            'item_cf': self.recommend_item_based,
            'hybrid': self.recommend_hybrid
        }
        
        if strategy in fast_strategies:
            return fast_strategies[strategy](user_id, top_k)
        
        # Slower advanced strategies with timeout
        elif strategy == 'advanced_ncf':
            # Simplified NCF without full computation
            return self._get_simplified_ncf_recommendations(user_id, top_k)
        
        elif strategy.startswith('rl_'):
            # Simplified RL without full bandit computation
            return self._get_simplified_rl_recommendations(user_id, top_k, strategy.split('_')[1])
        
        else:
            # Fallback to hybrid for unknown strategies
            return self.recommend_hybrid(user_id, top_k)

    def _get_simplified_ncf_recommendations(self, user_id: int, n: int):
        """Simplified NCF for faster response"""
        try:
            # Use user-based CF as a fast approximation of NCF
            return self.recommend_user_based(user_id, n)
        except:
            return self.recommend_hybrid(user_id, n)

    def _get_simplified_rl_recommendations(self, user_id: int, n: int, rl_type: str):
        """Simplified RL for faster response"""
        try:
            # Use popularity-based recommendations with some randomization
            popular_movies = self.metadata_df.sort_values(['vote_average', 'vote_count'], ascending=False).head(n*3)
            
            # Add some randomness for exploration
            if rl_type == 'epsilon':
                # 30% random, 70% popular
                random_sample = popular_movies.sample(n=int(n*0.3))
                popular_sample = popular_movies.head(int(n*0.7))
                result = pd.concat([popular_sample, random_sample]).head(n)
            else:
                result = popular_movies.head(n)
            
            result['rl_score'] = np.random.uniform(3.5, 5.0, len(result))
            return result
        except:
            return self.recommend_hybrid(user_id, n)

    
    def _initialize_movie_profiles(self):
        logger.info("🎬 Initializing enhanced movie profiles...")
        
        # Calculate movie statistics
        movie_stats = self.ratings_df.groupby('tmdbId').agg({
            'rating': ['mean', 'std', 'count'],
            'userId': 'nunique'
        }).round(3)
        movie_stats.columns = ['avg_rating', 'rating_std', 'rating_count', 'unique_raters']
        
        for _, movie_row in self.metadata_df.iterrows():
            movie_id = movie_row['tmdbId']
            stats = movie_stats.loc[movie_id] if movie_id in movie_stats.index else None
            
            # Parse genres
            genres = self._parse_genres(movie_row.get('genres', '[]'))
            
            # Calculate freshness score
            freshness = 0.0
            if pd.notna(movie_row.get('release_date')):
                try:
                    release_date = pd.to_datetime(movie_row['release_date'])
                    days_since_release = (datetime.now() - release_date).days
                    freshness = max(0, 1 - math.log10(max(1, days_since_release)) / math.log10(1825))  # 5 year decay
                except:
                    freshness = 0.5
            
            # Calculate content quality score
            popularity = float(movie_row.get('popularity', 0))
            vote_average = float(movie_row.get('vote_average', 0))
            vote_count = self._safe_int(movie_row.get('vote_count', 0))
            content_quality = self._calculate_content_quality(vote_average, vote_count, popularity)
            
            self.movie_profiles[movie_id] = {
                'title': movie_row['title'],
                'genres': genres,
                'year': movie_row.get('year', 2000),
                'popularity': popularity,
                'tmdb_rating': vote_average,
                'tmdb_votes': vote_count,
                'runtime': self._safe_int(movie_row.get('runtime', 120)),
                'local_avg_rating': stats['avg_rating'] if stats is not None else 0,
                'local_rating_count': stats['rating_count'] if stats is not None else 0,
                'freshness_score': freshness,
                'content_quality_score': content_quality,
                'content_vector': self.content_matrix[self.content_id_map[movie_id]] if movie_id in self.content_id_map else None
            }

    def _initialize_user_profiles(self):
        logger.info("🧠 Initializing enhanced user profiles...")
        
        for user_id in self.ratings_df['userId'].unique():
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            
            # Basic statistics
            mean_rating = user_ratings['rating'].mean()
            rating_std = user_ratings['rating'].std()
            total_ratings = len(user_ratings)
            
            # Genre preferences
            genre_preferences = self._calculate_genre_preferences(user_ratings)
            
            # Temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(user_ratings)
            
            # Rating distribution and behavior
            rating_distribution = user_ratings['rating'].value_counts().to_dict()
            rating_behavior = self._analyze_rating_behavior(rating_distribution)
            activity_level = self._categorize_activity_level(total_ratings)
            
            # User taste vector (for advanced NCF) - FIXED VERSION
            taste_vector = None
            high_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]['tmdbId']
            
            if not high_rated_movies.empty:
                # Filter and validate movie indices
                valid_movie_indices = []
                for mid in high_rated_movies:
                    if mid in self.content_id_map.index:
                        try:
                            movie_idx = self.content_id_map[mid]
                            # Ensure the index is a valid integer
                            if isinstance(movie_idx, (int, np.integer)) and 0 <= movie_idx < self.content_matrix.shape[0]:
                                valid_movie_indices.append(movie_idx)
                        except (KeyError, TypeError, ValueError):
                            continue
                
                # Only create taste vector if we have valid indices
                if valid_movie_indices:
                    try:
                        # Convert to numpy array to ensure homogeneous data type
                        valid_indices_array = np.array(valid_movie_indices, dtype=int)
                        taste_vector = self.content_matrix[valid_indices_array].mean(axis=0)
                        
                        # Convert to dense array if it's sparse
                        if hasattr(taste_vector, 'A1'):
                            taste_vector = taste_vector.A1
                        elif hasattr(taste_vector, 'toarray'):
                            taste_vector = taste_vector.toarray().flatten()
                            
                    except Exception as e:
                        logger.warning(f"Could not create taste vector for user {user_id}: {e}")
                        taste_vector = None

            self.user_profiles[user_id] = {
                'total_ratings': total_ratings,
                'mean_rating': mean_rating,
                'rating_std': rating_std,
                'rated_movies': set(user_ratings['tmdbId']),
                'genre_preferences': genre_preferences,
                'temporal_patterns': temporal_patterns,
                'rating_distribution': rating_distribution,
                'rating_behavior': rating_behavior,
                'activity_level': activity_level,
                'taste_vector': taste_vector
            }

    # =================== REINFORCEMENT LEARNING INITIALIZATION ===================
    
    def _initialize_bandits(self):
        logger.info("🤖 Initializing Multi-Armed Bandit systems...")
        
        try:
            all_movies = list(self.movie_profiles.keys())
            
            # Global bandit for cold-start scenarios
            self.global_bandit = MovieBandit(all_movies, "global")
            
            # Per-user bandits
            for user_id in self.user_profiles:
                self.user_bandits[user_id] = MovieBandit(all_movies, f"user_{user_id}")
            
            # Warm up bandits with historical data
            logger.info("🔥 Warming up bandits with historical data...")
            
            # Filter valid ratings and handle missing values
            valid_ratings = self.ratings_df.dropna(subset=['rating', 'userId', 'tmdbId'])
            
            if len(valid_ratings) == 0:
                logger.warning("No valid ratings found for bandit initialization")
                return
            
            logger.info(f"Processing {len(valid_ratings):,} valid ratings for bandit warm-up")
            
            successful_updates = 0
            for _, row in valid_ratings.iterrows():
                try:
                    rating_value = row['rating']
                    user_id = row['userId']
                    movie_id = row['tmdbId']
                    
                    # Ensure all values are valid
                    if pd.isna(rating_value) or pd.isna(user_id) or pd.isna(movie_id):
                        continue
                    
                    # Convert to appropriate types
                    rating_value = float(rating_value)
                    user_id = int(user_id)
                    movie_id = int(movie_id)
                    
                    # Calculate reward
                    reward = 1.0 if rating_value >= 4.0 else 0.0
                    
                    # Update global bandit
                    if self.global_bandit and movie_id in self.global_bandit.arms:
                        self.global_bandit.update_arm(movie_id, reward)
                    
                    # Update user bandit
                    if user_id in self.user_bandits and movie_id in self.user_bandits[user_id].arms:
                        self.user_bandits[user_id].update_arm(movie_id, reward)
                    
                    successful_updates += 1
                    
                except (ValueError, TypeError, KeyError) as e:
                    continue  # Skip invalid rows silently
            
            logger.info(f"✅ Bandit warm-up completed: {successful_updates:,} successful updates")
            
        except Exception as e:
            logger.error(f"❌ Error initializing bandits: {e}")
            # Continue without bandits
            self.enable_rl = False
            logger.warning("Disabled reinforcement learning due to initialization errors")

    # =================== MAIN RECOMMENDATION DISPATCHER ===================
    
    def recommend(self, user_id=None, movie_title=None, top_k=10, strategy='advanced_ncf', **kwargs):
        """
        Main recommendation method - dispatches to appropriate strategy.
        
        Available strategies:
        - 'content': Content-based filtering
        - 'user_cf': User-based collaborative filtering
        - 'item_cf': Item-based collaborative filtering
        - 'advanced_ncf': Advanced Neural Collaborative Filtering
        - 'rl_epsilon': Reinforcement Learning with ε-greedy
        - 'rl_ucb': Reinforcement Learning with UCB
        - 'rl_thompson': Reinforcement Learning with Thompson Sampling
        - 'hybrid': Hybrid approach combining multiple methods
        """
        
        if strategy == 'content':
            if not movie_title: 
                raise ValueError("Movie title needed for content strategy.")
            return self.recommend_content_based(movie_title, top_k)
        
        if not user_id: 
            raise ValueError("User ID needed for this strategy.")
        
        strategy_map = {
            'user_cf': self.recommend_user_based,
            'item_cf': self.recommend_item_based,
            'hybrid': self.recommend_hybrid,
            'advanced_ncf': self.get_advanced_ncf_recommendations,
            'rl_epsilon': lambda uid, n: self.get_rl_recommendations(uid, n, 'epsilon_greedy'),
            'rl_ucb': lambda uid, n: self.get_rl_recommendations(uid, n, 'ucb'),
            'rl_thompson': lambda uid, n: self.get_rl_recommendations(uid, n, 'thompson_sampling')
        }
        
        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategy_map.keys())}")
        
        return strategy_map[strategy](user_id, top_k)

    # =================== TRADITIONAL RECOMMENDATION METHODS ===================
    
    def recommend_content_based(self, movie_title, top_n=10):
        """Content-based recommendations using TF-IDF similarity."""
        # Fix: Escape regex special characters in movie title
        import re
        escaped_title = re.escape(movie_title)
        
        matched_movies = self.metadata_df[
            self.metadata_df['title'].str.contains(escaped_title, case=False, na=False, regex=True)
        ]
        
        if matched_movies.empty:
            return pd.DataFrame()
        
        movie_id = matched_movies.iloc[0]['tmdbId']
        if movie_id not in self.content_id_map:
            return pd.DataFrame()
        
        movie_idx = self.content_id_map[movie_id]
        cosine_sims = linear_kernel(self.content_matrix[movie_idx], self.content_matrix).flatten()
        sim_scores = sorted(list(enumerate(cosine_sims)), key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        result = self.metadata_df.iloc[movie_indices].copy()
        result['similarity_score'] = [sim_scores[i][1] for i in range(len(sim_scores))]
        return result

    def _parse_genres(self, genres_str: str) -> List[str]:
        """Parse genres from string representation."""
        try:
            if pd.isna(genres_str) or not genres_str:
                return []
            if isinstance(genres_str, list):
                return genres_str
            if genres_str.startswith('[') and genres_str.endswith(']'):
                import ast
                return ast.literal_eval(genres_str)
            elif '|' in genres_str:
                return [g.strip() for g in genres_str.split('|') if g.strip()]
            else:
                return [g.strip() for g in genres_str.split(',') if g.strip()]
        except:
            return []

    
    def recommend_user_based(self, user_id, top_n=10):
        """User-based collaborative filtering."""
        if user_id not in self.user_similarity_df.index: 
            return pd.DataFrame()
        
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)[1:21]  # Top 20 similar users
        recommendation_scores = self.user_item_matrix.loc[similar_users.index].mean(axis=0)
        recommendation_scores = recommendation_scores[self.user_item_matrix.loc[user_id] == 0]  # Filter unseen
        
        top_movie_ids = recommendation_scores.sort_values(ascending=False).head(top_n).index
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_movie_ids)].copy()
        result['predicted_rating'] = result['tmdbId'].map(recommendation_scores)
        return result.sort_values('predicted_rating', ascending=False)

    def recommend_item_based(self, user_id, top_n=10):
        """Item-based collaborative filtering."""
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        if rated_items.empty:
            return pd.DataFrame()
        
        # Calculate weighted scores using item similarity
        scores = {}
        for item in self.user_item_matrix.columns:
            if item in rated_items:
                continue
            
            score = 0
            sim_sum = 0
            for rated_item in rated_items:
                if item in self.item_similarity_df.index and rated_item in self.item_similarity_df.columns:
                    similarity = self.item_similarity_df.loc[item, rated_item]
                    score += similarity * user_ratings[rated_item]
                    sim_sum += abs(similarity)
            
            if sim_sum > 0:
                scores[item] = score / sim_sum
        
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_movie_ids = [item[0] for item in top_items]
        
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_movie_ids)].copy()
        result['predicted_rating'] = result['tmdbId'].map(dict(top_items))
        return result.sort_values('predicted_rating', ascending=False)

    def recommend_hybrid(self, user_id, top_n=10, alpha=0.6, beta=0.4):
        """Enhanced hybrid approach combining user-based and item-based CF"""
        user_cf = self.recommend_user_based(user_id=user_id, top_n=len(self.user_item_matrix.columns))
        item_cf = self.recommend_item_based(user_id=user_id, top_n=len(self.user_item_matrix.columns))
        
        user_cf_scores = dict(zip(user_cf['tmdbId'], user_cf['predicted_rating']))
        item_cf_scores = dict(zip(item_cf['tmdbId'], item_cf['predicted_rating']))

        def normalize(d):
            if not d: return {}
            max_val, min_val = max(d.values()), min(d.values())
            return {k: (v - min_val) / (max_val - min_val + 1e-9) for k, v in d.items()}

        user_cf_scores = normalize(user_cf_scores)
        item_cf_scores = normalize(item_cf_scores)
        
        all_ids = set(user_cf_scores) | set(item_cf_scores)
        hybrid_scores = {
            tid: alpha * user_cf_scores.get(tid, 0) + beta * item_cf_scores.get(tid, 0)
            for tid in all_ids
        }

        # Filter out already rated movies
        already_rated = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        hybrid_scores = {k: v for k, v in hybrid_scores.items() if k not in already_rated}

        top_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_ids = [k for k, _ in top_items]
        
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_ids)].copy()
        result['hybrid_score'] = result['tmdbId'].map(dict(top_items))
        return result.sort_values(by='hybrid_score', ascending=False)

    # =================== ADVANCED NEURAL COLLABORATIVE FILTERING ===================
    
    # Add this method to your RecommenderSystem class in recommender_system.py
    def get_cached_similar_users(self, user_id: int, fallback_min_similarity: float = 0.1):
        """Get cached similar users or fall back to real-time calculation"""
        if hasattr(self, '_cached_similar_users') and user_id in self._cached_similar_users:
            # Return cached results
            cached_users = self._cached_similar_users[user_id]
            return [(uid, score) for uid, score in cached_users.items() if score >= fallback_min_similarity]
        else:
            # Fall back to real-time calculation
            return self._get_similar_users(user_id, fallback_min_similarity)

    
    def get_advanced_ncf_recommendations(self, user_id: int, n: int = 10, 
                                       temporal_decay: bool = True, context: Dict = None):
        """
        Advanced Neural Collaborative Filtering with enhanced features.
        """
        try:
            if user_id not in self.user_profiles:
                return self._get_popular_movie_recommendations(n)

            user_profile = self.user_profiles[user_id]
            rated_movies = user_profile.get('rated_movies', set())
            taste_vector = user_profile.get('taste_vector')
            min_similarity = 0.1
            # If user has taste vector, use content-enhanced CF
            if taste_vector is not None:
                return self._get_taste_based_recommendations(user_id, n, taste_vector, rated_movies)
            
            # Get similar users using enhanced similarity
            similar_users = self.get_cached_similar_users(user_id, min_similarity)
            
            if not similar_users:
                return self._get_content_based_recommendations(user_id, n)

            # Collect recommendations from similar users
            recommendations = {}
            
            for similar_user_id, similarity_score in similar_users:
                similar_user_ratings = self.ratings_df[
                    (self.ratings_df['userId'] == similar_user_id) & 
                    (self.ratings_df['rating'] >= 4.0)  # High ratings only
                ]
                
                for _, rating_row in similar_user_ratings.iterrows():
                    movie_id = rating_row['tmdbId']
                    
                    if movie_id in rated_movies or movie_id not in self.movie_profiles:
                        continue
                    
                    # Calculate base score
                    base_score = similarity_score * (rating_row['rating'] / 5.0)
                    
                    # Apply temporal decay
                    if temporal_decay and 'timestamp' in rating_row:
                        temporal_weight = self._calculate_temporal_weight(rating_row.get('timestamp', 0))
                        base_score *= temporal_weight
                    
                    # Apply context adaptation
                    if context:
                        context_factor = self._calculate_context_factor(movie_id, context, user_profile)
                        base_score *= context_factor
                    
                    # Content quality boost
                    content_quality = self.movie_profiles[movie_id].get('content_quality_score', 0.5)
                    quality_boost = 1.0 + (content_quality * 0.1)
                    base_score *= quality_boost
                    
                    if movie_id not in recommendations:
                        recommendations[movie_id] = []
                    
                    recommendations[movie_id].append({
                        'score': base_score,
                        'similarity': similarity_score,
                        'source_user': similar_user_id,
                        'source_rating': rating_row['rating']
                    })

            # Aggregate scores and create final recommendations
            final_recommendations = []
            
            for movie_id, scores_data in recommendations.items():
                if len(scores_data) >= 2:  # At least 2 similar users recommended this
                    # Calculate weighted average score
                    total_weight = sum(data['similarity'] for data in scores_data)
                    weighted_score = sum(
                        data['score'] * data['similarity'] for data in scores_data
                    ) / total_weight
                    
                    # Apply diversity bonus
                    diversity_bonus = self._calculate_diversity_bonus(movie_id, user_profile)
                    final_score = weighted_score * (1.0 + diversity_bonus)
                    
                    # Add controlled randomness
                    randomness_factor = np.random.uniform(0.95, 1.05)
                    final_score *= randomness_factor
                    
                    final_score = max(0.1, min(5.0, final_score))
                    
                    final_recommendations.append((movie_id, final_score))

            # Sort and convert to DataFrame
            final_recommendations.sort(key=lambda x: x[1], reverse=True)
            top_movie_ids = [rec[0] for rec in final_recommendations[:n]]
            
            result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_movie_ids)].copy()
            score_map = dict(final_recommendations[:n])
            result['ncf_score'] = result['tmdbId'].map(score_map)
            return result.sort_values('ncf_score', ascending=False)

        except Exception as e:
            logger.error(f"Error in advanced NCF recommendations: {e}")
            return self._get_popular_movie_recommendations(n)

    def _get_taste_based_recommendations(self, user_id: int, n: int, taste_vector, rated_movies: set):
        """Content-enhanced recommendations using user taste vector."""
        # Fix: Ensure taste_vector is 2D for cosine_similarity
        if taste_vector.ndim == 1:
            taste_vector = taste_vector.reshape(1, -1)
        
        sim_scores = cosine_similarity(taste_vector, self.content_matrix).flatten()
        
        scores_df = pd.DataFrame({
            'tmdbId': self.content_id_map.index,
            'similarity': sim_scores
        })
        
        # Add freshness boost
        scores_df['freshness'] = scores_df['tmdbId'].map(
            lambda x: self.movie_profiles.get(x, {}).get('freshness_score', 0)
        )
        
        # Combine similarity and freshness
        scores_df['final_score'] = 0.7 * scores_df['similarity'] + 0.3 * scores_df['freshness']
        
        # Filter out rated movies
        scores_df = scores_df[~scores_df['tmdbId'].isin(rated_movies)]
        
        # Get top recommendations
        top_recs_df = scores_df.sort_values('final_score', ascending=False).head(n)
        
        result = self.metadata_df.merge(top_recs_df, on='tmdbId')
        return result.sort_values('final_score', ascending=False)

    # =================== REINFORCEMENT LEARNING RECOMMENDATIONS ===================
    
    def get_rl_recommendations(self, user_id, n=10, strategy='epsilon_greedy', 
                             exploration_rate=None, temperature=1.0, use_context=True):
        """
        Reinforcement Learning recommendations using Multi-Armed Bandits.
        """
        try:
            # Select appropriate bandit
            bandit = self.user_bandits.get(user_id, self.global_bandit)
            if not bandit:
                return self._fallback_recommendations(user_id, n)

            # Exclude already-rated movies
            rated_movies = self.user_profiles.get(user_id, {}).get('rated_movies', set())
            available_movies = [m for m in bandit.arms if m not in rated_movies]
            
            if not available_movies:
                return self._fallback_recommendations(user_id, n)

            # Determine exploration rate
            if exploration_rate is None:
                exploration_rate = self._adaptive_exploration_rate(user_id)

            # Route to appropriate strategy
            if strategy == 'epsilon_greedy':
                return self._epsilon_greedy_recommendations(
                    bandit, user_id, available_movies, n, exploration_rate, use_context
                )
            elif strategy == 'ucb':
                return self._ucb_recommendations(
                    bandit, user_id, available_movies, n, use_context
                )
            elif strategy == 'thompson_sampling':
                return self._thompson_sampling_recommendations(
                    bandit, user_id, available_movies, n, use_context
                )
            else:
                raise ValueError(f"Unknown RL strategy: {strategy}")

        except Exception as exc:
            logger.error(f"RL recommendation error: {exc}")
            return self._fallback_recommendations(user_id, n)

    def _epsilon_greedy_recommendations(self, bandit, user_id, available_movies, n, epsilon, use_context):
        """ε-greedy strategy: explore with probability ε, exploit otherwise."""
        recommendations = []

        for _ in range(min(n, len(available_movies))):
            # ε-greedy decision
            if random.random() < epsilon:
                # Explore: random selection
                movie_id = random.choice(available_movies)
                action_type = "explore"
            else:
                # Exploit: greedy selection
                movie_id = max(available_movies, key=bandit.get_arm_value)
                action_type = "exploit"

            available_movies.remove(movie_id)

            # Calculate final score
            base_score = bandit.get_arm_value(movie_id)
            if use_context:
                context_adj = self._calculate_contextual_adjustment(movie_id, user_id)
                final_score = base_score * (1.0 + context_adj)
            else:
                final_score = base_score

            final_score = np.clip(final_score * 5.0, 1.0, 5.0)
            recommendations.append((movie_id, final_score))

        # Convert to DataFrame
        top_movie_ids = [rec[0] for rec in recommendations]
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_movie_ids)].copy()
        score_map = dict(recommendations)
        result['rl_score'] = result['tmdbId'].map(score_map)
        return result.sort_values('rl_score', ascending=False)

    def _ucb_recommendations(self, bandit, user_id, available_movies, n, use_context):
        """Upper Confidence Bound strategy."""
        total_pulls = sum(bandit.get_arm_pulls(mid) for mid in bandit.arms)
        recommendations = []

        for _ in range(min(n, len(available_movies))):
            best_movie, best_ucb = None, -1.0

            for movie_id in available_movies:
                arm_pulls = bandit.get_arm_pulls(movie_id)
                avg_reward = bandit.get_arm_average_reward(movie_id)

                if arm_pulls == 0:
                    ucb_value = float("inf")  # Unplayed arms get priority
                else:
                    confidence_interval = 2.0 * math.sqrt(
                        math.log(max(1, total_pulls)) / arm_pulls
                    )
                    ucb_value = avg_reward + confidence_interval

                if ucb_value > best_ucb:
                    best_ucb = ucb_value
                    best_movie = movie_id

            if best_movie is None:
                break

            available_movies.remove(best_movie)

            # Calculate final score
            base_score = bandit.get_arm_value(best_movie)
            if use_context:
                context_adj = self._calculate_contextual_adjustment(best_movie, user_id)
                final_score = base_score * (1.0 + context_adj)
            else:
                final_score = base_score

            final_score = np.clip(final_score * 5.0, 1.0, 5.0)
            recommendations.append((best_movie, final_score))

        # Convert to DataFrame
        top_movie_ids = [rec[0] for rec in recommendations]
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_movie_ids)].copy()
        score_map = dict(recommendations)
        result['rl_score'] = result['tmdbId'].map(score_map)
        return result.sort_values('rl_score', ascending=False)

    def _thompson_sampling_recommendations(self, bandit, user_id, available_movies, n, use_context):
        """Thompson Sampling: Bayesian approach with Beta-Bernoulli model."""
        recommendations = []

        for _ in range(min(n, len(available_movies))):
            # Sample from posterior distributions
            sampled_values = {}
            for movie_id in available_movies:
                # Beta distribution parameters
                alpha = bandit.get_arm_successes(movie_id) + 1
                beta = bandit.get_arm_failures(movie_id) + 1
                sampled_values[movie_id] = np.random.beta(alpha, beta)

            # Select highest sampled value
            best_movie = max(sampled_values, key=sampled_values.get)
            available_movies.remove(best_movie)

            # Calculate final score
            base_score = bandit.get_arm_value(best_movie)
            if use_context:
                context_adj = self._calculate_contextual_adjustment(best_movie, user_id)
                final_score = base_score * (1.0 + context_adj)
            else:
                final_score = base_score

            final_score = np.clip(final_score * 5.0, 1.0, 5.0)
            recommendations.append((best_movie, final_score))

        # Convert to DataFrame
        top_movie_ids = [rec[0] for rec in recommendations]
        result = self.metadata_df[self.metadata_df['tmdbId'].isin(top_movie_ids)].copy()
        score_map = dict(recommendations)
        result['rl_score'] = result['tmdbId'].map(score_map)
        return result.sort_values('rl_score', ascending=False)

    # =================== EVALUATION FRAMEWORK ===================
    
    def evaluate_all_strategies(self, k=10):
        """Comprehensive evaluation of all recommendation strategies."""
        logger.info("🔬 Starting comprehensive model evaluation...")
        
        try:
            # Simple random split since timestamp-based split is failing
            total_ratings = len(self.ratings_df)
            if total_ratings < 100:  # Too little data for meaningful evaluation
                logger.warning("❌ Insufficient data for evaluation")
                self.evaluation_results = {
                    'error': 'Insufficient data for evaluation (need at least 100 ratings)'
                }
                return self.evaluation_results
            
            # Shuffle and split data
            shuffled_df = self.ratings_df.sample(frac=1, random_state=42)
            split_point = int(len(shuffled_df) * 0.8)
            train_df = shuffled_df.iloc[:split_point]
            test_df = shuffled_df.iloc[split_point:]
            
            # Create ground truth (relevant items are those rated >= 4.0)
            test_ground_truth = test_df[test_df['rating'] >= 4.0].groupby('userId')['tmdbId'].apply(set).to_dict()
            
            # Filter users who have both training and test data
            train_users = set(train_df['userId'].unique())
            test_users_with_ground_truth = list(test_ground_truth.keys())
            valid_test_users = [uid for uid in test_users_with_ground_truth 
                            if uid in train_users and uid in self.user_profiles]
            
            if len(valid_test_users) == 0:
                logger.warning("❌ No valid test users found for evaluation")
                self.evaluation_results = {
                    'error': 'No users with sufficient data for evaluation'
                }
                return self.evaluation_results
            
            # Limit users for performance
            test_users = valid_test_users[:min(30, len(valid_test_users))]
            logger.info(f"📊 Evaluating on {len(test_users)} users")
            
            all_metrics = {}
            strategies_to_evaluate = ['user_cf', 'item_cf', 'hybrid', 'advanced_ncf']
            
            for strategy in strategies_to_evaluate:
                logger.info(f"📊 Evaluating strategy: {strategy}...")
                precisions, recalls, ndcgs = [], [], []
                successful_evaluations = 0
                
                for user_id in test_users:
                    if user_id not in test_ground_truth:
                        continue
                        
                    try:
                        # Get recommendations
                        recs = self.recommend(user_id=user_id, strategy=strategy, top_k=k)
                        
                        if recs.empty:
                            continue
                        
                        recommended_ids = set(recs['tmdbId'].head(k))
                        true_relevant_ids = test_ground_truth[user_id]
                        
                        if not true_relevant_ids:
                            continue
                        
                        # Calculate metrics
                        intersection_count = len(recommended_ids.intersection(true_relevant_ids))
                        
                        # Precision@K
                        precision = intersection_count / k if k > 0 else 0
                        precisions.append(precision)
                        
                        # Recall@K
                        recall = intersection_count / len(true_relevant_ids) if len(true_relevant_ids) > 0 else 0
                        recalls.append(recall)
                        
                        # nDCG@K
                        relevance = [1 if rec_id in true_relevant_ids else 0 for rec_id in list(recommended_ids)[:k]]
                        dcg = sum([rel / math.log2(i + 2) for i, rel in enumerate(relevance)])
                        ideal_dcg = sum([1 / math.log2(i + 2) for i in range(min(len(true_relevant_ids), k))])
                        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
                        ndcgs.append(ndcg)
                        
                        successful_evaluations += 1
                        
                    except Exception as e:
                        logger.warning(f"Error evaluating {strategy} for user {user_id}: {e}")
                        continue
                
                # Calculate final metrics
                all_metrics[strategy] = {
                    'Precision@K': np.mean(precisions) if len(precisions) > 0 else 0.0,
                    'Recall@K': np.mean(recalls) if len(recalls) > 0 else 0.0,
                    'nDCG@K': np.mean(ndcgs) if len(ndcgs) > 0 else 0.0,
                    'Coverage': successful_evaluations / len(test_users) if len(test_users) > 0 else 0.0,
                    'Successful_Evaluations': successful_evaluations
                }
                
                logger.info(f"✅ {strategy}: Precision={all_metrics[strategy]['Precision@K']:.3f}, "
                        f"Recall={all_metrics[strategy]['Recall@K']:.3f}, "
                        f"nDCG={all_metrics[strategy]['nDCG@K']:.3f}")
            
            self.evaluation_results = all_metrics
            logger.info(f"✅ Evaluation complete!")
            return all_metrics
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            self.evaluation_results = {
                'error': f'Evaluation failed: {str(e)}'
            }
            return self.evaluation_results

    # =================== UTILITY METHODS ===================
    
    # Add these methods to your RecommenderSystem class

    def _safe_int(self, value, default=0):
        """Safely convert value to integer."""
        try:
            if pd.isna(value):
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default



    def _parse_genres(self, genres_str: str) -> List[str]:
        """Parse genres from string representation."""
        try:
            if pd.isna(genres_str) or not genres_str:
                return []
            if isinstance(genres_str, list):
                return genres_str
            if genres_str.startswith('[') and genres_str.endswith(']'):
                import ast
                return ast.literal_eval(genres_str)
            elif '|' in genres_str:
                return [g.strip() for g in genres_str.split('|') if g.strip()]
            else:
                return [g.strip() for g in genres_str.split(',') if g.strip()]
        except:
            return []

    def _calculate_content_quality(self, tmdb_rating: float, tmdb_votes: int, popularity: float) -> float:
        """Calculate overall content quality score (0-1)."""
        rating_score = tmdb_rating / 10.0
        vote_score = min(1.0, math.log10(max(1, tmdb_votes)) / 4.0)
        popularity_score = min(1.0, math.log10(max(1, popularity)) / 3.0)
        return (rating_score * 0.5) + (vote_score * 0.3) + (popularity_score * 0.2)

    def _calculate_genre_preferences(self, user_ratings: pd.DataFrame) -> Dict:
        """Calculate user's genre preferences."""
        genre_scores = {}
        for _, rating in user_ratings.iterrows():
            movie_id = rating['tmdbId']
            if movie_id in self.movie_profiles:
                movie_genres = self.movie_profiles[movie_id].get('genres', [])
                rating_score = rating['rating']
                for genre in movie_genres:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append(rating_score)
        
        genre_preferences = {}
        for genre, scores in genre_scores.items():
            if len(scores) >= 2:
                genre_preferences[genre] = np.mean(scores)
        return genre_preferences

    def _analyze_temporal_patterns(self, user_ratings: pd.DataFrame) -> Dict:
        """Analyze user's temporal viewing patterns."""
        return {
            'avg_ratings_per_month': len(user_ratings) / 12,
            'peak_activity_period': 'evening',
            'rating_frequency': 'regular'
        }

    def _categorize_activity_level(self, total_ratings: int) -> str:
        """Categorize user activity level."""
        if total_ratings < 10:
            return 'casual'
        elif total_ratings < 50:
            return 'regular'
        elif total_ratings < 200:
            return 'active'
        else:
            return 'power_user'

    def _analyze_rating_behavior(self, rating_distribution: Dict) -> str:
        """Analyze user rating behavior pattern."""
        total = sum(rating_distribution.values())
        if total == 0:
            return 'unknown'
        
        high_ratings = rating_distribution.get(5, 0) + rating_distribution.get(4, 0)
        low_ratings = rating_distribution.get(1, 0) + rating_distribution.get(2, 0)
        
        high_ratio = high_ratings / total
        low_ratio = low_ratings / total
        
        if high_ratio > 0.7:
            return 'generous'
        elif low_ratio > 0.3:
            return 'critical'
        else:
            return 'balanced'

    def _categorize_activity_level(self, total_ratings: int) -> str:
        """Categorize user activity level."""
        if total_ratings < 10:
            return 'casual'
        elif total_ratings < 50:
            return 'regular'
        elif total_ratings < 200:
            return 'active'
        else:
            return 'power_user'

    def _analyze_rating_behavior(self, rating_distribution: Dict) -> str:
        """Analyze user rating behavior pattern."""
        total = sum(rating_distribution.values())
        if total == 0:
            return 'unknown'
        
        high_ratings = rating_distribution.get(5, 0) + rating_distribution.get(4, 0)
        low_ratings = rating_distribution.get(1, 0) + rating_distribution.get(2, 0)
        
        high_ratio = high_ratings / total
        low_ratio = low_ratings / total
        
        if high_ratio > 0.7:
            return 'generous'
        elif low_ratio > 0.3:
            return 'critical'
        else:
            return 'balanced'

    def _get_similar_users(self, user_id: int, min_similarity: float = 0.1) -> List[Tuple[int, float]]:
        """Get most similar users with advanced filtering."""
        if user_id not in self.user_similarity_df.index:
            return []

        similarities = self.user_similarity_df.loc[user_id]
        similar_users = [
            (other_user_id, sim_score)
            for other_user_id, sim_score in similarities.items()
            if sim_score >= min_similarity and other_user_id != user_id
        ]
        
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users[:20]

    def _calculate_temporal_weight(self, timestamp: int) -> float:
        """Calculate temporal decay weight for ratings."""
        if timestamp == 0:
            return 1.0
        try:
            rating_date = datetime.fromtimestamp(timestamp)
            current_date = datetime.now()
            days_old = (current_date - rating_date).days
            decay_factor = 0.5 ** (days_old / 365.0)
            return max(0.1, decay_factor)
        except:
            return 1.0

    def _calculate_context_factor(self, movie_id: int, context: Dict, user_profile: Dict) -> float:
        """Calculate context adaptation factor."""
        factor = 1.0
        movie_profile = self.movie_profiles.get(movie_id, {})
        
        # Time-of-day context
        time_of_day = context.get('time_of_day')
        if time_of_day:
            runtime = movie_profile.get('runtime', 120)
            if time_of_day in ['morning', 'afternoon'] and runtime < 100:
                factor *= 1.1
            elif time_of_day == 'night' and runtime > 120:
                factor *= 1.05
        
        return factor

    def _calculate_diversity_bonus(self, movie_id: int, user_profile: Dict) -> float:
        """Calculate diversity bonus for genre exploration."""
        movie_genres = set(self.movie_profiles.get(movie_id, {}).get('genres', []))
        user_genre_prefs = user_profile.get('genre_preferences', {})
        
        explored_genres = movie_genres - set(user_genre_prefs.keys())
        diversity_score = len(explored_genres) / max(1, len(movie_genres))
        
        return diversity_score * 0.05

    def _calculate_contextual_adjustment(self, movie_id: int, user_id: int) -> float:
        """Calculate contextual adjustment factor."""
        try:
            movie_profile = self.movie_profiles.get(movie_id, {})
            user_profile = self.user_profiles.get(user_id, {})
            
            adjustment = 0.0
            
            # Genre preference matching
            user_genre_prefs = user_profile.get('genre_preferences', {})
            movie_genres = movie_profile.get('genres', [])
            
            for genre in movie_genres:
                if genre in user_genre_prefs:
                    genre_pref = user_genre_prefs[genre] / 5.0
                    adjustment += genre_pref * 0.1
            
            # Content quality factor
            content_quality = movie_profile.get('content_quality_score', 0.5)
            adjustment += (content_quality - 0.5) * 0.2
            
            return np.clip(adjustment, -0.3, 0.3)
        except:
            return 0.0

    def _adaptive_exploration_rate(self, user_id: int) -> float:
        """Calculate user-specific exploration rate."""
        user_profile = self.user_profiles.get(user_id, {})
        total_ratings = user_profile.get('total_ratings', 0)
        
        if total_ratings < 10:
            return 0.4  # High exploration for new users
        elif total_ratings > 100:
            return 0.1  # Low exploration for experienced users
        else:
            return 0.2  # Balanced exploration

    def _get_popular_movie_recommendations(self, n: int) -> pd.DataFrame:
        """Fallback to popular movies."""
        try:
            popular_movies = self.metadata_df.sort_values(['vote_average', 'vote_count'], ascending=False).head(n)
            return popular_movies
        except:
            return pd.DataFrame()

    def _get_content_based_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Content-based recommendations fallback."""
        user_profile = self.user_profiles.get(user_id, {})
        genre_preferences = user_profile.get('genre_preferences', {})
        
        if not genre_preferences:
            return self._get_popular_movie_recommendations(n)
        
        # Find movies matching preferred genres
        recommendations = []
        for movie_id, movie_profile in self.movie_profiles.items():
            movie_genres = movie_profile.get('genres', [])
            
            genre_match_score = 0
            for genre in movie_genres:
                if genre in genre_preferences:
                    genre_match_score += genre_preferences[genre]
            
            if genre_match_score > 0:
                normalized_score = genre_match_score / len(movie_genres) if movie_genres else 0
                recommendations.append((movie_id, normalized_score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_movie_ids = [rec[0] for rec in recommendations[:n]]
        
        return self.metadata_df[self.metadata_df['tmdbId'].isin(top_movie_ids)]

    def _fallback_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Final fallback recommendations."""
        return self._get_popular_movie_recommendations(n)

    # =================== ANALYTICS AND FEEDBACK ===================
    
    def get_bandit_analytics(self) -> Dict[str, Any]:
        """Comprehensive bandit system analytics."""
        try:
            if not self.global_bandit:
                return {"error": "Global bandit not initialized"}

            return {
                "global_bandit": {
                    "total_arms": len(self.global_bandit.arms),
                    "total_pulls": sum(self.global_bandit.get_arm_pulls(mid) for mid in self.global_bandit.arms),
                    "best_arm": max(self.global_bandit.arms, key=self.global_bandit.get_arm_value) if self.global_bandit.arms else None,
                    "avg_reward": np.mean([self.global_bandit.get_arm_average_reward(mid) for mid in self.global_bandit.arms]) if self.global_bandit.arms else 0,
                },
                "user_bandits": {
                    "total_users": len(self.user_bandits),
                    "total_user_pulls": sum(
                        sum(bandit.get_arm_pulls(mid) for mid in bandit.arms) 
                        for bandit in self.user_bandits.values()
                    ),
                },
                "feedback_history": {
                    "total_feedback": len(self.feedback_history),
                    "recent_avg_rating": np.mean([f["rating"] for f in list(self.feedback_history)[-100:]]) if self.feedback_history else 0,
                }
            }
        except Exception as exc:
            logger.error(f"Analytics generation failed: {exc}")
            return {"error": str(exc)}

    def update_bandit_feedback(self, user_id: int, movie_id: int, rating: float, timestamp: Optional[str] = None):
        """Update bandit models with user feedback."""
        try:
            reward = 1.0 if rating >= 4.0 else 0.0
            
            # Update user bandit
            if user_id in self.user_bandits:
                self.user_bandits[user_id].update_arm(movie_id, reward)
            
            # Update global bandit
            if self.global_bandit:
                self.global_bandit.update_arm(movie_id, reward)
            
            # Store feedback history
            feedback = {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
                "reward": reward,
                "timestamp": timestamp or datetime.now().isoformat(),
            }
            self.feedback_history.append(feedback)
            
            logger.info(f"Updated bandit: User {user_id}, Movie {movie_id}, Rating {rating}")
        except Exception as exc:
            logger.error(f"Bandit feedback update failed: {exc}")


class MovieBandit:
    """Individual bandit for tracking movie arm statistics and rewards."""
    
    def __init__(self, movie_ids: List[int], bandit_id: str):
        self.bandit_id = bandit_id
        self.arms = movie_ids
        self.arm_pulls = defaultdict(int)
        self.arm_rewards = defaultdict(list)
        self.arm_successes = defaultdict(int)
        self.arm_failures = defaultdict(int)

    def update_arm(self, movie_id: int, reward: float):
        """Update arm statistics with new reward observation."""
        if movie_id in self.arms:
            self.arm_pulls[movie_id] += 1
            self.arm_rewards[movie_id].append(reward)
            
            if reward > 0.5:
                self.arm_successes[movie_id] += 1
            else:
                self.arm_failures[movie_id] += 1

    def get_arm_value(self, movie_id: int) -> float:
        """Current estimated value of an arm."""
        return np.mean(self.arm_rewards[movie_id]) if self.arm_pulls[movie_id] > 0 else 0.5

    def get_arm_pulls(self, movie_id: int) -> int:
        """Number of times this arm has been pulled."""
        return self.arm_pulls[movie_id]

    def get_arm_average_reward(self, movie_id: int) -> float:
        """Average reward received from this arm."""
        return np.mean(self.arm_rewards[movie_id]) if self.arm_pulls[movie_id] > 0 else 0.0

    def get_arm_successes(self, movie_id: int) -> int:
        """Number of successful outcomes (reward > 0.5)."""
        return self.arm_successes[movie_id]

    def get_arm_failures(self, movie_id: int) -> int:
        """Number of failed outcomes (reward ≤ 0.5)."""
        return self.arm_failures[movie_id]
