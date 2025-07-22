import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# =============================================================================
# 1. SETUP: LOAD THE RECOMMENDER SYSTEM
# =============================================================================

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)
    from src.recommenders.recommender_system import RecommenderSystem
except ImportError as e:
    print(f"FATAL ERROR: Could not import project files: {e}")
    sys.exit(1)

print("Initializing Recommender System...")

ratings_path = os.path.join(project_root, "data", "processed", "ratings_cleans.csv")
metadata_path = os.path.join(project_root, "data", "processed", "movies_metadata_full.csv")
content_cols = ['overview', 'tagline', 'genres', 'cast', 'crew', 'keywords', 'reviews']

recommender = RecommenderSystem(
    ratings_path=ratings_path,
    metadata_path=metadata_path,
    content_cols=content_cols,
    verbose=False  # Disable verbose logging for cleaner API
)
recommender.run_all()

print("\n✅ Recommender System is ready and loaded.")

# =============================================================================
# 2. DEFINE DATA MODELS (PYDANTIC)
# =============================================================================

class Movie(BaseModel):
    tmdbId: int
    title: str
    predicted_score: float
    genres: Optional[str] = None
    poster_url: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Movie]

class SimilarMovie(BaseModel):
    tmdbId: int
    title: str
    similarity_score: float
    genres: Optional[str] = None
    poster_url: Optional[str] = None

class SimilarMovieResponse(BaseModel):
    source_movie_id: int
    similar_movies: List[SimilarMovie]

# =============================================================================
# 3. CREATE THE FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="MoviePulse API",
    description="API for the MoviePulse Recommendation & Insights Platform.",
    version="1.0.0"
)

# =============================================================================
# 4. DEFINE API ENDPOINTS
# =============================================================================

@app.get("/")
def read_root():
    return {"message": "Welcome to the MoviePulse API! The recommender is online."}

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int):
    """Fetches top-10 movie recommendations for a given user ID using hybrid strategy."""
    print(f"Received request for user_id: {user_id}")
    try:
        recs_df = recommender.recommend(user_id=user_id, strategy='hybrid', top_k=10)

        if recs_df.empty:
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found or no recommendations available.")

        movie_list = []
        for _, row in recs_df.iterrows():
            poster_path = row.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if pd.notna(poster_path) else None

            movie_list.append(Movie(
                tmdbId=row['tmdbId'],
                title=row['title'],
                predicted_score=row['hybrid_score'],
                genres=row.get('genres'),
                poster_url=poster_url
            ))

        return RecommendationResponse(user_id=user_id, recommendations=movie_list)

    except Exception as e:
        print(f"An unexpected error occurred for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/similar_movies/{movie_id}", response_model=SimilarMovieResponse)
def get_similar_movies(movie_id: int):
    """Fetches top-10 movies similar to a given movie ID using the content-based model."""
    print(f"Received request for similar movies to movie_id: {movie_id}")
    try:
        source_movie = recommender.metadata_df[recommender.metadata_df['tmdbId'] == movie_id]
        if source_movie.empty:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found in the dataset.")

        source_title = source_movie.iloc[0]['title']
        similar_movies_df = recommender.recommend_content_based(movie_title=source_title, top_n=10)

        if similar_movies_df.empty:
            raise HTTPException(status_code=404, detail=f"No similar movies found for movie ID {movie_id}.")

        movie_list = []
        for _, row in similar_movies_df.iterrows():
            poster_path = row.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if pd.notna(poster_path) else None

            movie_list.append(SimilarMovie(
                tmdbId=row['tmdbId'],
                title=row['title'],
                similarity_score=row['similarity'],
                genres=row.get('genres'),
                poster_url=poster_url
            ))

        return SimilarMovieResponse(source_movie_id=movie_id, similar_movies=movie_list)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"An unexpected error occurred for movie {movie_id}: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
