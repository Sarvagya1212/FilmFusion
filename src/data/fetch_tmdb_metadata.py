import requests
import pandas as pd
from time import sleep

TMDB_API_KEY = "e52ee85b9fa9966a3e9db5aa141ef9cc"

def get_movie_details(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits,reviews"}
    headers = {"Accept": "application/json"}


    response = requests.get(url, params=params , headers=headers)
    
    if response.status_code != 200:
        return None

    data = response.json()
    return {
        "tmdb_id": tmdb_id,
        "title": data.get("title"),
        "overview": data.get("overview"),
        "genres": [g["name"] for g in data.get("genres", [])],
        "release_date": data.get("release_date"),
        "rating": data.get("vote_average"),
        "poster_url": f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}",
        "cast": [c["name"] for c in data.get("credits", {}).get("cast", [])[:5]],
        "director": next((c["name"] for c in data.get("credits", {}).get("crew", []) if c["job"] == "Director"), None),
        "top_reviews": [r["content"] for r in data.get("reviews", {}).get("results", [])[:3]]
    }
