import os
import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# API key from environment or hardcoded
TMDB_API_KEY = os.getenv("TMDB_API_KEY") or "e52ee85b9fa9966a3e9db5aa141ef9cc"
BASE_URL = "https://api.themoviedb.org/3"

# ------------------- API Call with Retry ------------------- #
def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(2)
            else:
                time.sleep(1)
        except:
            time.sleep(1)
    return {}

# ------------------- TMDB Fetch Functions ------------------- #
def get_movie_details(tmdb_id):
    return safe_get(f"{BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US")

def get_credits(tmdb_id):
    return safe_get(f"{BASE_URL}/movie/{tmdb_id}/credits?api_key={TMDB_API_KEY}")

def get_keywords(tmdb_id):
    return safe_get(f"{BASE_URL}/movie/{tmdb_id}/keywords?api_key={TMDB_API_KEY}")

def get_reviews(tmdb_id):
    return safe_get(f"{BASE_URL}/movie/{tmdb_id}/reviews?api_key={TMDB_API_KEY}&language=en-US")

# ------------------- Combine All Metadata ------------------- #
def fetch_tmdb_metadata(tmdb_id):
    try:
        details = get_movie_details(tmdb_id)
        credits = get_credits(tmdb_id)
        keywords = get_keywords(tmdb_id)
        reviews = get_reviews(tmdb_id)

        genres = ", ".join([g['name'] for g in details.get("genres", [])])
        cast = ", ".join([c['name'] for c in credits.get("cast", [])[:5]])
        crew = ", ".join([c['name'] for c in credits.get("crew", []) if c['job'] in ["Director", "Writer"]])
        keyword_tags = ", ".join([k['name'] for k in keywords.get("keywords", [])])
        review_texts = " || ".join([r['content'] for r in reviews.get("results", [])[:3]])

        return {
            "tmdbId": tmdb_id,
            "title": details.get("title"),
            "overview": details.get("overview"),
            "tagline": details.get("tagline"),
            "genres": genres,
            "runtime": details.get("runtime"),
            "release_date": details.get("release_date"),
            "language": details.get("original_language"),
            "vote_average": details.get("vote_average"),
            "vote_count": details.get("vote_count"),
            "popularity": details.get("popularity"),
            "cast": cast,
            "crew": crew,
            "keywords": keyword_tags,
            "reviews": review_texts
        }
    except Exception as e:
        return {"tmdbId": tmdb_id, "error": str(e)}

# ------------------- Batch Fetch with Threading ------------------- #
def fetch_metadata_batch(tmdb_ids, batch_num, save_dir= r"C:\Users\sarva\MoviePulse\data\metadata_batches"):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_tmdb_metadata, tmdb_id): tmdb_id for tmdb_id in tmdb_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_num}"):
            result = future.result()
            results.append(result)

    batch_df = pd.DataFrame(results)
    batch_path = os.path.join(save_dir, f"batch_{batch_num}.csv")
    batch_df.to_csv(batch_path, index=False)
    print(f"Saved batch {batch_num} to {batch_path}")
    return batch_path

# ------------------- Full Process with Chunking ------------------- #
def fetch_all_metadata(tmdb_ids, batch_size=500):
    total_batches = (len(tmdb_ids) + batch_size - 1) // batch_size
    for i in range(total_batches):
        batch_ids = tmdb_ids[i * batch_size: (i + 1) * batch_size]
        batch_file = f"c:/Users/sarva/MoviePulse/data/metadata_batches/batch_{i}.csv"
        if os.path.exists(batch_file):
            print(f"⏭Skipping batch {i} (already exists)")
            continue
        fetch_metadata_batch(batch_ids, i)

# ------------------- Example Run ------------------- #
#if __name__ == "__main__":
    #df = pd.read_csv("data/movies_clean.csv")
    #tmdb_ids = df['tmdbId'].dropna().astype(int).unique().tolist()

   # # Quick test (remove or replace):
    #tmdb_ids = [550, 24428, 27205, 278, 157336] * 2

    #fetch_all_metadata(tmdb_ids)