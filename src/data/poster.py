import os
import requests
import pandas as pd
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import glob

# ------------------- TMDB Configuration ------------------- #
TMDB_API_KEY = os.getenv("TMDB_API_KEY") or "e52ee85b9fa9966a3e9db5aa141ef9cc"
BASE_URL = "https://api.themoviedb.org/3"

def get_full_poster_url(poster_path, size="w500"):
    return f"https://image.tmdb.org/t/p/{size}{poster_path}" if poster_path else None

# ------------------- Safe API Call with Retry ------------------- #
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

# ------------------- Minimal Metadata Fetch ------------------- #
def fetch_tmdb_poster(tmdb_id):
    try:
        details = safe_get(f"{BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US")
        poster_path = details.get("poster_path")
        poster_url = get_full_poster_url(poster_path)
        return {
            "tmdbId": tmdb_id,
            "poster_path": poster_path,
            "poster_url": poster_url
        }
    except Exception as e:
        return {"tmdbId": tmdb_id, "error": str(e)}

# ------------------- Batch Fetch ------------------- #
def fetch_poster_batch(tmdb_ids, batch_num, save_dir=r"C:\Users\sarva\MoviePulse\data\metadata_batches"):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_tmdb_poster, tmdb_id): tmdb_id for tmdb_id in tmdb_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_num}"):
            result = future.result()
            results.append(result)
            time.sleep(random.uniform(0.1, 0.3))  # Optional delay

    batch_df = pd.DataFrame(results)
    batch_path = os.path.join(save_dir, f"poster_{batch_num}.csv")
    batch_df.to_csv(batch_path, index=False)
    print(f" Saved batch {batch_num} to {batch_path}")
    return batch_path

# ------------------- Full Process ------------------- #
def fetch_all_posters(tmdb_ids, batch_size=500, save_dir=r"C:\Users\sarva\MoviePulse\data\metadata_batches"):
    total_batches = (len(tmdb_ids) + batch_size - 1) // batch_size
    for i in range(total_batches):
        batch_ids = tmdb_ids[i * batch_size: (i + 1) * batch_size]
        batch_path = os.path.join(save_dir, f"poster_{i}.csv")
        if os.path.exists(batch_path):
            print(f"⏭ Skipping batch {i} (already exists)")
            continue
        fetch_poster_batch(batch_ids, i, save_dir)

# ------------------- Combine All Poster Batches ------------------- #
def combine_all_posters(batch_dir=r"C:\Users\sarva\MoviePulse\data\metadata_batches", output_path=r"C:\Users\sarva\MoviePulse\data\processed\posters_minimal.csv"):
    all_files = glob.glob(os.path.join(batch_dir, "poster_*.csv"))
    df_all = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    df_all.drop_duplicates(subset=["tmdbId"], inplace=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, index=False)
    print(f"✅ Combined all posters into {output_path}")

# ------------------- Main ------------------- #
if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\sarva\MoviePulse\data\processed\movies_clean.csv")
    tmdb_ids = df['tmdbId'].dropna().astype(int).unique().tolist()

    print(f"🎬 Total TMDB IDs to fetch posters for: {len(tmdb_ids)}")
    fetch_all_posters(tmdb_ids, batch_size=500)
    combine_all_posters()
