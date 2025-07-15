import pandas as pd 

def load_movielens(path = 'C:/Users/sarva/MoviePulse/data/raw'):
    links = pd.read_csv(f"{path}/links.csv")
    movies = pd.read_csv(f"{path}/movies.csv")
    ratings = pd.read_csv(f"{path}/ratings.csv")
    tags = pd.read_csv(f"{path}/tags.csv")
    
    merged_data = movies.merge(links , on = 'movieId' , how='left')
    
    return { 'movies' : movies,
            'links' : links,
            'ratings' :ratings,
            'tags' :tags,
            'movies_merge' : merged_data
            }
