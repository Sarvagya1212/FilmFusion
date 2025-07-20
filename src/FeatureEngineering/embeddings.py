import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from ast import literal_eval
import sys

class EmbeddingFeatureEngineer:
    def __init__(self, input_path, save_path_npy, save_path_csv=None, model_name='all-MiniLM-L6-v2', text_fields=None):
        """
        Parameters:
            input_path (str): Path to input CSV file
            save_path_npy (str): Path to save the NumPy embedding matrix
            save_path_csv (str): (Optional) Path to save CSV with tmdbId + embeddings
            model_name (str): SentenceTransformer model name
            text_fields (list): Metadata fields to include in text (e.g., ['title', 'genres', 'overview'])
        """
        self.input_path = input_path
        self.save_path_npy = save_path_npy
        self.model_name = model_name
        self.save_path_csv = save_path_csv
        self.text_fields = text_fields or ['title', 'genres', 'overview', 'tagline', 'keywords']
        self.df = None
        self.embedding = None
        self.model = SentenceTransformer(model_name)
        
    def load_data(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f'Input file {self.input_path} does not exist.')

        self.df = pd.read_csv(self.input_path)

        def parse_genres_safe(x):
            if isinstance(x, str):
                try:
                    val = literal_eval(x)
                    if isinstance(val, list):
                        return ' '.join([d['name'] for d in val if isinstance(d, dict) and 'name' in d])
                except Exception:
                    pass
            return ''

        if 'genres' in self.df.columns:
            self.df['genres'] = self.df['genres'].apply(parse_genres_safe)
        else:
            self.df['genres'] = ''

        def combine_text(row):
            return ' '.join(str(row[col]) for col in self.text_fields if col in row and pd.notna(row[col]))

        self.df['text'] = self.df.apply(combine_text, axis=1).str.lower()

        
    def compute_embeddings(self):
        if self.df is None:
            raise ValueError('Data not loaded. Call load_data() first.')
        
        print("--> Encoding embeddings...")
        self.embedding = self.model.encode(self.df['text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
        print("--> Embeddings encoded successfully.")
        
    def save_features(self):
        print("--> Saving embeddings...")
        if self.embedding is not None:
            np.save(self.save_path_npy, self.embedding)
            print(f"Embeddings saved to {self.save_path_npy}")
            
        if self.save_path_csv:
            print('Saving tmdbId and embeddings to CSV...')
            embed_df = pd.DataFrame(self.embedding)
            embed_df.insert(0, 'tmdbId', self.df['tmdbId'])
            embed_df.to_csv(self.save_path_csv, index=False)
            print(f"Embeddings saved to {self.save_path_csv}")
        
    def run(self):
        self.load_data()
        self.compute_embeddings()
        self.save_features()
        print("Embedding feature engineering completed successfully.")
        