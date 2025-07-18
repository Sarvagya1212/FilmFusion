import pandas as pd
import os
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
import joblib

class TFIDFFeatureEngineer:
    def __init__(self, input_path, save_path_csv, save_path_npz, vectorizer_path=None):
        self.input_path = input_path
        self.save_path_csv = save_path_csv
        self.save_path_npz = save_path_npz
        self.vectorizer_path = vectorizer_path  # optional save for future use
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)

    def load_data(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        self.df = pd.read_csv(self.input_path)

        # ---------- Safe Genre Parsing ----------
        def safe_literal_eval(val):
            try:
                if isinstance(val, str):
                    parsed = literal_eval(val)
                    return parsed if isinstance(parsed, list) else []
                return []
            except Exception:
                return []

        if 'genres' in self.df.columns:
            self.df['genres'] = self.df['genres'].apply(safe_literal_eval)
            self.df['genres_str'] = self.df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        else:
            self.df['genres_str'] = ''

        # ---------- Combine title and genres ----------
        self.df['title'] = self.df['title'].fillna('')
        self.df['text'] = self.df['title'] + ' ' + self.df['genres_str'].fillna('')

    def compute_tfidf(self):
        print("Computing TF-IDF features...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text'])

    def save_features(self):
        print("Saving TF-IDF features...")

        # Create save directory if not exists
        save_dir = os.path.dirname(self.save_path_csv)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save TF-IDF matrix
        scipy.sparse.save_npz(self.save_path_npz, self.tfidf_matrix)

        # Save a sample CSV with tmdbId index (optional but helpful for reference)
        tfidf_df = pd.DataFrame(self.tfidf_matrix.toarray(), index=self.df['tmdbId'])
        tfidf_df.to_csv(self.save_path_csv)

        # Save the vectorizer for reuse
        if self.vectorizer_path:
            joblib.dump(self.vectorizer, self.vectorizer_path)

        print(f"Features saved:\n- Matrix: {self.save_path_npz}\n- Sample CSV: {self.save_path_csv}")
        if self.vectorizer_path:
            print(f"- Vectorizer: {self.vectorizer_path}")

    def run(self):
        self.load_data()
        self.compute_tfidf()
        self.save_features()
