import os
import sys

# Step 1: Add project root (MoviePulse) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)


import pandas as pd
from src.Evaluate.Evaluate_holdout import evaluate_holdout
from utils.logger import log_evaluation_results
from src.recommenders.recommender_system import RecommenderSystem


def evaluate_all_models(model, ratings_df, strategies=['user', 'item', 'hybrid'], k=10, users_to_evaluate=50):
    results = []

    for strategy in strategies:
        print(f"\nEvaluating {strategy.upper()}...\n")
        try:
            metrics = evaluate_holdout(
                model=model,
                ratings_df=ratings_df,
                k=k,
                users_to_evaluate=users_to_evaluate,
                strategy=strategy
            )

            metrics['Model'] = strategy
            results.append(metrics)

            log_evaluation_results(
                metrics=metrics,
                model_name=type(model).__name__,
                top_k=k,
                features_used=["TF-IDF", "Metadata"],
                extra_params={"strategy": strategy, "users": users_to_evaluate}
            )

        except Exception as e:
            print(f"Error in evaluating {strategy}: {e}")
            continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    ratings_path = r"c:\Users\sarva\MoviePulse\data\processed\ratings_cleans.csv"
    metadata_path = r"C:\Users\sarva\MoviePulse\data\processed\movie_feature.csv"
    tfidf_path = r"c:\Users\sarva\MoviePulse\data\Features\tfidf_matrix.npz"
    tfidf_index_path = r"C:\Users\sarva\MoviePulse\data\Features\tfidf_sample.csv"

    recommender = RecommenderSystem(
        ratings_path=ratings_path,
        metadata_path=metadata_path,
        tfidf_path=tfidf_path,
        tfidf_index_path=tfidf_index_path
    )
    recommender.run_all()

    ratings_df = pd.read_csv(ratings_path)
    df_results = evaluate_all_models(recommender, ratings_df, k=10, users_to_evaluate=50)

    print("\nFinal Comparison:")
    print(df_results)
