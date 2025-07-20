import os
import sys
import itertools
import pandas as pd
# Step 1: Add project root (MoviePulse) to sys.path

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(project_root)
from src.Evaluate.Evaluate_holdout import evaluate_holdout
from utils.logger import log_evaluation_results
from src.recommenders.recommender_system import RecommenderSystem

# =====================================================================================
# CORE EVALUATION LOGIC
# =====================================================================================

def evaluate_all_models(model, ratings_df, strategies=['user', 'item', 'hybrid'], k=10, users_to_evaluate=50):
    """
    Evaluates all specified recommendation strategies.
    """
    results = []
    print("\n--- Running General Model Comparison ---")
    for strategy in strategies:
        print(f"\nEvaluating {strategy.upper()}...")
        try:
            # Use default alpha, beta, gamma for the general hybrid evaluation
            model_params = {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3} if strategy == 'hybrid' else {}

            metrics = evaluate_holdout(
                model=model,
                ratings_df=ratings_df,
                k=k,
                users_to_evaluate=users_to_evaluate,
                strategy=strategy,
                **model_params
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
    print("\n--- General Comparison Complete ---")
    return pd.DataFrame(results)


# =====================================================================================
# NEW: HYBRID MODEL GRID SEARCH FUNCTION
# =====================================================================================

def tune_hybrid_model(model, ratings_df, k=10, users_to_evaluate=50):
    """
    Performs a grid search to find the best alpha, beta, and gamma weights for the hybrid model.
    """
    print("\n--- Starting Hybrid Model Tuning (Grid Search) ---")
    
    # Define the grid of weights. Feel free to adjust these values.
    # Based on prior results, we favor higher alpha (user-based) and lower beta (item-based).
    param_grid = {
        'alpha': [0.6, 0.7, 0.8],
        'beta': [0.1, 0.2],
        'gamma': [0.1, 0.2, 0.3]
    }

    tuning_results = []
    
    # Generate all combinations of the parameters
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        # Create a dictionary for the current combination of parameters
        model_params = dict(zip(keys, v))
        
        # We only test combinations that sum to 1.0 for a weighted average
        if sum(model_params.values()) != 1.0:
            continue

        print(f"\nTuning with Alpha={model_params['alpha']}, Beta={model_params['beta']}, Gamma={model_params['gamma']}")
        
        try:
            metrics = evaluate_holdout(
                model=model,
                ratings_df=ratings_df,
                k=k,
                users_to_evaluate=users_to_evaluate,
                strategy='hybrid',
                **model_params  # Pass the specific weights to the evaluation function
            )

            metrics.update(model_params) # Add the parameters to the results dictionary
            tuning_results.append(metrics)

        except Exception as e:
            print(f"Error during tuning run: {e}")
            continue

    print("\n--- Hybrid Model Tuning Complete ---")
    if not tuning_results:
        print("No valid parameter combinations found that sum to 1.0. Please check your param_grid.")
        return pd.DataFrame()
        
    return pd.DataFrame(tuning_results)


# =====================================================================================
# MAIN EXECUTION BLOCK
# =====================================================================================

if __name__ == "__main__":
    # Step 1: Add project root to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)

    # Step 2: Define file paths
    ratings_path = r"c:\Users\sarva\MoviePulse\data\processed\ratings_cleans.csv"
    metadata_path = r"C:\Users\sarva\MoviePulse\data\processed\movie_feature.csv"
    tfidf_path = r"c:\Users\sarva\MoviePulse\data\Features\tfidf_matrix.npz"
    tfidf_index_path = r"C:\Users\sarva\MoviePulse\data\Features\tfidf_sample.csv"

    # Step 3: Initialize and run the recommender system
    recommender = RecommenderSystem(
        ratings_path=ratings_path,
        metadata_path=metadata_path,
        tfidf_path=tfidf_path,
        tfidf_index_path=tfidf_index_path
    )
    recommender.run_all()
    ratings_df = pd.read_csv(ratings_path)


    # Run the general evaluation of all models
    df_results = evaluate_all_models(recommender, ratings_df, k=10, users_to_evaluate=50)
    
    print("\nFinal Comparison of All Models:")
    # Ensure the dataframe is printed in a consistent order
    cols_to_show = ['Model', 'Precision@K', 'Recall@K', 'NDCG@K', 'Users Evaluated']
    print(df_results[[col for col in cols_to_show if col in df_results.columns]])

    # Run the grid search to tune the hybrid model
    tuning_df = tune_hybrid_model(recommender, ratings_df, k=10, users_to_evaluate=50)
    
    if not tuning_df.empty:
        print("\nHybrid Tuning Results (Best First):")
        # Sort by NDCG as it's a very comprehensive metric
        print(tuning_df.sort_values(by="NDCG@K", ascending=False))