import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# =============================================================================
# STEP 1: SETUP AND IMPORTS
# This pathing logic ensures the script can find your 'src' folder.
# =============================================================================
try:
    # Get the directory of the current script ('comparision')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory, which is the project root ('MoviePulse')
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # Add the project root to the Python path to find the 'src' module
    sys.path.append(project_root)

    from src.recommenders.recommender_system import RecommenderSystem
    from src.Evaluate.Evaluate_holdout import evaluate_holdout
except ImportError as e:
    print(f"FATAL ERROR: Could not import project files: {e}")
    print("Please make sure this script is inside a 'comparision' folder, which is in your project's root directory ('MoviePulse').")
    sys.exit(1)


def plot_final_results(results_df, save_path='MoviePulse_Final_Comparison.png'):
    """Generates and saves a professional, grouped bar chart of the final model results."""
    if results_df.empty:
        print("Cannot generate plot, the results DataFrame is empty.")
        return

    # Define the metrics to plot and check which ones are available in the dataframe
    metrics_to_plot = ['Precision@K', 'Recall@K', 'NDCG@K']
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
    
    if not available_metrics:
        print("No valid metric columns found in the results to plot.")
        return

    # Prepare data for Seaborn's grouped bar plot
    df_melted = results_df.melt(id_vars='Model', value_vars=available_metrics, var_name='Metric', value_name='Score')

    # Create the plot
    plt.figure(figsize=(14, 8)) # Made the figure slightly wider for more models
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, edgecolor='black', palette='viridis')
    
    # Customize the plot for a professional look
    ax.set_title('MoviePulse: Final Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model Strategy', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12, rotation=0) # Ensure x-axis labels are horizontal
    ax.legend(title='Metric', fontsize=11)
    ax.set_ylim(0, max(df_melted['Score']) * 1.15)

    # Add score labels on top of each bar for clarity
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
    
    # Save the chart inside the 'comparision' folder
    plt.tight_layout()
    final_chart_path = os.path.join(current_dir, save_path)
    plt.savefig(final_chart_path, dpi=300)
    print(f"\n Final chart saved successfully to: {os.path.abspath(final_chart_path)}")


if __name__ == "__main__":
    # --- Configuration ---
    # Paths are built from the project_root, ensuring they work correctly
    ratings_path = os.path.join(project_root, "data", "processed", "ratings_cleans.csv")
    metadata_path = os.path.join(project_root, "data", "processed", "movie_feature.csv")
    tfidf_path = os.path.join(project_root, "data", "Features", "tfidf_matrix.npz")
    tfidf_index_path = os.path.join(project_root, "data", "Features", "tfidf_sample.csv")

    # 1. Initialize Recommender System (this will now also train SVD)
    print("Initializing Recommender System...")
    recommender = RecommenderSystem(
        ratings_path=ratings_path, metadata_path=metadata_path,
        tfidf_path=tfidf_path, tfidf_index_path=tfidf_index_path
    )
    recommender.run_all()
    ratings_df = pd.read_csv(ratings_path)
    print("Initialization complete.")

    # =================================================================================
    # STEP 2: TUNE HYBRID MODEL FIRST
    # =================================================================================
    print("\n--- Step 1: Tuning Hybrid Model to Find Best Parameters ---")
    param_grid = {
        'alpha': [0.6, 0.7, 0.8], 'beta': [0.1, 0.2], 'gamma': [0.1, 0.2, 0.3]
    }
    best_params = None
    best_ndcg = -1

    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        if abs(sum(params.values()) - 1.0) > 1e-9:
            continue
        
        print(f"Testing Hybrid with: {params}")
        metrics = evaluate_holdout(recommender, ratings_df, strategy='hybrid', **params)
        
        current_ndcg = metrics.get('NDCG@K', -1)
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_params = params

    if not best_params:
        print("\nFATAL: Tuning failed. Exiting.")
        sys.exit(1)
        
    print(f"\n Best hybrid parameters found: {best_params} (with NDCG: {best_ndcg:.4f})")

    # =================================================================================
    # STEP 3: RUN THE FINAL COMPARISON WITH ALL BEST MODELS
    # =================================================================================
    print("\n--- Step 2: Running Final Comparison with Best Models ---")
    final_results = []
    
    # Define all models to be tested in the final comparison
    strategies_to_evaluate = ['user', 'item', 'svd']
    
    for strategy in strategies_to_evaluate:
        print(f"Evaluating final {strategy.capitalize()}-Based model...")
        metrics = evaluate_holdout(recommender, ratings_df, strategy=strategy)
        metrics['Model'] = strategy.capitalize().replace('_', '-') + '-Based'
        final_results.append(metrics)

    # Evaluate the BEST Tuned Hybrid model using the best_params we just found
    print("Evaluating final Tuned Hybrid model...")
    tuned_hybrid_metrics = evaluate_holdout(recommender, ratings_df, strategy='hybrid', **best_params)
    tuned_hybrid_metrics['Model'] = 'Tuned Hybrid'
    final_results.append(tuned_hybrid_metrics)

    # =================================================================================
    # STEP 4: DISPLAY AND PLOT THE FINAL RESULTS
    # =================================================================================
    final_df = pd.DataFrame(final_results)
    
    print("\n\n--- FINAL MODEL COMPARISON  ---")
    # Sort the final results by NDCG for a clear ranking
    final_df_sorted = final_df.sort_values(by="NDCG@K", ascending=False)
    print(final_df_sorted[['Model', 'Precision@K', 'Recall@K', 'NDCG@K']])
    
    print("\n--- Step 3: Generating Final Plot ---")
    # Plot the results, ensuring the models are ordered by performance
    plot_final_results(final_df_sorted)