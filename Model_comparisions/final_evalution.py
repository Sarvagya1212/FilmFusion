import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# =============================================================================
# 1. SETUP AND IMPORTS
# =============================================================================
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
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

    metrics_to_plot = ['Precision@K', 'Recall@K', 'NDCG@K']
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
    
    if not available_metrics:
        print("No valid metric columns found in the results to plot.")
        return

    df_melted = results_df.melt(id_vars='Model', value_vars=available_metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, edgecolor='black', palette='viridis')
    
    ax.set_title('MoviePulse: Final Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model Strategy', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12, rotation=0)
    ax.legend(title='Metric', fontsize=11)
    ax.set_ylim(0, max(df_melted['Score']) * 1.15)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    final_chart_path = os.path.join(current_dir, save_path)
    plt.savefig(final_chart_path, dpi=300)
    print(f"\n✅ Final chart saved successfully to: {os.path.abspath(final_chart_path)}")


if __name__ == "__main__":
    # --- Configuration ---
    ratings_path = os.path.join(project_root, "data", "processed", "ratings_cleans.csv")
    metadata_path = os.path.join(project_root, "data", "processed", "movies_metadata_enriched.csv")
    
    # Define the content columns to match your final data file 
    content_feature_columns = [
        'overview', 'tagline', 'genres_y', 'cast', 'crew', 'keywords', 'reviews'
    ]

    # 1. Initialize Recommender System
    print("Initializing Recommender System...")
    recommender = RecommenderSystem(
        ratings_path=ratings_path, 
        metadata_path=metadata_path,
        content_cols=content_feature_columns
    )
    recommender.run_all()
    ratings_df = pd.read_csv(ratings_path)
    print("Initialization complete.")

    # =================================================================================
    # 2. TUNE HYBRID MODEL FIRST
    # =================================================================================
    print("\n--- Step 1: Tuning Hybrid Model to Find Best Parameters ---")
    
    param_grid = {
        'alpha': [0.6, 0.7, 0.8], # Weight for User-Based CF
        'beta': [0.1, 0.2],      # Weight for Item-Based CF
        'delta': [0.1, 0.2, 0.3]   # Weight for Sentiment score
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
        
    print(f"\n✅ Best hybrid parameters found: {best_params} (with NDCG: {best_ndcg:.4f})")

    # =================================================================================
    # 3. RUN THE FINAL COMPARISON WITH ALL BEST MODELS
    # =================================================================================
    print("\n--- Step 2: Running Final Comparison with Best Models ---")
    final_results = []
    
    strategies_to_evaluate = ['user', 'item', 'svd']
    
    for strategy in strategies_to_evaluate:
        print(f"Evaluating final {strategy.capitalize()}-Based model...")
        metrics = evaluate_holdout(recommender, ratings_df, strategy=strategy)
        metrics['Model'] = strategy.capitalize().replace('_', '-') + '-Based'
        final_results.append(metrics)

    print("Evaluating final Tuned Hybrid model...")
    tuned_hybrid_metrics = evaluate_holdout(recommender, ratings_df, strategy='hybrid', **best_params)
    tuned_hybrid_metrics['Model'] = 'Tuned Hybrid'
    final_results.append(tuned_hybrid_metrics)

    # =================================================================================
    # 4. DISPLAY AND PLOT THE FINAL RESULTS
    # =================================================================================
    final_df = pd.DataFrame(final_results)
    
    print("\n\n--- 🏆 FINAL MODEL COMPARISON 🏆 ---")
    final_df_sorted = final_df.sort_values(by="NDCG@K", ascending=False)
    print(final_df_sorted[['Model', 'Precision@K', 'Recall@K', 'NDCG@K']])
    
    print("\n--- Step 3: Generating Final Plot ---")
    plot_final_results(final_df_sorted)
