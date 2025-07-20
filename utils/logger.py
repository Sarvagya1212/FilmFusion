import os
from datetime import datetime

def log_evaluation_results(metrics, 
                           model_name=None, 
                           top_k=None, 
                           features_used=None,
                           extra_params=None, 
                           log_dir="logs", 
                           filename="evaluation_log.txt"):
    """
    Appends evaluation metrics and model details to a log file with a timestamp.

    Args:
        metrics (dict): Dictionary containing evaluation metrics like Precision@K, Recall@K, NDCG@K, etc.
        model_name (str): Name or type of the model used.
        top_k (int): Value of K used in top-K recommendation.
        features_used (str or list): Features used for recommendation (e.g., "TF-IDF + Metadata").
        extra_params (dict): Any additional configuration (e.g., test_ratio, similarity metric, etc.)
        log_dir (str): Folder to save the log file.
        filename (str): Name of the log file.
    """

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"\n----- Evaluation @ {timestamp} -----\n"

    if model_name:
        log_entry += f"Model: {model_name}\n"
    if top_k:
        log_entry += f"Top@K: {top_k}\n"
    if features_used:
        if isinstance(features_used, list):
            features_used = ", ".join(features_used)
        log_entry += f"Features Used: {features_used}\n"
    if extra_params:
        for key, val in extra_params.items():
            log_entry += f"{key}: {val}\n"

    for key, value in metrics.items():
        log_entry += f"{key}: {value:.4f}\n" if isinstance(value, float) else f"{key}: {value}\n"

    log_entry += "-------------------------------------\n"

    with open(log_path, "a") as f:
        f.write(log_entry)

    print(f"Evaluation results logged to: {log_path}")
    
    
'''if __name__ == "__main__":


    log_evaluation_results(
        metrics=metrics,
        model_name=type(recommender).__name__,
        top_k=10,
        features_used=["TF-IDF", "Overview", "Tagline", "Genres", "Cast"],
        extra_params={"test_ratio": 0.2, "similarity": "cosine"}
    )'''

