import numpy as np
import logging

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    if k == 0: return 0.0
    return len([r for r in recommended_k if r in relevant_set]) / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    if not relevant_set: return 0.0
    return len([r for r in recommended_k if r in relevant_set]) / len(relevant_set)

# NEW: Function to calculate NDCG@K
def ndcg_at_k(recommended, relevant_ratings, k):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG)@K.
    Args:
        recommended (list): List of recommended item IDs.
        relevant_ratings (dict): A dictionary of item IDs to their relevance scores (actual ratings).
        k (int): The number of top recommendations to consider.
    Returns:
        float: The NDCG score.
    """
    recommended_k = recommended[:k]
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0
    for i, item_id in enumerate(recommended_k):
        # The relevance is the actual rating, or 0 if the user didn't rate it.
        relevance = relevant_ratings.get(item_id, 0)
        dcg += relevance / np.log2(i + 2)  # log base 2 of rank+1 (i starts at 0, so i+2)
        
    # Calculate IDCG (Ideal Discounted Cumulative Gain)
    # Sort the user's true ratings to get the ideal ranking
    ideal_ratings = sorted(relevant_ratings.values(), reverse=True)
    ideal_ratings_k = ideal_ratings[:k]
    
    idcg = 0
    for i, relevance in enumerate(ideal_ratings_k):
        idcg += relevance / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_holdout(model, ratings_df, k=10, users_to_evaluate=100, strategy='user', rating_threshold=4.0, **model_params):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    user_ids = model.user_item_matrix.index.astype(int).tolist()
    np.random.seed(42)
    num_users_to_sample = min(users_to_evaluate, len(user_ids))
    if num_users_to_sample == 0:
        logger.warning("No users available to evaluate.")
        return {"Precision@K": 0, "Recall@K": 0, "NDCG@K": 0, "Users Evaluated": 0}

    sampled_users = np.random.choice(user_ids, size=num_users_to_sample, replace=False)
    
    precision_list, recall_list, ndcg_list = [], [], []

    for user_id in sampled_users:
        try:
            user_ratings = model.user_item_matrix.loc[user_id]
            
            # UPDATE: Get both the list of relevant items and a dictionary with their ratings
            relevant_items_series = user_ratings[user_ratings >= rating_threshold]
            relevant_items = relevant_items_series.index.tolist()
            relevant_ratings_dict = relevant_items_series.to_dict()

            if not relevant_items:
                continue

            recommendations = model.recommend(
                user_id=user_id, top_k=k, strategy=strategy, filter_seen=False, **model_params
            )

            if 'tmdbId' not in recommendations.columns:
                continue
            
            recommended_items = recommendations['tmdbId'].astype(int).tolist()
            if not recommended_items:
                continue

            # Calculate all three metrics
            precision = precision_at_k(recommended_items, relevant_items, k)
            recall = recall_at_k(recommended_items, relevant_items, k)
            ndcg = ndcg_at_k(recommended_items, relevant_ratings_dict, k) # NEW

            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg) #  NEW

        except Exception as e:
            logger.error(f"Error evaluating user {user_id}: {e}", exc_info=True)
            continue

    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0 # NEW

    return {
        "Precision@K": round(avg_precision, 4),
        "Recall@K": round(avg_recall, 4),
        "NDCG@K": round(avg_ndcg, 4), # NEW
        "Users Evaluated": len(precision_list)
    }