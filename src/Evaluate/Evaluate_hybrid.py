import numpy as np
import logging

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len([r for r in recommended_k if r in relevant_set]) / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len([r for r in recommended_k if r in relevant_set]) / len(relevant_set) if relevant_set else 0

def evaluate_model(model, k=10, sample_size=100, rating_threshold=4.0, alpha=0.4, beta=0.3, gamma=0.3):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    user_ids = model.user_item_matrix.index.tolist()
    np.random.seed(42)
    sampled_users = np.random.choice(user_ids, size=min(sample_size, len(user_ids)), replace=False)

    precision_list, recall_list = [], []

    for user_id in sampled_users:
        try:
            user_ratings = model.user_item_matrix.loc[user_id]
            relevant_items = user_ratings[user_ratings >= rating_threshold].index.tolist()

            if not relevant_items:
                logger.debug(f"Skipping user {user_id}: no relevant items.")
                continue

            recommendations = model.recommend_hybrid(user_id, top_n=k, alpha=alpha, beta=beta, gamma=gamma)

            if 'tmdbId' in recommendations.columns:
                recommended_items = recommendations['tmdbId'].tolist()
            else:
                logger.warning(f"User {user_id}: No 'tmdbId' column in recommendations.")
                continue

            if not recommended_items:
                logger.debug(f"Skipping user {user_id}: no recommendations.")
                continue

            precision = precision_at_k(recommended_items, relevant_items, k)
            recall = recall_at_k(recommended_items, relevant_items, k)

            precision_list.append(precision)
            recall_list.append(recall)

        except Exception as e:
            logger.warning(f"Error evaluating user {user_id}: {str(e)}")
            continue

    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0

    return {
        "Precision@K": round(avg_precision, 4),
        "Recall@K": round(avg_recall, 4),
        "Users Evaluated": len(precision_list),
        "Alpha (User-CF)": alpha,
        "Beta (Item-CF)": beta,
        "Gamma (Content)": gamma
    }
