from collections import defaultdict
from sklearn.metrics import ndcg_score
import numpy as np

def precision_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(ground_truth))
    return hits / k

def recall_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0

def ndcg_at_k(recommended, ground_truth, k):
    relevance = [1 if item in ground_truth else 0 for item in recommended[:k]]
    return ndcg_score([relevance], [sorted(relevance, reverse=True)])

def evaluate_strategy(recommender, test_data_dict, strategy='user', top_k=10):
    precision_list, recall_list, ndcg_list = [], [], []

    for user_id, test_items in test_data_dict.items():
        try:
            recommendations = recommender.recommend(user_id, top_k=top_k, strategy=strategy)
            recommended_ids = recommendations['tmdbId'].tolist()

            precision = precision_at_k(recommended_ids, test_items, top_k)
            recall = recall_at_k(recommended_ids, test_items, top_k)
            ndcg = ndcg_at_k(recommended_ids, test_items, top_k)

            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)

        except Exception as e:
            continue  # skip users with errors or cold-start issues

    return {
        "strategy": strategy,
        "precision@{}".format(top_k): np.mean(precision_list),
        "recall@{}".format(top_k): np.mean(recall_list),
        "ndcg@{}".format(top_k): np.mean(ndcg_list),
        "users_evaluated": len(precision_list)
    }
