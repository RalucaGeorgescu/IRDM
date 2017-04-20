import numpy as np
class NDCG:
    def __init__(self, queries, predictions):
        self.queries = queries
        self.predictions = predictions

    def dcg(self, rel_scores, rank=10):
        rel_scores = np.asarray(rel_scores)[:rank]
        num_scores = len(rel_scores)
        if num_scores == 0:
            return 0
        gains = 2**rel_scores -1
        discounts = np.log2(np.arange(num_scores)+2)
        return np.sum(gains/discounts)

    def ndcg(self, rank_rel_scores, rank=10):
        dcg_score = self.dcg(rank_rel_scores, rank)
        opt_dcg_score = self.dcg(sorted(rank_rel_scores, reverse=True), rank)
        if opt_dcg_score == 0:
            return 0
        return dcg_score / opt_dcg_score

    def mean_ndcg(self, rank=10):
        all_ndcgs = []
        predictions = self.predictions['relevance']
        
        grouped_data = self.queries.groupby('qid')
        for _, group in grouped_data:
            true_relevances = group['relevance']
            
            pred_scores = predictions.iloc[group.index.values]
            sorted_pred = pred_scores.sort(ascending=False, inplace=False)
            indices = sorted_pred.index.values - max(sorted_pred.index.values)
            
            ranked_relevances = true_relevances.iloc[indices]
            all_ndcgs.append(self.ndcg(ranked_relevances, rank))
        return np.mean(all_ndcgs)

