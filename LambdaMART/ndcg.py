import numpy as np
class NDCG:
    def __init__(self, queries, rankings):
        self.queries = queries
        self.rankings = rankings

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
        for ind, query_id in enumerate(self.queries.query_ids):
            true_relevances = self.queries.get_query(query_id).relevance_scores
            ranked_relevances = true_relevances[self.rankings[ind]]
            all_ndcgs.append(self.ndcg(ranked_relevances, rank))
        return np.mean(all_ndcgs)
