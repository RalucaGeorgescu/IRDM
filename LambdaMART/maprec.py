import numpy as np
class MAP:
    def __init__(self, queries, rankings):
        self.queries = queries
        self.rankings = rankings

    def precision_at_r(self, rel_scores, rank):
        rel_scores = np.asarray(rel_scores)[:rank]
        return np.mean(rel_scores)

    def average_precision(self, binary_relevances):
        num_relevant = np.sum(binary_relevances)
        if num_relevant == 0:
            denom = 0
        else:
            denom = 1. / num_relevant
        avg_precision =  sum([(sum(binary_relevances[:ind+1]) / (ind + 1.)) * denom for ind, val in enumerate(binary_relevances) if val])
        return avg_precision

    def mean_average_precision(self):
        all_precisions = []
        for ind, query_id in enumerate(self.queries.query_ids):
            true_relevances = self.queries.get_query(query_id).relevance_scores
            ranked_relevances = true_relevances[self.rankings[ind]]
            binary_relevances = (ranked_relevances > 0).astype(int)
            all_precisions.append(self.average_precision(binary_relevances))
        return np.mean(all_precisions)
