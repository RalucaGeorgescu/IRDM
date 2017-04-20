import numpy as np

class MAP:
    def __init__(self, queries, predictions):
        self.queries = queries
        self.predictions = predictions

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
        predictions = self.predictions['relevance']
        
        grouped_data = self.queries.groupby('qid')
        for _, group in grouped_data:
            true_relevances = group['relevance']
            
            pred_scores = predictions.iloc[group.index.values]
            sorted_pred = pred_scores.sort(ascending=False, inplace=False)
            indices = sorted_pred.index.values - max(sorted_pred.index.values)
            
            ranked_relevances = true_relevances.iloc[indices]
            binary_relevances = (ranked_relevances > 0).astype(int)
            all_precisions.append(self.average_precision(binary_relevances))
        return np.mean(all_precisions)
