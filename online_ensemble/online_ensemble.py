from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.forest import RandomForestClassifier
import math

__all__ = ["OnlineEnsemble"]


class OnlineEnsemble:
    def __init__(self):
        self._estimators = {}
        self._cnt = 0

    def predict_results(self, x):
        results = {}
        for idx in self._estimators:
            results[idx] = self._estimators[idx].predict(x)
        return results

    def predict_weighted_sum(self, x, weights):
        results = self.predict_results(x)
        sum_ = 0.0
        for idx in results.keys():
            if not (idx in self._estimators):
                raise ValueError('estimator with idx {} does not exist!')
            sum_ += weights[idx] * results[idx]
        return float(sum_)

    def predict_weighted_classification_result(self, x, weights):
        sum_ = self.predict_weighted_sum(x, weights)
        if math.tanh(sum_) > 0:
            return 1
        else:
            return -1

    def predict_weighted_vote(self, x, weights):
        results = self.predict_results(x)
        vote_dict = {}
        max_vote = 0
        result = 0
        for idx in results.keys():
            if not (idx in self._estimators):
                raise ValueError('estimator with idx {} does not exist!')
            single_result = results[idx]
            vote_dict[single_result] += weights[idx]
            if vote_dict[single_result] > max_vote:
                max_vote = vote_dict[single_result]
                result = single_result
        return result

    def get_idx_list(self):
        return self._estimators.keys()

    def delete(self, idx_list):
        for idx in idx_list:
            del self._estimators[idx]

    def insert_with_estimators(self, estimators):
        for estimator in estimators:
            self._estimators[self._cnt] = estimator
            self._cnt += 1

    def insert_with_random_forest_regressor(self, n_estimators, x, y):
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(x, y)
        self.insert_with_estimators(rf.estimators_)

    def insert_with_random_forest_classifier(self, n_estimators, x, y):
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit(x, y)
        self.insert_with_estimators(rf.estimators_)

    def insert(self, n_estimators, x, y, category):
        if category == 'random_forest_regressor':
            self.insert_with_random_forest_regressor(n_estimators, x, y)
        if category == 'random_forest_classifier':
            self.insert_with_random_forest_classifier(n_estimators, x, y)