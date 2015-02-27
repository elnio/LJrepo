import numpy as np
from data_reader import DataReader
from sklearn.ensemble.forest import RandomForestRegressor

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
        return sum_

    def get_idx_list(self):
        return self._estimators.keys()

    def delete(self, idx_list):
        for idx in idx_list:
            del self._estimators[idx]

    def insert(self, estimators):
        for estimator in estimators:
            self._estimators[self._cnt] = estimator
            self._cnt += 1

    def insert_with_rf(self, n_estimators, x, y):
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(x, y)
        self.insert(rf.estimators_)