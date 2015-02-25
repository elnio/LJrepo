import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor

__all__ = ["OnlineEnsemble"]


class OnlineEnsemble:
    def __init__(self):
        self._estimators = []

    def first_build(self, n_estimators, x, y):
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(x, y)
        for tree in rf.estimators_:
            self._estimators.append(tree)

    def predict_results(self, x):
        results = []
        for estimator in self._estimators:
            results.append(estimator.predict(x))
        return results

    def predict_weighted_sum(self, x, weights):
        results = self.predict_results(x)
        return np.dot(results, weights)

    def delete(self, idx_list):
        new_estimators = []
        last = 0
        for idx in idx_list:
            for i in range(last, idx):
                new_estimators.append(self._estimators[i])
            last = idx
        self._estimators = new_estimators

    def insert(self, estimators):
        for estimator in estimators:
            self._estimators.append(estimator)

    def insert(self, n_estimators, x, y):
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(x, y)
        for tree in rf.estimators_:
            self._estimators.append(tree)