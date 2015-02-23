import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
__all__ = ["OnlineRF"]


class OnlineRF(RandomForestRegressor):
    def __init__(self,
                 n_estimators=10,
                 n_jobs=1):
        super(OnlineRF, self).__init__(
            n_estimators=n_estimators,
            n_jobs=n_jobs)
        self.chunk_size = None
        self.last_X_chunk = None
        self.last_y_chunk = None
        self.last_idx = 0

    def fit(self, X, y):
        super(OnlineRF, self).fit(X, y)
        self.chunk_size = X.shape[0]
        self.last_X_chunk = X
        self.last_y_chunk = y
        self.last_idx = 0
        print "fit done."

    def predict(self, X):
        if X.ndim > 1:
            raise ValueError('Please predict only one data point')
        result = np.zeros(self.n_estimators)
        for i, estimator in enumerate(self.estimators_):
            result[i] = estimator.predict(X)
        return result

    def replace_tree(self, flag, cnt):
        print "replace " + str(cnt) + " trees"
        new_forest = RandomForestRegressor(n_estimators=cnt)
        new_forest.fit(self.last_X_chunk, self.last_y_chunk)
        k = 0;
        for i in range(self.n_estimators):
            if flag[i] == 1:
                self.estimators_[i] = new_forest.estimators_[k]
                k += 1

    def update_chunk(self, X, y):
        if X.ndim > 1:
            raise ValueError('Please insert only one latest data point')
        self.last_X_chunk[self.last_idx] = X
        self.last_y_chunk[self.last_idx] = y
        self.last_idx = (self.last_idx + 1) % self.chunk_size