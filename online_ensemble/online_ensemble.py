from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn import svm, grid_search
from sklearn.externals.joblib import Parallel, delayed
import math
import random
import scipy.stats as stats
__all__ = ["OnlineEnsemble"]


def _predict(tree, x):
        return tree.predict(x)


class OnlineEnsemble:
    def __init__(self, n_jobs):
        self._estimators = {}
        self._cnt = 0
        self.n_jobs = n_jobs

    def predict_results(self, x):
        results = {}
        #for idx in self._estimators:
        #    results[idx] = self._estimators[idx].predict(x)
        r = Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(_predict)(self._estimators[idx], x) for idx in self._estimators.keys())
        i = 0
        for idx in self._estimators.keys():
            results[idx] = r[i]
            i += 1
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
            single_result = int(results[idx])
            if not (single_result in vote_dict.keys()):
                vote_dict[single_result] = 0
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
            
    def insert_with_SVM_regressor(self, n_estimators, x, y):
            
        # defining the parameter grids to optimize on.
        # we are performing randomized grid search with statistical distributions for probabilistic searching.
        #param_grid = [ 
        #    {'C': stats.expon(scale=100),'kernel': ['linear'], 'class_weight':['auto', None]}, 
        #    {'C': stats.expon(scale=100), 'gamma': stats.expon(scale=.1), 'kernel': ['rbf'], 'class_weight':['auto', None]} ]
        param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
        # put some grid search for 'poly' kernel and maybe for 'sigmoid'        
        svr = svm.SVR()
        # 3-fold cross validation, 
        optimalSvr = grid_search.GridSearchCV(svr, param_grid, n_jobs=self.n_jobs)
        # this takes a lot of time to compute every time. Need to change that in the future and save the best parameters, but then 
        # there is a risk of the best parameters changing when the distribution changes as well.
        # also if we set n_jobs != 1, we can exploit parallelization 
        bootstrapingSvr = BaggingRegressor(base_estimator = optimalSvr, n_estimators=n_estimators, oob_score = True, n_jobs=self.n_jobs)
        bootstrapingSvr.fit(x, y)
        self.insert_with_estimators(bootstrapingSvr.estimators_)
        
    def insert_with_random_forest_regressor(self, n_estimators, x, y):
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=self.n_jobs)
        rf.fit(x, y)
        self.insert_with_estimators(rf.estimators_)

    def insert_with_random_forest_classifier(self, n_estimators, x, y):
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=self.n_jobs)
        rf.fit(x, y)
        self.insert_with_estimators(rf.estimators_)

    def insert(self, n_estimators, x, y, category):
        if category == 'svm_regressor':
            self.insert_with_SVM_regressor(n_estimators, x, y)
        if category == 'random_forest_regressor':
            self.insert_with_random_forest_regressor(n_estimators, x, y)
        if category == 'random_forest_classifier':
            self.insert_with_random_forest_classifier(n_estimators, x, y)
        if category == 'combination':
            if random.random() < 0.25:
                self.insert_with_SVM_regressor(n_estimators, x, y)
            else:
                self.insert_with_random_forest_regressor(n_estimators, x, y)