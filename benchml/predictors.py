import numpy as np
import sklearn.linear_model
import sklearn.kernel_ridge
import sklearn.ensemble
import sklearn.svm

class Ridge(sklearn.linear_model.Ridge):
    def covers(meta):
        return meta["task"] == "regress"
    def __init__(self, X, y, whiten=True, **kwargs):
        sklearn.linear_model.Ridge.__init__(self, **kwargs)
        self.whiten = whiten
        self.y_mean = 0.
        self.y_std = 1.
    def fit(self, X, y):
        y_train = y
        if self.whiten:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            y_train = (y-self.y_mean)/self.y_std
        super().fit(X, y_train)
    def predict(self, X):
        return super().predict(X)*self.y_std + self.y_mean

class KernelRidge(sklearn.kernel_ridge.KernelRidge):
    def covers(meta):
        return meta["task"] == "regress"
    def __init__(self, K, y, whiten=True, **kwargs):
        sklearn.kernel_ridge.KernelRidge.__init__(self, **kwargs)
        self.whiten = whiten
        self.y_mean = 0.
        self.y_std = 1.
    def fit(self, K, y):
        y_train = y
        if self.whiten:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            y_train = (y-self.y_mean)/self.y_std
        super().fit(K, y_train)
    def predict(self, K):
        return super().predict(K)*self.y_std + self.y_mean

class RandomForestRegressor(sklearn.ensemble.RandomForestRegressor):
    def covers(meta):
        return meta["task"] == "regress"
    def __init__(self, X, y, **kwargs):
        sklearn.ensemble.RandomForestRegressor.__init__(self, **kwargs)

class SVC(sklearn.svm.SVC):
    def covers(meta):
        return meta["task"] == "classify"
    def __init__(self, K, y, **kwargs):
        sklearn.svm.SVC(self, **kwargs)

