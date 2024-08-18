import numpy as np
import pandas as pd
from sklearn.utils import resample

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.tree_ = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if num_samples <= 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)
        
        left_indices = X[:, best_split['feature']] <= best_split['value']
        right_indices = X[:, best_split['feature']] > best_split['value']
        
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'feature': best_split['feature'], 
            'value': best_split['value'], 
            'left': left_tree, 
            'right': right_tree
        }
    
    def _find_best_split(self, X, y):
        best_split = None
        best_mse = float('inf')
        num_features = X.shape[1]

        for feature in range(num_features):
            values = np.unique(X[:, feature])
            for value in values:
                left_indices = X[:, feature] <= value
                right_indices = X[:, feature] > value
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                left_y = y[left_indices]
                right_y = y[right_indices]
                
                mse = (np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)) / len(y)
                
                if mse < best_mse:
                    best_split = {'feature': feature, 'value': value}
                    best_mse = mse
        
        return best_split

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict(sample, self.tree_) for sample in X])
    
    def _predict(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        
        if float(sample[tree['feature']]) <= float(tree['value']):
            return self._predict(sample, tree['left'])
        else:
            return self._predict(sample, tree['right'])


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_resampled, y_resampled)
            self.trees.append(tree)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)
