import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


# Define the Perceptron class (unchanged)
class Perceptron:
    def __init__(self, alpha=0.01, lambda_=0.01, epochs=1000, mode='classification'):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epochs = epochs
        self.W = None
        self.losses = []
        self.mode = mode.lower()
        if self.mode not in ['classification', 'regression']:
            raise ValueError("Mode must be 'classification' or 'regression'")

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if self.mode == 'classification':
            assert np.all(np.abs(y) == 1), "Labels must be ±1 for classification"
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.losses = []
        for epoch in range(self.epochs):
            shuffled_idx = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[shuffled_idx], y[shuffled_idx]
            for Xi, yi in zip(X_shuffled, y_shuffled):
                score = np.dot(self.W, Xi)
                if self.mode == 'classification':
                    pred = 1 if score >= 0 else -1
                    if yi * pred < 0:
                        self.W = self.W * (1 - self.alpha * self.lambda_) + self.alpha * yi * Xi
                    else:
                        self.W *= (1 - self.alpha * self.lambda_)
                else:
                    error = yi - score
                    self.W = self.W * (1 - self.alpha * self.lambda_) + self.alpha * error * Xi
            scores = np.dot(X, self.W)
            if self.mode == 'classification':
                losses = np.maximum(0, -y * scores)
            else:
                losses = (y - scores) ** 2
            self.losses.append(np.mean(losses))
        return self

    def predict(self, X):
        X = np.array(X)
        scores = np.dot(X, self.W)
        if self.mode == 'classification':
            return np.where(scores >= 0, 1, -1)
        else:
            return scores


# Example 1: Classification (synthetic data)
# Generate synthetic classification data
X_cls, y_cls = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2)
y_cls = np.where(y_cls == 0, -1, 1)  # Convert labels to ±1
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2)

# Train and evaluate classifier
perceptron_cls = Perceptron(mode='classification', alpha=0.1, epochs=100)
perceptron_cls.fit(X_train_cls, y_train_cls)
predictions_cls = perceptron_cls.predict(X_test_cls)
print("Classification Predictions:", predictions_cls)

# Example 2: Regression (synthetic data)
# Generate synthetic regression data
X_reg, y_reg = make_regression(n_samples=100, n_features=2, noise=0.1)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2)

# Train and evaluate regressor
perceptron_reg = Perceptron(mode='regression', alpha=0.01, lambda_=0.001, epochs=200)
perceptron_reg.fit(X_train_reg, y_train_reg)
predictions_reg = perceptron_reg.predict(X_test_reg)
print("Regression Predictions:", predictions_reg)
