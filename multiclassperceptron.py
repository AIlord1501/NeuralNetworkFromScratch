import numpy as np
from scipy.linalg import null_space


class multiclassperceptron:
    def __init__(self,num_classes, alpha= 1.0, epochs = 1000):
        self.num_classes = num_classes
        self.alpha = alpha
        self.epochs = epochs
        self.w = None

    def fit(self,X,y):
        n_samples , d = X.shape
        self.w = np.zeros((self.num_classes))

        for _ in range(self.epochs):
            for i in range(n_samples):
                Xi = X[i]
                c = y[i]
                scores = np.dot(self.w ,Xi)
                r_hat = np.argmax(scores)

                if r_hat!= c:

                    self.w[c] += self.alpha*Xi
                    self.w[r_hat]-=self.alpha*Xi
        return self

    def predict(self,X):
        scores = np.dot(X,self.w.T)
        return np.argmax(scores,axis=1)
