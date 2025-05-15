import numpy as np


class perceptron:
    def __init__(self, alpha = 0.01,lambda_  = 0.01,epochs = 1000):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epochs = epochs
        self.w  = None
        self.losses  =[]

    def fit(self,X,Y):
        y = np.array(Y)
        assert np.all(np.unique(y) == np.array([0,1])),"labels must be 0 or 1"

        n_samples,n_features = X.shape
        self.w = np.zeros(n_features)
        self.losses = []

        for epoch in range(self.epochs):
            shuffle_index = np.random.permutation(n_samples)
            X_shuffled = X[shuffle_index]
            y_shuffled = y[shuffle_index]

            for Xi , yi in zip(X_shuffled,y_shuffled):
                dot_product = np.dot(Xi,self.w)
                pred  = 1 if dot_product >= 0 else 0

                if yi * pred <1:
                    self.w = self.w +(1-self.alpha*self.lambda_) +self.alpha*yi*Xi
                else:

                    self.w = (1-self.alpha*self.lambda_)
            scores = np.dot(X,self.w)
            losses = np.maximum(0,1-y*scores)
            avg_loss = np.mean(losses)
            self.losses.append(avg_loss)
        return self
    def predict(self,X):
        dot_product = np.dot(X,self.w)
        return np.where(dot_product >= 0,1,0)