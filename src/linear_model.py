import numpy as np 

# Linear Regression using OLS 
class LinearRegression :

    def __init__(self):
        self.coef_ = None 
        self.intercept_ = None 
        
        
    def fit (self , X , y ):
        X = np.insert(X , 0 , 1 , axis = 1 )
        
        #β* = (X_transposeX)-¹X_transposey
        weights = np.linalg.inv((X.T@X))@(np.transpose(X))@y
        self.intercept_ = weights[0]
        self.coef_  = weights[1:]

    def predict(self, X):
        return np.dot(X,self.coef_) + self.intercept_
        
        
        
class LinearRegressionGD:

    def __init__(self, lr=0.01, epoches=50):
        self.lr = lr
        self.epoches = epoches
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        weights = np.zeros(X.shape[1])

        for i in range(self.epoches):
            y_pred = X @ weights
            gradient = (X.T @ (y - y_pred)) / X.shape[0]
            weights = weights + self.lr * gradient

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict(self, X):
        return np.dot(X,self.coef_) + self.intercept_
    
    import numpy as np

class LinearRegressionSGD:

    def __init__(self, lr=0.01, epoches=10):
        self.lr = lr
        self.epoches = epoches
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # add bias term
        X = np.insert(X, 0, 1, axis=1)
        weights = np.zeros(X.shape[1])

        n = X.shape[0]

        for _ in range(self.epoches):
            for _ in range(n):
                r = np.random.randint(0, n)
                y_pred = X[r] @ weights
                error = y[r] - y_pred
                gradient = error * X[r]          # <-- vector gradient
                weights = weights + self.lr * gradient

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict(self, X):
        return X @ self.coef_ + self.intercept_        


class Perceptron:

    def __init__(self):
        self.weights = None

    def step(self , z):
        return 1 if z > 0 else 0

    def fit(self, X, y, lr=0.1, epochs=1):
        # add bias term
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.ones(X.shape[1])

        for _ in range(epochs):
            for i in range(X.shape[0]):
                j = np.random.randint(0, X.shape[0])
                y_hat = self.step(X[j] @ self.weights)
                self.weights = self.weights + lr * (y[j] - y_hat) * X[j]

        return self.weights[0], self.weights[1:]
    


    # logisticRegression using sgd

class LogisticRegressionSGD:

    def __init__(self, lr=0.01, epoches=10):
        self.lr = lr
        self.epoches = epoches
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # add bias term
        X = np.insert(X, 0, 1, axis=1)
        weights = np.zeros(X.shape[1])

        n = X.shape[0]

        for _ in range(self.epoches):
            for _ in range(n):
                r = np.random.randint(0, n)
                z = X[r] @ weights
                y_pred = self._sigmoid(z)
                error = y[r] - y_pred
                gradient = error * X[r]          # SGD gradient
                weights = weights + self.lr * gradient

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict(self, X):
        preds = []
        for x in X:
            z = x @ self.coef_ + self.intercept_
            p_hat = self._sigmoid(z)
            preds.append(1 if p_hat >= 0.5 else 0)
        return np.array(preds)
