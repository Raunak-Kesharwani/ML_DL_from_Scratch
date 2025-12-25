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
        return X @ self.coef_ + self.intercept_
