import numpy as np
import math

class logistic_regression:

    def __init__ (self, theta=None, alpha=0.1, lamda=1, 
                  threshold=0.001, max_iterations=10000, verbose=False):
        self.theta = theta
        self.alpha = alpha
        self.lamda = lamda
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.verbose = verbose
        
    def sigmod(self, x):
        if x < 0:
            return 1 - 1/ (1+math.exp(x))
        else:
            return (1.0 / ( 1 + math.exp(-1 * x)))

    def predict(self, X):
        return X.dot(self.theta).apply(lambda x: self.sigmod(x))
    
    def fit (self, X, Y):
        convergence_rate = []
        if self.theta is None:
            self.theta = np.zeros(X.shape[1], dtype=np.float)
        n = X.shape[0] #size of the data
        m = X.shape[1] #number of features
        for iteration in range(0, self.max_iterations):
            hx = self.predict(X)
            #print "hx is ", hx
            gradient = (1.0/n) * X.T.dot(hx - Y) + (float(self.lamda)/n * self.theta)
            #print "gradient", gradient
            self.theta -= (self.alpha * gradient)

            loss = np.linalg.norm(gradient)
            convergence_rate.append(loss)
            
            #consider adding difference in gradient as the convergence rate as gradient often diminish
            if loss < self.threshold:
                break;
        '''
        Plot and visualization of the converence rate. Verbose option will trigger this output
        '''
        if self.verbose is True:
            print "Total Iteration: ", iteration
            print "Final loss: ", loss
            plt.plot(convergence_rate)
            
    def score (self, X, Y):
        '''
        Helper function to score the accuracy of the model

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples,)
            True labels for X.

        Returns
        -------
        out : float
            Accuracy of the model in %.
        '''
        m = Y.shape[0]
        prediction = self.predict(X).apply(lambda x: 1 if x > 0.5 else 0)
        error_count = 0.0
        for x in range(0, m):
            if prediction.iloc[x] != Y.iloc[x]:
                error_count += 1
        error = (1 - (error_count / float(m)))
        return error
