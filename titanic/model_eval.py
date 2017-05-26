import numpy as np

class model_eval:
    '''
    Set of model evaluation functions to help evaluate the performance of model
    '''
    
    def __init__ (self, classifer):
        self.classifer = classifer
        
    def split_data(data, percent):
        train_size = len(data) * percent * 0.01
        train_size = int(train_size)
        return data.iloc[:train_size, :], data.iloc[train_size:, :]

    def k_cross_validation (self,K, X, Y):
        '''
        Main function to perform cross validation by splitting data and averaging performances 
        Divide data into k parts and rotato to score the model. 
        Note: Please use subset of the data to find the hyper parameter and use full set to train the final model
        Return the average performance of the model to avoid bias

        Parameters
        ----------
        K: integer (prefer range 1-20) 
            Divide data into k parts, take 1 part as test and rest as train data

        X : array-like dataframe, shape = (n_samples, n_features)
            Test samples.

        y : array-like dataframe, shape = (n_samples,)
            True labels for X.

        Return:
        ----------
        out: float of the accuracy of the model
        '''
        m = X.shape[0]
        size = m / K
        print size
        #form the array of inputable data
        total_performance = 0
        for i in range(0, K):
            if (size > 0 and size < m):
                test_input = X.iloc[size:size+size]
                test_output = Y.iloc[size:size+size]
                train_input = X.iloc[0:size].append(X.iloc[size+size:])
                train_output = Y.iloc[0:size].append(Y.iloc[size+size:])
            else:
                test_input = X.iloc[0:size]
                test_output = Y.iloc[0, size]
                train_input = X.iloc[size:].append(X.iloc[size:])
                train_output = Y.iloc[size].append(Y.iloc[size:])
            #train a new model base on the training data, and test data will be used to avoid bias
            model=self.classifer.fit(train_input, train_output)
            total_performance += self.classifer.score(test_input, test_output)
        return (total_performance / float(K))
