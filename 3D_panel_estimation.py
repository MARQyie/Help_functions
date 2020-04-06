# OLS estimation
''' This script contains all the methods to estimate a three-dimensional
    panel data with OLS. The script contains the following methods:
        
        1. Fixed effects data transformation
        2. OLS estimation of the transformed data
        3. (Multi-way) standard error clustering
'''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import numpy as np
import pandas as pd

#------------------------------------------------------------
# Regression class and nested methods
#------------------------------------------------------------

# TODO: DONT FORGET INHERITANCE
class MultiDimensionalOLS:
    
    # Initialize the class
    def __init__(self, fit_intercept = True, clustered = True, cluster_column):
        self.params = None
        self.std = None
        self.resid = None
        self.cluster_id = None
        
        self.fit_intercept = fit_intercept
        self.clustered = clustered 
        self.cluster_column = cluster_column
               
    # Calculate parameters, residuals, and standard deviation
    def fit(self, X, y):
        ''' Calculates the parameters by OLS.
        
            Arguments:
                X: 1D or 2D numpy array
                y: 1D numpy array
        '''
        
        # Check if X is 1D or 2D. If 1D transform to column vector
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        # Add constant
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        # Get N and K from X
        n, k = X.shape
        
        # Calculate parameters with OLS
        xtx = np.dot(X.T,X)
        xtx_inv = np.linalg.inv(xtx)
        xty = np.dot(X.T,y)
        params = np.dot(xtx_inv, xty)
        
        # Calculate the residuals
        resid = np.subtract(y, np.dot(X, params))
        
        # Calculate standard deviation
        if clustered:
            pass
        else:
            resid_sqr = np.power(resid, 2)
            sigma = np.divide(np.sum(resid_sqr), (n - k))
            std = np.dot(sigma, xtx_inv)
        
        # Set attributes
        self.params = params
        self.resid = resid
        self.std = std
        
        
            