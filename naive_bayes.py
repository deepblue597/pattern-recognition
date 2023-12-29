# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:57:04 2023

@author: iason

we will use the Gaussian Naive Bayes 
"""

import numpy as np 

class NaiveBayes: 
    
    def fit(self , X , y) : 
        
        n_samples , n_features = X.shape 
        self._classes = np.unique(y)  
        n_classes = len(self._classes) 
        
        #calculate mean , var and prior for each class 
        self._mean = np.zeros(n_classes  , n_features )
        self._var = np.zeros(n_classes  , n_features )
        self._prior = np.zeros(n_classes)  
        
        

    
    def predict(self , X) : 
        pass 
