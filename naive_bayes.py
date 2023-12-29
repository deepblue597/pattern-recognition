# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:57:04 2023

@author: iason

we will use the Gaussian Naive Bayes 
"""
#%% NaiveBayes 

import numpy as np 

class NaiveBayes: 
    
    def fit(self , X , y) : 
        
        n_samples , n_features = X.shape 
        self._classes = np.unique(y)  
        n_classes = len(self._classes) 
        
        #calculate mean , var and prior for each class 
        self._mean = np.zeros(n_classes  , n_features )
        self._var = np.zeros(n_classes  , n_features )
        self._priors = np.zeros(n_classes)  
        
        for i , c in enumerate(self._classes): 
            X_c = X[y==c] #samples of the class
            self._mean[i, :] = X_c.mean(axis=0) 
            self._var[i, :] = X_c.var(axis=0) 
            self._priors[i] = X_c.shape[0]  / float(n_samples)

    
    def predict(self , X) : 
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred) 
    
    def _predict(self , x) : 
        posteriors = [] 
        
        # caclculate th eposterior probabiliti for each class 
        for i , c  in enumerate(self._classes) : 
          prior = np.log(self._priors[i])
          posterior = np.sum(np.log(self._pdf(i , x)))
          posterior = posterior + prior 
          posteriors.append(posterior) 
          
          
        #return the class with the highest posterior 
        
        return self._classes[np.argmax(posteriors)] 
    
    def _pdf(self , class_i , x): 
        mean = self._mean[class_i] 
        var = self._var[class_i]  
        numerator = np.exp(-((x-mean)**2 /( 2 * var)))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator 
    

#%% 

