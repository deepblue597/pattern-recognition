# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:26:12 2023

@author: iason
"""

#%% import libraries 

import pandas as pd
import matplotlib.pyplot as plt  # for data visualization purposes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix  # To check for TP , TN , FP , FN
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification

# import SVC classifier
from sklearn.svm import SVC
import time 

#%% Import data 
# Get the current working directory
current_dir = os.getcwd()

# Define the file name
file_name = 'dataset.csv'  # Adjust the file name as needed

# Create the file path by joining the current directory and the file name
data = os.path.join(current_dir, file_name)

#data = 'C:/pattern-recognition/dataset.csv'

df = pd.read_csv(data , header = None)

df.shape

#%% change to nd array for easiness

df = pd.DataFrame(df).to_numpy()


#%% load the X and y data 

X=df[:,0:-1]
y=df[:,-1]
#%% training and test data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

#%% Scaling 

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#%%  Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.

# instantiate classifier with default hyperparameters
svc=SVC() 

#%% linear kernel 

# instantiate classifier with polynomial kernel and C=1.0
linear_svc=SVC(kernel='linear') 

# degree int, default=3 
start_time = time.perf_counter()
# fit classifier to training set
linear_svc.fit(X_train,y_train)

end_time = time.perf_counter()

#%% 
# make predictions on test set
y_pred=linear_svc.predict(X_test)

# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
#Model accuracy score with linear kernel and C=1.0: 0.8000
print(end_time-start_time)

#%% training set

# Plot training set
plt.figure(figsize=(6, 6))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=100, linewidth=1, cmap=plt.cm.Paired)

plt.title('Scatter Plot of Training Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()
#%% test set

# Plot training set
plt.figure(figsize=(6, 6))

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='o', s=100, linewidth=1, cmap=plt.cm.Paired)

plt.title('Scatter Plot of Test Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()


#%% false classifications

# Scatter plot of training set
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, edgecolors='k', marker='o', s=100, linewidth=1, cmap=plt.cm.Paired)

# Circle misclassified samples in the test set
misclassified_indices = np.where(y_test != y_pred)[0]
plt.scatter(X_test[misclassified_indices, 0], X_test[misclassified_indices, 1], facecolors='none', edgecolors='r', marker='o', s=300, linewidth=2, label='Misclassified')

plt.title('Training Set with Circled Misclassified Test Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()




#%% rbf 

# import GridSearchCV
from sklearn.model_selection import GridSearchCV

# declare parameters for hyperparameter tuning
parameters = [  {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              ]


grid_search = GridSearchCV(estimator = svc,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)

#%% fit the data
grid_search.fit(X_train, y_train)


# examine the best model


# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))


# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))


# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))


#print(grid_search.cv_results_)
print(grid_search.refit_time_)
results = grid_search.cv_results_


#test data in gridsearch
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test , y_test)))

#%% plot 



best_estimator = grid_search.best_estimator_

# Make predictions on the test set using the best estimator
y_test_pred = best_estimator.predict(X_test)

# Plot the actual vs predicted values
# Scatter plot of training set
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, edgecolors='k', marker='o', s=100, linewidth=1, cmap=plt.cm.Paired)

# Circle misclassified samples in the test set
misclassified_indices = np.where(y_test != y_test_pred)[0]
plt.scatter(X_test[misclassified_indices, 0], X_test[misclassified_indices, 1], facecolors='none', edgecolors='r', marker='o', s=300, linewidth=2, label='Misclassified')

plt.title('Training Set with Circled Misclassified Test Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()

#%%  linear svm
from mlxtend.plotting import plot_decision_regions
# Plotting decision regions
plot_decision_regions(X_test, y_test.astype(np.int_), clf=best_estimator, legend=2)
plt.show() 
plot_decision_regions(X_test, y_test.astype(np.int_), clf=linear_svc, legend=2)
plt.show() 


#%% self made linear 

class SVM_from_scratch : 
    
    def __init__(self , learning_rate = 0.001 , lambda_param = 0.01 , n_iters = 1000 ): 
        self.learning_rate = learning_rate 
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None 
        #self.iteration = iteration
        #run[self.iteration]['params/learning_rate'] = learning_rate
        #run[self.iteration]['params/lambda_param'] = lambda_param
        
    def fit(self , X , y): 
        n_samples , n_features = X.shape  
        
        y_ = np.where(y <= 0 , -1 , 1)
        
        #init weights 
        self.weights = np.zeros(n_features) 
        self.bias = 0  
        
        for _ in range(self.n_iters): 
            for index , x_i in enumerate(X): 
                condition = y_[index] * (np.dot(x_i , self.weights) - self.bias) >= 1 
                
                if condition: 
                    
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights) # a -> learning rate 
                
                else: 
                    
                    self.weights  -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i , y_[index]))
                    self.bias -= self.learning_rate * y_[index] 
                    
            predicted = self.predict(X)
            accuracy = accuracy_score(y, predicted) 
            #model_version["accuracy"] = accuracy
            #run[self.iteration]["params/accuracy"].append(accuracy)
    
    def predict(self  , X): 
        approx = np.dot(X , self.weights) - self.bias
        
        return np.sign(approx)


#%%  data preperation for self-made svm 

class1, class2 = 1, 2  # You can choose the class indices 

# Filter training data and labels for the selected classes
selected_train_indices = np.where((y_train == class1) | (y_train == class2))[0]
X_train_selected = X_train[selected_train_indices]
y_train_selected = y_train[selected_train_indices]

# Filter test data and labels for the selected classes
selected_test_indices = np.where((y_test == class1) | (y_test == class2))[0]
X_test_selected = X_test[selected_test_indices]
y_test_selected = y_test[selected_test_indices]


# Convert class names to numeric labels in y_train_selected and y_test_selected
y_train_selected_numeric = np.where(y_train_selected == class1, -1, 1)
y_test_selected_numeric = np.where(y_test_selected == class1, -1, 1)

#%% 

y_train_unified = np.where((y_train == 2) | (y_train == 3), 2, y_train)

y_test_unified = np.where((y_test == 2) | (y_test == 3), 2, y_test)

y_train_unified_numeric = np.where(y_train_unified == class1, -1, 1)
y_test_unified_numeric = np.where(y_test_unified == class1, -1, 1)






#%% add the data to the self made svm 

# Define parameters
learning_rate = 0.01 #Model accuracy score with 0.001 and 0.01 200 itrs ker: 0.7710
lambda_param = 0.02 #Model accuracy score with polynomial kernel and C=1.0 : 0.8040
n_iters = 200
svm_self_made = SVM_from_scratch(learning_rate , lambda_param , n_iters) 


start_time = time.perf_counter()
svm_self_made.fit(X_train_selected, y_train_selected_numeric)
end_time = time.perf_counter()

 
predictions = svm_self_made.predict(X_test_selected)


# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test_selected_numeric, predictions)))
print(classification_report(y_test_selected_numeric, predictions))
print(end_time-start_time)
# Log metrics
accuracy = accuracy_score(y_test_selected_numeric, predictions)
#run["accuracy"].append(accuracy)
#model_version["accuracy"] = accuracy

#%% plot
plot_decision_regions(X_test_selected, y_test_selected_numeric, clf=svm_self_made, legend=2)
plt.xlabel('1st feature')
plt.ylabel('2nd feature')
plt.title('SVM from scratch')
plt.show() 

#%% unified 

start_time = time.perf_counter()
svm_self_made.fit(X_train, y_train_unified_numeric)
end_time = time.perf_counter()

 
predictions = svm_self_made.predict(X_test)


# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test_unified_numeric, predictions)))
print(classification_report(y_test_unified_numeric, predictions))
print(end_time-start_time)
# Log metrics
accuracy = accuracy_score(y_test_unified_numeric, predictions)

#%% plot
plot_decision_regions(X_test, y_test_unified_numeric, clf=svm_self_made, legend=2)
plt.xlabel('1st feature')
plt.ylabel('2nd feature')
plt.title('SVM from scratch')
plt.show() 