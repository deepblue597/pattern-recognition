"""
Created on Sun Nov  5 12:29:10 2023

@author: iasonas kakandris 
@author: kostantina Gatzi

Course : Pattern Recognition
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

#%% knn classifiers initialization

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = [] #list of classifiers
classifiers  = 10
y_prediction = [] #predictions of classifiers


#initialization of the knn classifiers
for i in range(classifiers):
  knn = KNeighborsClassifier(n_neighbors=i+1)
  knn_classifier.append(knn)

#%% model fit
# fit the model to the training set
for i in range(classifiers) :
  knn_classifier[i].fit(X_train , y_train)
  y_prediction.append(knn_classifier[i].predict(X_test))
  
  
#%% accuracy score 

for i in range(classifiers) :
  print('Model accuracy score for ' , i+1 , ' neighboor(s): {0:0.4f}'. format(
    accuracy_score(y_test, y_prediction[i])))
  

#%% training set score 

y_prediction_train = []

for i in range(classifiers) :
  y_prediction_train.append(knn_classifier[i].predict(X_train))
  print('Training-set accuracy score for' , i+1 ,' neighboor(s): {0:0.4f}'. format(
    accuracy_score(y_train, y_prediction_train[i])))
  

#%% Confusion Matrix 

class_names = ['1' , '2' , '3']

for i in range(classifiers):
  print("Number of neighbors is {}".format(i+1))
  # Create the  confusion matrixes
  cm = confusion_matrix(y_test, y_prediction[i])
  heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                          xticklabels=class_names, yticklabels=class_names)
  heatmap.set_title("Confusion matrix for {} neighbor(s)".format(i+1))
  #plt.figure(figsize=(5,4))

  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  print(classification_report(y_test, y_prediction[i]))
  plt.show()
  print("\n")
  print("\n")
  print("\n")
  
  
#%% Data presentation 

for i in range(classifiers):
  class1 =  1
  # Find indices of data belonging to the specified classes
  indices_to_keep1 = np.isin(y_train, class1).flatten()
  # Filter the data based on the selected indices
  X_train1= X_train[indices_to_keep1]

  class2=  2
  # Find indices of data belonging to the specified classes
  indices_to_keep2 = np.isin(y_train, class2).flatten()
  # Filter the data based on the selected indices
  X_train2= X_train[indices_to_keep2]

  class3=  3
  # Find indices of data belonging to the specified classes
  indices_to_keep3 = np.isin(y_train, class3).flatten()
  # Filter the data based on the selected indices
  X_train3= X_train[indices_to_keep3]

 # plt.figure(figsize=(4,4))
  plt.subplot(2, 2, 1)
  plt.scatter(X_train1[:,0],X_train1[:,1])
  plt.scatter(X_train2[:,0],X_train2[:,1])
  plt.scatter(X_train3[:,0],X_train3[:,1])
  plt.legend(['Class 1','Class 2', 'Class 3'])
  plt.title('Data of train set')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  #plt.show()


  class1 =  1
  # Find indices of data belonging to the specified classes
  indices_to_keep4 = np.isin(y_prediction_train[i], class1).flatten()
  # Filter the data based on the selected indices
  X_train4= X_train[indices_to_keep4]

  class5=  2
  # Find indices of data belonging to the specified classes
  indices_to_keep5 = np.isin(y_prediction_train[i], class2).flatten()
  # Filter the data based on the selected indices
  X_train5= X_train[indices_to_keep5]

  class6=  3
  # Find indices of data belonging to the specified classes
  indices_to_keep6 = np.isin(y_prediction_train[i], class3).flatten()
  # Filter the data based on the selected indices
  X_train6= X_train[indices_to_keep6]
  #plt.figure(figsize=(4,4))
  plt.subplot(2, 2, 2)
  plt.scatter(X_train4[:,0],X_train4[:,1])
  plt.scatter(X_train5[:,0],X_train5[:,1])
  plt.scatter(X_train6[:,0],X_train6[:,1])
  plt.legend(['Class 1','Class 2', 'Class 3'])
  plt.title('Data of predicted train set')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  #plt.show()

  class1 =  1
  # Find indices of data belonging to the specified classes
  indices_to_keep7 = np.isin(y_test, class1).flatten()
  # Filter the data based on the selected indices
  X_test1= X_test[indices_to_keep7]

  class2=  2
  # Find indices of data belonging to the specified classes
  indices_to_keep8 = np.isin(y_test, class2).flatten()
  # Filter the data based on the selected indices
  X_test2= X_test[indices_to_keep8]

  class3=  3
  # Find indices of data belonging to the specified classes
  indices_to_keep9 = np.isin(y_test, class3).flatten()
  # Filter the data based on the selected indices
  X_test3= X_test[indices_to_keep9]

  #plt.figure(figsize=(4,4))
  plt.subplot(2, 2, 3)
  plt.scatter(X_test1[:,0],X_test1[:,1])
  plt.scatter(X_test2[:,0],X_test2[:,1])
  plt.scatter(X_test3[:,0],X_test3[:,1])
  plt.legend(['Class 1','Class 2', 'Class 3'])
  plt.title('Data of test set')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  #plt.show()


  class1 =  1
  # Find indices of data belonging to the specified classes
  indices_to_keep10 = np.isin(y_prediction[i], class1).flatten()
  # Filter the data based on the selected indices
  X_test4= X_test[indices_to_keep10]

  class5=  2
  # Find indices of data belonging to the specified classes
  indices_to_keep11 = np.isin(y_prediction[i], class2).flatten()
  # Filter the data based on the selected indices
  X_test5= X_test[indices_to_keep11]

  class6=  3
  # Find indices of data belonging to the specified classes
  indices_to_keep12 = np.isin(y_prediction[i], class3).flatten()
  # Filter the data based on the selected indices
  X_test6= X_test[indices_to_keep12]
  #plt.figure(figsize=(4,4))
  plt.subplot(2, 2, 4)
  plt.scatter(X_test4[:,0],X_test4[:,1])
  plt.scatter(X_test5[:,0],X_test5[:,1])
  plt.scatter(X_test6[:,0],X_test6[:,1])
  plt.legend(['Class 1','Class 2', 'Class 3'])
  plt.title('Data of predicted test set')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')


  plt.suptitle("Number of neighbors is {}".format(i+1))
  plt.show()
  
# =============================================================================
#   We can see that the best model is the one with the 5 neighboors 
#   Model accuracy score for  5  neighboor(s): 0.8929
#   Training-set accuracy score for 5  neighboor(s): 0.9000
#   we also see that we dont have overfitting
# =============================================================================
