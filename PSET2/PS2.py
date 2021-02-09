#Coded by Ricardo Saucedo, with assistance by class TAs
#PPHA 30545: Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
pd.set_option('display.max_columns', 50)

#Chapter 4, 11
#

path = os.getcwd()
auto = pd.read_csv(os.path.join(path, 'Data-Auto.csv'), engine = 'python')
auto.describe()
auto['mpg01'] = np.where(auto['mpg']  > 22.75, 1, 0)
#default['student01'] = np.where(auto['student'] == 'Yes',1,0)
'''
#11B) Which of the other
#features seem most likely to be useful in predicting mpg01?
for col in auto.iloc[:,1:9].columns:
    print(col+' correlation to mpg01')
    print(auto['mpg01'].corr(auto[col]))
    sns.scatterplot(data = auto, x = col, y = 'mpg01')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('mpg')
    plt.show()
'''
# check correlation
# I feel as if cylinders, displacement, horsepower, weight, and acceleration are good predictors
# for mpg01. They each have high correlations and intuitively make sense in describing miles per gallon.
# Even if they are negative, I believe these predictors are still valuable to add.

#11C) 
# We'll split our dataframe in to the predictors (X) and the label (y)
X = auto[['cylinders','displacement','horsepower','weight','acceleration']]
y = auto['mpg01']

# We can specify the fraction of the test size using test_size paramter
# random_state allows us to specify a seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=420)


#11D)
# We can specify the fraction of the test size using test_size paramter
# random_state allows us to specify a seed for reproducibility
# We'll import the LinearDiscriminantAnalysis class from scikit-learn package 
# and build our classifier using the default parameters. 
print()
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
y_pred = lda_model.predict(X_test)
print()
# Calculating the the percentage of correctly classified labels from the test set
print('Accuracy score for LDA model')
print(accuracy_score(y_test, y_pred))
test_error = 1 - accuracy_score(y_test, y_pred)
print('Test Error for LDA model')
print(test_error)
print()

#11E) QDA
# Simlarly, we'll use the QuadraticDiscriminantAnalysis class to build a QDA model on our data
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
y_pred = qda_model.predict(X_test)
print('Accuracy score for QDA model')
print(accuracy_score(y_test, y_pred))
test_error = 1 - accuracy_score(y_test, y_pred)
print('Test Error for QDA model')
print(test_error)
print()

#11F) Logistic Regression
logR = LogisticRegression(random_state=0).fit(X, y)
y_predict = logR.predict(X_test)
score = accuracy_score(y_test,y_predict)
print('Accuracy for Logistic Regression model')
print(score)
test_error = 1 - score
print('Test Error for a Logistic Regression model')
print(test_error)
print()

#11G) KNN on training data 
# 
print('K-nearest Neighbors')
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print('KNN model accuracy')
print(accuracy_score(y_test, y_pred))
test_error = 1 - accuracy_score(y_test, y_pred)
print('Test error of KNN')
print(test_error)
print()



#Chapter 5
print()
print('Chapter 5 ')
print()
#A
default = pd.read_csv(os.path.join(path, 'Data-Default.csv'), engine = 'python')
default['student01'] = np.where(default['student'] == 'Yes',1,0)
#print(default)
X = default[['balance','income']]
#print(X)
y = default['default']
d = {'Yes': True, 'No': False}
y = y.map(d)
print()

#5A
print('part A')
regA = LogisticRegression(random_state=123).fit(X, y)
#predict = regA.predict(X)
predict = regA.predict(X)
score = accuracy_score(y,predict)
print('Accuracy score')
print(score)
print('Validation set error')
test_error = 1- score
print(test_error)
print()
#5B 

#i)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=123)
#ii
print('part B')
print('part ii: Multiple logistic regression')
fiveB = LogisticRegression().fit(X_train, y_train)
y_predict = fiveB.predict(X_val)
score = accuracy_score(y_val,y_predict)
print('Accuracy score')
print(score)
print('Validation Set Error')
solution = 1 - score
print(solution)

# part iii
y_posterior = fiveB.predict_proba(X_val)
y_posterior = pd.DataFrame(data = y_posterior)
#print(y_posterior)
y_posterior['default'] = y_posterior[1]>0.5
#part iv
print('part iv')
y_predict = y_posterior['default']
validation_set_error = 1 - accuracy_score(y_val, y_predict)
print('Validation Set Error')
print(validation_set_error)

#C
print()
print('part C')
for i, seed in enumerate([13, 121, 365]):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)
    fiveB = LogisticRegression().fit(X_train, y_train)
    y_posterior = fiveB.predict_proba(X_val)
    y_posterior = pd.DataFrame(data = y_posterior)
    y_posterior['default'] = y_posterior[1]>0.5
    y_predict = y_posterior['default']
    validation_set_error = 1 - accuracy_score(y_val, y_predict)
    
    print("Validation set error with set", i+1 ," is: " , validation_set_error)

#C) In this case, I observe a validation set error that is different
# 100% of the time. 


#D) logistic regression model that predicts probability of default using income, balance, and student.
print()
print('part D')
X = default[['balance','income','student01']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=124)
fiveD = LogisticRegression().fit(X_train, y_train)
y_predict = fiveD.predict(X_val)
#y_posterior = pd.DataFrame(data = y_posterior)
#print(y_predict)
score = accuracy_score(y_val,y_predict)
print('Accuracy score Using Balance, Income, and Student status')
print(score)
print('Test Error by using Validation Set ')
validation_set_error = 1 - score
print(validation_set_error)

