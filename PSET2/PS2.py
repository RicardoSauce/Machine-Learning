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

pd.set_option('display.max_columns', 500)

#git init
#git add <folder1> <folder2> <etc.>
#git commit -m "Your message about the commit"
#git remote add origin https://github.com/yourUsername/yourRepository.git
#git push -u origin master
#git push origin master

path = os.getcwd()
auto = pd.read_csv(os.path.join(path, 'Data-Auto.csv'), engine = 'python')
auto.describe()
auto['mpg01'] = np.where(auto['mpg']  > 22.75, 1, 0)
#11B) Which of the other
#features seem most likely to be useful in predicting mpg01?
for col in auto.iloc[:,1:9].columns:
    print(col)
    print(auto['mpg01'].corr(auto[col]))
    sns.scatterplot(auto[col],auto['mpg01'])
    #sns.boxplot(auto[col],auto['mpg01'],)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('mpg')
    plt.show()

# check correlation
# I feel as if cylinders, displacement, horsepower, weight, and acceleration are good predictors
# for mpg01. They each have high correlations and intuitively make sense in describing miles per gallon.

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
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# We'll import the LinearDiscriminantAnalysis class from scikit-learn package 
# and build our classifier using the default parameters. 

lda_model = LinearDiscriminantAnalysis()
# Let's train our model using the fit method
lda_model.fit(X_train, y_train)
# Using predict method to find the set of predictions
y_pred = lda_model.predict(X_test)
# Calculating the the percentage of correctly classified labels from the test set
print('Accuracy score for LDA')
print(accuracy_score(y_test, y_pred))
test_error = 1 - accuracy_score(y_test, y_pred)
print('Test Error for LDA')
print(test_error)

#11E) QDA
# Simlarly, we'll use the QuadraticDiscriminantAnalysis class to build a QDA model on our data
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
y_pred = qda_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
test_error = 1 - accuracy_score(y_test, y_pred)
print('Test Error for a QDA')
print(test_error)

#11F) Logistic Regression
logR = LogisticRegression(random_state=0).fit(X, y)
y_predict = logR.predict(X_test)
score = accuracy_score(y_test,y_predict)
print(score)
print('Logistic Regression')
print(logR.score(X,y))
test_error = 1 - logR.score(X,y)
print('Test Error for a Logistic Regression')
print(test_error)

#11G) KNN on training data 
# The KNeighborsClassifier class from scikit-learn will allow us to build a KNN model
# We're setting the number of neighbors to 4 below
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)


#Chapter 5
#A
default = pd.read_csv(os.path.join(path, 'Data-Default.csv'), engine = 'python')
print(default)
X = default[['balance','income']]
#print(X)
y = default['default']
d = {'Yes': True, 'No': False}
y = y.map(d)

#5A
regA = LogisticRegression(random_state=123).fit(X, y)
#predict = regA.predict(X)
predict = regA.predict(X)
score = accuracy_score(y,predict)
print('Accuracy score')
print(score)
print('Validation set error')
test_error = 1- score
print(test_error)

#5B 
#i)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=123)
#ii
fiveB = LogisticRegression().fit(X_train, y_train)
y_predict = fiveB.predict(X_val)
score = accuracy_score(y_val,y_predict)
print('Accuracy score')
print(score)
print('Validation Set Error')
solution = 1 - score
print(solution)

# iii
y_posterior = fiveB.predict_proba(X_val)
y_posterior = pd.DataFrame(data = y_posterior)
#print(y_posterior)
y_posterior['default'] = y_posterior[1]>0.5
y_predict = y_posterior['default']
validation_set_error = 1 - accuracy_score(y_val, y_predict)
validation_set_error

#C
for i, seed in enumerate([13, 121, 365]):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)
    fiveB = LogisticRegression().fit(X_train, y_train)
    y_posterior = fiveB.predict_proba(X_val)
    y_posterior = pd.DataFrame(data = y_posterior)
    y_posterior['default'] = y_posterior[1]>0.5
    y_predict = y_posterior['default']
    #score = accuracy_score(y_val,y_predict)
    validation_set_error = 1 - accuracy_score(y_val, y_predict)
    
    print("Validation set error with set", i+1 ," is: " , validation_set_error)

#C) In this case, I observe a validation set error that is different
# 100% of the time, with only one set providing a similar validation 
# error as my answer in part B.

