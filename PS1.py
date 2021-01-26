#!/usr/bin/env python
# coding: utf-8

# # Ricardo Saucedo #
# ## PPHA 30545 Problem Set  1: Machine Learning ##
# ### Due 1/25/2021 ###
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#create path for us to obtain our file
path = os.getcwd()


# In[2]:



#Chapter 2, Exercise 10:
boston = pd.read_csv(os.path.join(path, 'Boston.csv'), engine = 'python')
#A: 
#How many rows are in this data set?
#How many columns?
#What do the rows and columns represent?
boston = pd.read_csv(os.path.join(path, 'Boston.csv'), engine = 'python')
#boston


# In[3]:


#A: 
#How many rows are in this data set?
#506 rows
#How many columns?
#14 columns
#What do the rows and columns represent?
#This data comes from a study where Boston housing market data is used to generate
#quantitative estimates of the willingess to pay for air quality
#improvements, where each observation contains different neighborhoods
#variables/indicators in Boston. It is necessary to isolate 
#independent influence of air pollution to reduce bias in drawn
#conclusions.


# In[4]:


#B:
#Make some pairwise scatterplots of the predictors in this data
#set. Describe your findings.

sns.pairplot(boston, vars = ['CRIM','NOX','RM','AGE', 'B', 'TAX', 'MDEV', 'LSTAT'])


# In[5]:


#My findings show that throughout Boston there persists a high volume of nitric oxide concentrations.
#The higher proportion of Blacks neighborhoods that are subject to higher levels of NOX gives evidence
#to the notion of environmental oppression, as well as the narrative of having higher crime rates.


# In[6]:


#C:
#Are any of the predictors associated with per capita crime rate? If so, explain the relationship.

#Nitric oxide levels, neighborhoods with higher representation of Blacks in a town, the median value of
#owner-occupied homes, and a larger share of being in the lower percentage status in comparison to the population.


# In[7]:


#D:
#Do any of the suburbs of Boston appear to have particularly high crime rates? Tax rates? Pupil-teacher ratios?
#Comment on the range of each predictor.
boston_Min = boston.describe().iloc[3]
boston_Max = boston.describe().iloc[7]
boston_Range = boston_Max-boston_Min

boston_CRIM = boston.sort_values(by='CRIM', ascending = False)
boston_TAX = boston.sort_values(by='TAX', ascending = False)
boston_PTRATIO = boston.sort_values(by='PTRATIO',ascending = False)
print(boston_CRIM.head())
print(boston_TAX.head())
print(boston_PTRATIO.head())
print()
print(boston_Range)
#When we look at the range of some of the predictors, we can see that there are large disparities between
#crime, tax rate, and pupil-teacher ratio by town. Where the tax rate is highest, crime is relatively low.
#Where crime is low, the pupil-teacher is usually pretty high. In the range of these three predictors,
#it is evident that some neighborhoods are subject to higher rates of unequal opportunity to prosper.


# In[8]:


#E: How many of the suburbs in this data set bound the Charles river?
boston[boston['CHAS']== 1].shape
#We see there are 35 suburbs in this data set that bound the Charles river.


# In[9]:


#F: What is the median pupil-teacher ratio among the towns in this data set?
boston.describe()
#We see that the median is 19.05 for pupil-teacher ratio.


# In[10]:


#G: Which suburb of Boston has lowest median value of owner-occupied homes?
#What are the values of the other predictors for that suburb, and how do those values compare to the overall
#ranges for those predictors? Comment on your findings.
boston_MEDV = boston.sort_values(by='MDEV', ascending = True)
boston_MEDV.head()

#Suburbs '398' and '405' have the lowest median value of owner-occupied homes, and the other predictors are
#just as alarming. Crime is high, all units were built prior to 1940, hold a high proportion of blacks in
#the town, and are at a higher percentage of being the lower status in the Boston population. The 5 suburbs that
#have the lowest median value of owner-occupied homes also do have high levels of nitric oxides concentration.


# In[11]:


#H: In this data set, how many of the suburbs average more than seven rooms per dwelling?
#More than eight rooms per dwelling?
#Comment on the suburbs that average more than eight rooms per dwelling. 

dwellings_Boston = boston[boston['RM'] >= 7].sort_values(by='RM',ascending = False)
#64 average more than seven rooms per welling.
#boston[boston['RM'] >= 8].shape
#13 suburbs average more than 8 rooms per their dwelling. 
dwellings_Boston.head()
dwellings_Boston.tail()
#My understanding of the suburbs that averamge more than eight rooms per dwelling are either due to
#high rent driving households to increase the amount dwelling in their spaces. I assume that these still
#could be suburbs that suffer from high crime rates, higher levels of NOX, and and yet some surprising predictor
#values (low-vs-high median value of owner-occupied homes, fluctuating tax rates, and varying pupil-teacher ratio).


# ### Chapter 3 ###

# In[12]:


import statsmodels.formula.api as smf

uni_List = []
for col in boston.iloc[:,1:14].columns:
    result = smf.ols('CRIM ~ boston[col]', data = boston).fit()
    params = result.params
    #uni_List.append(params)
    print('OLS results for predictor:'+col)
    print()
    print(result.summary())
    
uniReg_Dict = {'0':[0.1071, -0.0355, -1.8715, -1.5428, 0.5068, 0.5444, -0.3606, 30.9753, 1.1446, 0.6141, -2.6910, 0.0296, 4.4292]}
uni_Reg = pd.DataFrame(data = uniReg_Dict)
uni_Reg


# In[13]:


for col in boston.iloc[:,1:14].columns:
    sns.regplot(x = col,y = 'CRIM', data = boston)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('CRIM')
    plt.show()


# In[14]:


#B
predictors = ' + '.join(boston.columns.difference(['CHAS','CRIM']))
results = smf.ols('CRIM ~ {}'.format(predictors),data = boston).fit()
print(results.summary())


# In[15]:


multiReg_results = results.params
multiReg_results

multi_Reg = multiReg_results.to_frame().reset_index()
multi_Reg = multi_Reg[2:14]

uni_Reg
multi_Reg
joined_DF = multi_Reg.join(uni_Reg)
joined_DF
joined_DF.columns = ['index','multi','uni']
joined_DF
plt.figure(figsize=(12,8))
sns.regplot(x = 'uni',y = 'multi', data = joined_DF)
plt.xlabel("Univariate regression coefficients", fontsize=15)
plt.ylabel("Multivariate regression coefficients", fontsize=15)
plt.title('Comparison between multivariate and univariate coefficients', fontsize=20)


# In[17]:


#D
#Is there evidence of non-linear association between any of the predictors and the response?
#To answer this question, for each predictor X, fit a model of the form:
predictors = ' + '.join(boston.columns.difference(['CHAS','CRIM']))

result_D = smf.ols(formula = 'CRIM ~ {} + np.power(DIS, 2) + np.power(DIS,3)'.format(predictors),data = boston).fit()
result_D.summary()


# In[ ]:




