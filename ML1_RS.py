#Ricardo Saucedo
#IPUMS https://ipums.org/
#IPUMS Provides Census and survey data from around the world integrated across
#time and space.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# File Preparation
#create path for us to obtain our file.
#create a dataframe for our dataset and one for
#the crosswalk to map over values. 

path = os.getcwd()
ipums = pd.read_csv(os.path.join(path, 'usa_00001.csv.gz'), engine = 'python')
crosswalk = pd.read_csv(os.path.join(path,'PPHA_30545_MP01-Crosswalk.csv'), engine = 'python')


#create dict from the two columns
crosswalk_dict = dict(zip(crosswalk.educd, crosswalk.educdc))


#
#3.2 crosswalk and replace values in IPUMS data set
#crosswalk_dict
ipums['EDUCDC'] = ipums['EDUCD']
ipums['EDUCDC'].replace(crosswalk_dict, inplace = True)



#3.2: Create Dummy Variables for explanatory values
#
#
#
#high school diploma indicator
ipums['hsdip'] = np.where(ipums['EDUCDC'] == 12, 1, 0)
#using np.where to create 
#ipums = ipums['hsdip'].replace(to_replace = 'True', value = 1)

#college diploma indicator
ipums['coldip'] = np.where(ipums['EDUCDC'] == 16, 1, 0)
ipums['coldipplus'] = np.where(ipums['EDUCDC'] > 16, 1, 0)

#indicator for white individual
ipums['white'] = np.where(ipums['RACE'] == 1, 1, 0)

#ipums['white'] = ipums['RACE'] == 1
#ipums['white'] = ipums['white'].astype(int)
#indicator for black individual
ipums['black'] = np.where(ipums['RACE'] == 2, 1, 0)

#ipums['black'] = ipums['RACE'] == 2
#ipums['black'] = ipums['black'].astype(int)

#did not code for 9,'not reported'
ipums['hispan'] = np.where(ipums['HISPAN'] != 0, 1, 0)
#

#marital status
#IPUMS data is coded accordingly
ipums['married'] = np.where(ipums['MARST'] == 1 | 2, 1, 0)
#

#indicator variable for female
ipums['female'] = np.where(ipums['SEX'] == 2, 1, 0)

#indicator for being a Veteran
ipums['vet'] = np.where(ipums['VETSTAT'] == 2, 1, 0)

# Education interaction terms
ipums['educXhs'] = ipums['EDUCDC']*ipums['hsdip']
ipums['educXcol'] = ipums['EDUCDC']*ipums['coldip']

#age squared variable
ipums['ageSq'] = ipums['AGE']*ipums['AGE']

#take the log income wage
ipums['LNincwage'] = np.log(ipums['INCWAGE'])
#print(ipums['LNincwage'])
#drop -inf
ipums = ipums[ipums['LNincwage'] > 1]

#4.1
print('Year summary... in case you\'re curious')
year_Sum = ipums['YEAR'].describe()
print(year_Sum) 
print()
print('Income wage summary')
incwage_Sum = ipums['INCWAGE'].describe()
print(incwage_Sum)
print()
print('Log income wage summary')
lnincwage_Sum = ipums['LNincwage'].describe()
print(lnincwage_Sum)
print()
print('Education year count summarized')
educdc_Sum = ipums['EDUCDC'].describe()
print(educdc_Sum)
print()
print('Female summary')
female_Sum = ipums['female'].describe()
print(female_Sum)
print()
print('Age summary')
age_Sum = ipums['AGE'].describe()
print(age_Sum)
print()
print('Age square summary')
ageSQ_Sum = ipums['ageSq'].describe()
print(ageSQ_Sum)
print()
print('White summary')
white_Sum = ipums['white'].describe()
print(white_Sum)
print()
print('Black summary')
black_Sum = ipums['black'].describe()
print(black_Sum)
print()
print('Hispanic summary')
hispan_Sum = ipums['hispan'].describe()
print(hispan_Sum)
print()
print('Married sum')
married_Sum = ipums['married'].describe()
print(married_Sum)
print()
print('Number of children')
nchild_Sum = ipums['NCHILD'].describe()
print(nchild_Sum)
print()
print('Veteran status summary')
vet_Sum = ipums['vet'].describe()
print(vet_Sum)
print()
print('Highschool diploma summary')
hsdip_Sum = ipums['hsdip'].describe()
print(hsdip_Sum)
print()
print('College diploma summary')
coldip_Sum = ipums['coldip'].describe()
print(coldip_Sum)
print()
print('Interaction term summary')
educXhs_Sum = ipums['educXhs'].describe()
print(educXhs_Sum)
print()
educXcol_Sum = ipums['educXcol'].describe()
print(educXcol_Sum)


#4.2
plt.figure(figsize=(12,8))
sns.regplot(x= 'EDUCDC', y = 'LNincwage', data = ipums)
plt.title("Years of education against log income wage",fontsize=20)
plt.xlabel('Years of education',fontsize=15)
plt.ylabel('log income wage', fontsize=15)
plt.show()

### 4.3
import statsmodels.formula.api as smf
result = smf.ols('LNincwage ~ EDUCDC + female + AGE + ageSq + white + black + hispan + married + NCHILD + vet', data = ipums).fit()
print(result.summary())


#A
#The R-Squared is .3, which means that the fraction of the variation in log wages is explained by approximately
#30 percent of the model.

#B
# With an alpha of 0.1, we are able to reject the null at a 10 percent significance level.
#The F-Statistic is 376.9, as well as our p-value. Thus, we are certain that we are able to
#reject the null hypothesis. Furthermore, this means that our X variables all have predictive values;
#the predictors we should be careful for are indicators for being white, hispanic, NCHILD, and vet. 

#C
#Because this is the log income wage, an additional year of education yields a 9.6 percent increase in
#one's wages. This is statistically significant, with certainty in our p-values, confidence intervals, and
#t-statistic. Given that this is an econometric measure, it is also practically significant when one's
#income is at a certain threshold. The magnitude at which the amount of years of education one has affecting 
#their log wages will have a practical and realistic significance whenever the amount one gains in income is
#substantial to a 9.6 percent increase.

#D
#When we take the derivative with respects to age in our regression model and maximize, we find that
#the model predicts that an individual will achieve the highest wage at 46.36 years old.
#[attach photo]

#E
#Seeing as how the coefficient for female is negative, the model predicts that those who identify as female
#will reportedly have lower wages, even if we hold all else equal. Since the model does not account for tax 
#or tax credits, the data would otherwise support the predicted values the model produced.

#F
#The coefficient on the white variable is statistically insignificant, even at the alpha = .1 level.
#Black and hispanic variables, however, are statistically significant, both with negative effects. Therefore
#this would translate into receiving negative wages solely from the fact that one identifies as black or hispanic.

#G
# Considering the rhetoric that some would argue Hispanic not being a race, but more so an ethnicity, I will
# proceed with the assumption that race is equal to white and black for this question. 

# The Null Hypothesis would be that race has no effect on wages, namely that Beta_5 + Beta_6 = 0; thus 
# the indicator variable would have a 0 percent change in one's wages.

# Based off of my model's predictions, I will take a one sided approach.
# The alternative Hypothesis would state that race does have a negative non-zero effect on one's wage, namely 
# that Beta_5 + Beta_6 =/= 0.

# In my calculations, I conducted a linear regression of the influence race had on the response.
# Based on my model's predictions, I am able to statistically reject the null hypothesis, for we see that race
# still does have an effect on one's wages. 

four_g = smf.ols('LNincwage ~ white + black', data = ipums).fit()
print(four_g.summary())

#4.4
#https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
#merge all into one data frame

HS_df = ipums
HS_df = HS_df.loc[HS_df['EDUCDC'] == 12]
m1, b1= np.polyfit(HS_df['EDUCDC'], HS_df['LNincwage'], 1)

noHS_df = ipums
noHS_df = noHS_df.loc[noHS_df['EDUCDC'] < 12]
m2, b2= np.polyfit(noHS_df['EDUCDC'], noHS_df['LNincwage'], 1)

COL_df = ipums
COL_df = COL_df.loc[COL_df['EDUCDC'] == 16]
m3, b3= np.polyfit(COL_df['EDUCDC'], COL_df['LNincwage'], 1)

#sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);

plt.figure(figsize=(12,8))
plt.plot(HS_df['EDUCDC'],HS_df['LNincwage'], 'o', label='HS degree')
plt.plot(noHS_df['EDUCDC'], noHS_df['LNincwage'], 'o', label='No HS degree')
plt.plot(COL_df['EDUCDC'],COL_df['LNincwage'],'o', label = 'College degree')
plt.plot(HS_df['EDUCDC'], m1*HS_df['EDUCDC'] + b1)
plt.plot(noHS_df['EDUCDC'], m2*noHS_df['EDUCDC'] + b2)
plt.plot(COL_df['EDUCDC'], m3*COL_df['EDUCDC'] + b3)
plt.title("Log income wage for different years of education",fontsize=20)
plt.xlabel('Years of education',fontsize=15)
plt.ylabel('log income wage', fontsize=15)
plt.legend()
plt.show()

#5 Determine whether a college degree is a strong predictor of wages
#write down a model that will allow the returns to education to vary by degree acquired
print('Inserting our interaction terms to our model')
result = smf.ols('LNincwage ~ EDUCDC + female + AGE + ageSq + white + black + hispan + married + NCHILD + vet + educXhs + educXcol', data = ipums).fit()
print(result.summary())


#Taking our formula from 3 and adding on the two interaction terms, we still can see that white, married, NCHILD, and 
#vet variables are insignificant. However, we see that there is a statistical significance at the 
#level with the addition of the two interaction terms. I believe this new model explains, despite the statistical insignificance 
#of some variables, a realistic view of the way the world works; each of these variables contribute to one’s income wage. 
#When we incorporate the interaction terms, we see that having a college degree is positive, which the President could interpret
#as being a strong predictor of wages. Also, R2 is higher (slightly), which means our predictors explain the variation in log income wages based on the model.


#6. 
#Estimate the model you proposed in the previous question and report your results.
#(a)  Predict the wages of an 22 year old, female individual (who is neither white, black
#,nor Hispanic, is not married, has no children, and is not a veteran) with a high school diploma and
#an all else equal individual with a college diploma.
#Assume that it takes someone 12 years to graduate high school and 16 years to graduate college.

#calculate from 
six_A_HS = 5.6499 + (.088)*(12) + (-0.4034)*(1) + (0.1731)*(22) + (-0.0018)*(22^2) + (-0.0064)*(12) + (0.0090)*(0)
six_A_COL = 5.6499 + (.088)*(12) + (-0.4034)*(1) + (0.1731)*(22) + (-0.0018)*(22^2) + (-0.0064)*(0) + (0.0090)*(1)
print(six_A_HS)
print(six_A_COL)

#6B
#By my models prediction, my estimate results show that individuals with college degrees have
#nearly 50 percent higher predicted wages than those who have only received a high school diploma.
#The intercept difference between a high school education against a college degree is quite large, giving
#evidence to college educated folk having higher predicted wages.

#6C
#If I were the president, I would study or investigate the longevity of one's involvement in the workforce for those
#pursuing college education. If there remains a positive slope coefficient, then it might be worthwhile to pursue
#increasing student loan subsidies 

#7.	There are many ways that this model could be improved.  How would you do things differently if you were asked to predict 
#the returns to education given the data available on IPUMS?

#Depending on what parameters I would use to predict the returns to education, I would ensure some geographical 
#context is also provided. Using data like proximal distance to city center, communal health indexes, and other 
#more contextual specifiers that are identifiable like mental disabilities or other handicaps.
