#!/usr/bin/env python
# coding: utf-8

# In[2]:


#PPHA 30545 
#Machine Learning

import rpy2.robjects.packages as rpackages
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
diftrans = rpackages.importr('diftrans')
base = rpackages.importr('base')
stats = rpackages.importr('stats')

from matplotlib import pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np


# ## 3.2 Understanding the Data

# In[3]:


# Beijing_sample = pd.read_csv('Beijing_sample.csv')
# Tianjin_sample = pd.read_csv('Tianjin_sample.csv')

Beijing_sample = base.get("Beijing_sample")
Tianjin_sample = base.get("Tianjin_sample")


# ## 3.3 Clean Data of Beijing and Tianjin Car Sales

# In[56]:


#Exercise 3.1. For each of the following, ensure that the first column is MSRP and the second column is count.
#a. Clean data of Beijing car sales in 2011, and store the data frame in a variable called Beijing_post.
#b. Clean data of Tianjin car sales in 2010 as a variable called Tianjin_pre.
#c. Clean data of Tianjin car sales in 2011 as a variable called Tianjin_post.

#keep 2010 and 2011 data only
Beijing = Beijing_sample[(Beijing_sample['year']>= 2010) & (Beijing_sample['year'] < 2012)]
#collect unique MSRP values
uniqueMSRP = pd.DataFrame(Beijing.MSRP.unique()).rename(columns={0:'MSRP'})
# aggregate sales at each price for 2010 (pre-lottery)
Beijing10_sales = Beijing[(Beijing['year']== 2010)].groupby('MSRP').aggregate({'sales':[sum]})
Beijing10_sales = Beijing10_sales.unstack().reset_index().rename_axis(None, axis=1)
Beijing10_sales = Beijing10_sales.drop(columns=['level_0', 'level_1']).rename(columns={0:'count'})
#merge the MSRP and sales
Beijing_pre = uniqueMSRP.merge(Beijing10_sales, how='left', on = "MSRP")
Beijing_pre[['count']] = Beijing_pre[['count']].fillna(value=0)
Beijing_pre = Beijing_pre.sort_values('MSRP')

Beijing11_sales = Beijing[(Beijing['year']== 2011)].groupby('MSRP').aggregate({'sales':[sum]})
Beijing11_sales = Beijing11_sales.unstack().reset_index().rename_axis(None, axis=1)
Beijing11_sales = Beijing11_sales.drop(columns=['level_0', 'level_1']).rename(columns={0:'count'})
#merge MSRP and sales
Beijing_post = uniqueMSRP.merge(Beijing11_sales, how='left', on = 'MSRP')
Beijing_post[['count']] = Beijing_post[['count']].fillna(value=0)
Beijing_post = Beijing_post.sort_values('MSRP')

Tianjin = Tianjin_sample[(Tianjin_sample['year']>= 2010) & (Tianjin_sample['year'] < 2012)]
#collect unique MSRP values
uniqueMSRP = pd.DataFrame(Tianjin.MSRP.unique()).rename(columns={0:'MSRP'})
# aggregate sales at each price for 2010 (pre-lottery)
Tianjin10_sales = Tianjin[(Tianjin['year']== 2010)].groupby('MSRP').aggregate({'sales':[sum]})
Tianjin10_sales = Tianjin10_sales.unstack().reset_index().rename_axis(None, axis=1)
Tianjin10_sales = Tianjin10_sales.drop(columns=['level_0', 'level_1']).rename(columns={0:'count'})
#merge the MSRP and sales
Tianjin_pre = uniqueMSRP.merge(Tianjin10_sales, how='left', on = "MSRP")
Tianjin_pre[['count']] = Tianjin_pre[['count']].fillna(value=0)
Tianjin_pre = Tianjin_pre.sort_values('MSRP')

Tianjin11_sales = Tianjin[(Tianjin['year']== 2011)].groupby('MSRP').aggregate({'sales':[sum]})
Tianjin11_sales = Tianjin11_sales.unstack().reset_index().rename_axis(None, axis=1)
Tianjin11_sales = Tianjin11_sales.drop(columns=['level_0', 'level_1']).rename(columns={0:'count'})
#merge MSRP and sales
Tianjin_post = uniqueMSRP.merge(Tianjin11_sales, how='left', on = 'MSRP')
Tianjin_post[['count']] = Tianjin_post[['count']].fillna(value=0)
Tianjin_post = Tianjin_post.sort_values('MSRP')


# In[5]:


print(Beijing_pre.head(5))
print(Beijing_post.head(5))
print()
print(Tianjin_pre.head(5))
print(Tianjin_post.head(5))


# ## 3.4 Visualize Beijing Car Sales

# In[6]:


# uncount
df2 = Beijing_pre.pop('count')
Beijing_distribution_pre = pd.DataFrame(Beijing_pre.values.repeat(df2, axis=0), columns=Beijing_pre.columns)
df3 = Beijing_post.pop('count')
Beijing_distribution_post = pd.DataFrame(Beijing_post.values.repeat(df3, axis=0), columns=Beijing_post.columns)


# In[7]:


import seaborn as sns
fig, ax = plt.subplots() 
for a in [Beijing_distribution_pre, Beijing_distribution_post]:
    sns.distplot(a/1000, ax=ax, kde=False)
plt.xlabel("MSRP(1000RMB)", size=14) 
plt.ylabel("Density", size=14) 
plt.title("Pre-lottery (blue) vs. Post-lottery (brown)\n  Sales Distributions of   Beijing Cars", size=18) 
plt.legend(loc='upper right')
ax.set_xlim([0, 1200])


# In[8]:


df4 = Tianjin_pre.pop('count')
Tianjin_distribution_pre = pd.DataFrame(Tianjin_pre.values.repeat(df4, axis=0), columns=Tianjin_pre.columns)
df5 = Tianjin_post.pop('count')
Tianjin_distribution_post = pd.DataFrame(Tianjin_post.values.repeat(df5, axis=0), columns=Tianjin_post.columns)


# In[9]:


fig2, ax2 = plt.subplots() 
for a in [Tianjin_distribution_pre, Tianjin_distribution_post]:
    sns.distplot(a/1000, ax=ax2, kde=False)
plt.xlabel("MSRP(1000RMB)", size=14) 
plt.ylabel("Density", size=14) 
plt.title("Pre-lottery (blue) vs. Post-lottery (brown)\n  Sales Distributions of   Tianjin Cars", size=18) 
plt.legend(loc='upper right')
plt.show()


# ## 3.5 Compute Before-and-After Estimator

# In[10]:


base.set_seed(0) # for reproducibility
n_observations = 100000
placebo_demonstration = pd.DataFrame({'sample1': np.random.normal(0, 1, n_observations), 'sample2': np.random.normal(0, 1, n_observations)})
placebo_demonstration.head()


# In[11]:


fig, ax = plt.subplots()
ax = sns.distplot(placebo_demonstration['sample1'], ax=ax, kde=False, norm_hist=True)
ax = sns.distplot(placebo_demonstration['sample2'], ax=ax, kde=False, norm_hist=True)
plt.xlabel("Support", size=12)
plt.ylabel("Density", size=14)
plt.title("Two Samples from Standard Normal Distribution", size=18)


# 

# In[40]:


# set the seed for reproducibility set.seed(1)
# We will use the `rmultinom` function to construct our placebo.
# Imagine the same number of cars as in 2010. (see `size` argument)
# For each MSRP value, we will decide how many of these imaginary cars will
# be sold at this price. The number of of these imaginary cars to be sold at
# the particular MSRP value will be proportional to the actual number of cars
# sold in the pre-lottery distribution. (see `prob` argument) # We only want 
# one placebo distribution. (see `n` argument) placebo_1 <- data.frame(MSRP = Beijing_pre[‘MSRP’],
base.set_seed(1)
Beijing_pre = uniqueMSRP.merge(Beijing10_sales, how='left', on = "MSRP")
Beijing_pre[['count']] = Beijing_pre[['count']].fillna(value=0)
Beijing_pre = Beijing_pre.sort_values('MSRP')
count =  stats.rmultinom(n = 1, size = sum(Beijing_pre['count']), prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_1 = pd.DataFrame(data=d)
print(placebo_1)
#print(placebo_1.dtypes)



# In[124]:





# In[13]:


Beijing_post = uniqueMSRP.merge(Beijing11_sales, how='left', on = "MSRP")
Beijing_post[['count']] = Beijing_post[['count']].fillna(value=0)
Beijing_post = Beijing_post.sort_values('MSRP')


# In[14]:


base.set_seed(1)
count =  stats.rmultinom(n = 1, size = sum(Beijing_post['count']), prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_2 = pd.DataFrame(data=d)
print(placebo_2)
#print(placebo_2.dtypes)


# In[18]:


placebo_at_0 = diftrans.diftrans(pre_main = placebo_1, post_main = placebo_2, bandwidth_seq = 0)
placebo_at_0


# In[64]:





# In[15]:


cols = ['bandwidth','main']
bandwidth = []
main = []
for i in range(0, 105000, 5000):
    bandwidth.append(diftrans.diftrans(pre_main = placebo_1, post_main = placebo_2, bandwidth_seq = i)['bandwidth'].values)
    main.append(diftrans.diftrans(pre_main = placebo_1, post_main = placebo_2, bandwidth_seq = i)['main'].values)

placebo_transport_df = pd.DataFrame({'bandwidth': bandwidth, 'main':main})    
print(placebo_transport_df)


# In[71]:



cols = ['bandwidth','main']
bandwidth2 = []
main2 = []
for i in range(0,105000, 5000):
    bandwidth2.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, bandwidth_seq = i)['bandwidth'].values)
    main2.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, bandwidth_seq = i)['main'].values)

empirical_transport_df = pd.DataFrame({'bandwidth': bandwidth2, 'main':main2})    
print(placebo_transport_df)


# In[72]:


cols = ['bandwidth','main']
bandwidth2 = []
main2 = []
for i in range(0,105000, 5000):
    bandwidth2.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, bandwidth_seq = i)['bandwidth'].values)
    main2.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, bandwidth_seq = i)['main'].values)

empirical_transport_df = pd.DataFrame({'bandwidth': bandwidth2, 'main':main2})    
print(empirical_transport_df)


# In[38]:


transport_costs = pd.concat([empirical_transport_df.assign(dataset='empirical_transport_df'), placebo_transport_df.assign(dataset = 'placebo_transport_df')])
transport_costs[['bandwidth','main']] = transport_costs[['bandwidth','main']].astype(float)
#print(transport_costs.dtypes)
sns.barplot(x = "bandwidth", y = "main", data = transport_costs, hue='dataset')
plt.xlabel("Bandwidth", size=14) 
plt.ylabel("costs", size=14) 
plt.title("Transport Costs at different ", size=18) 
plt.legend(loc='upper right')
plt.xticks(rotation=45)


# ## 3.6 Compute Differences-in-Transports Estimator

# In[46]:


dit_at_0 = diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, pre_control = Tianjin_pre, post_control = Tianjin_post, bandwidth_seq = 0, conservative = True)
print(dit_at_0)
#3.5
#a

cols = ['bandwidth','main','main2d','control','diff','diff2d']
bandwidth3 = []
main3 = []
main2d = []
control = []
diff = []
diff2d = []

for i in range(0,52500, 2500):
    bandwidth3.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, pre_control = Tianjin_pre, post_control = Tianjin_post, bandwidth_seq = i, conservative = True)['bandwidth'].values)
    main3.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, pre_control = Tianjin_pre, post_control = Tianjin_post, bandwidth_seq = i, conservative = True)['main'].values)
    main2d.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, pre_control = Tianjin_pre, post_control = Tianjin_post, bandwidth_seq = i, conservative = True)['main2d'].values)
    control.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, pre_control = Tianjin_pre, post_control = Tianjin_post, bandwidth_seq = i, conservative = True)['control'].values)
    diff.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, pre_control = Tianjin_pre, post_control = Tianjin_post, bandwidth_seq = i, conservative = True)['diff'].values)
    diff2d.append(diftrans.diftrans(pre_main = Beijing_pre, post_main = Beijing_post, pre_control = Tianjin_pre, post_control = Tianjin_post, bandwidth_seq = i, conservative = True)['diff2d'].values)

placebo_dit_df = pd.DataFrame({'bandwidth': bandwidth3, 'main':main3, 'main2d':main2d, 'control':control, 'diff':diff,'diff2d':diff2d})    
placebo_dit_df = placebo_dit_df.astype(float)

print(placebo_dit_df)


# In[62]:


#3.5b
base.set_seed(0)

count =  stats.rmultinom(n = 1, size = sum(Beijing_pre['count']), prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_Beijing_1 = pd.DataFrame(data=d)
print(placebo_1)
#print(placebo_1.dtypes)


# In[63]:


#3.5 c
count =  stats.rmultinom(n = 1, size = sum(Beijing_post['count']), prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_Beijing_2 = pd.DataFrame(data=d)
print(placebo_Beijing_2)
#print(placebo_Beijing_2.dtypes)


# In[64]:


base.set_seed(0)

Tianjin_pre = uniqueMSRP.merge(Tianjin10_sales, how='left', on = "MSRP")
Tianjin_pre[['count']] = Tianjin_pre[['count']].fillna(value=0)
Tianjin_pre = Tianjin_pre.sort_values('MSRP')
#Tianjin_pre


# In[65]:


#3.5d

count =  stats.rmultinom(n = 1, size = sum(Tianjin_pre['count']), prob = Tianjin_pre['count'])
count2 = count[:,0]
d = {'MSRP': Tianjin_pre['MSRP'], 'count' : count2}
placebo_Tianjin_1 = pd.DataFrame(data=d)
print(placebo_Tianjin_1)
#print(placebo_Tianjin_1.dtypes)


# In[66]:


#3.5e
count =  stats.rmultinom(n = 1, size = sum(Tianjin_post['count']), prob = Tianjin_pre['count'])
count2 = count[:,0]
d = {'MSRP': Tianjin_pre['MSRP'], 'count' : count2}
placebo_Tianjin_2 = pd.DataFrame(data=d)
print(placebo_Tianjin_2)
#print(placebo_Tianjin_1.dtypes)


# In[68]:


#3.5f
#
cols = ['bandwidth','main']
bandwidth = []
main = []

for i in range(0,50000,2500):
    bandwidth.append(diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = i, conservative = True)['bandwidth'].values)

    
dit_at_16666 = diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = 16666, conservative = True)
print(dit_at_16666)
dit_at_33333 = diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = 33333, conservative = True)
print(dit_at_33333)
dit_at_50000 = diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = 50000, conservative = True)
print(dit_at_50000)


cols = ['bandwidth','main','main2d','control','diff','diff2d']
bandwidth3 = []
main3 = []
main2d = []
control = []
diff = []
diff2d = []

for i in range(0,52500, 2500):
    bandwidth3.append(diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = i, conservative = True)['bandwidth'].values)
    main3.append(diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = i, conservative = True)['main'].values)
    main2d.append(diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = i, conservative = True)['main2d'].values)
    control.append(diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = i, conservative = True)['control'].values)
    diff.append(diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = i, conservative = True)['diff'].values)
    diff2d.append(diftrans.diftrans(pre_main = placebo_Beijing_1, post_main = placebo_Beijing_2, pre_control = placebo_Tianjin_1, post_control = placebo_Tianjin_2, bandwidth_seq = i, conservative = True)['diff2d'].values)

three_five_g = pd.DataFrame({'bandwidth': bandwidth3, 'main':main3, 'main2d':main2d, 'control':control, 'diff':diff,'diff2d':diff2d})    
three_five_g = three_five_g.astype(float)


# In[70]:


#3.5g

three_five_g['Dif-In-Transports'] = np.abs(three_five_g['diff2d'])
#print(three_five_g)
ax = sns.barplot(x = 'bandwidth', y = 'Dif-In-Transports', data = three_five_g)
plt.xlabel('Bandwidth')
plt.ylabel('Difference in Transports Estimator')
plt.title('Difference-in-Transport Estimates in Absolute Value Form')
plt.xticks(rotation=45)


# In[76]:


#three_five_h
three_five_h = three_five_g[['bandwidth','main','Dif-In-Transports']]
print(three_five_h)


# In[ ]:




