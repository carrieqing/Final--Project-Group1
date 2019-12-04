#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.getcwd()
os.chdir('/Users/qingruan/Desktop/DM_project')


# In[4]:


aisles = pd.read_csv('instacart-market-basket-analysis/aisles.csv')
departments = pd.read_csv('instacart-market-basket-analysis/departments.csv')
train = pd.read_csv('instacart-market-basket-analysis/order_products__train.csv')
prior = pd.read_csv('instacart-market-basket-analysis/order_products__prior.csv')
orders = pd.read_csv('instacart-market-basket-analysis/orders.csv')
products = pd.read_csv('instacart-market-basket-analysis/products.csv')
sample = pd.read_csv('instacart-market-basket-analysis/sample_submission.csv')


# In[10]:


print(orders.shape)
print(orders.columns)
print(orders.dtypes)
orders.count()


# # Exploratory Data Analysis

# ## import dataset and take a look 

# In[10]:


orders=pd.read_csv('instacart-market-basket-analysis/orders.csv')


# In[11]:


orders.head()


# As we could see, orders.csv has columns about order_id, user_id, eval_set,order_number	order_dow	order_hour_of_day	days_since_prior_order

# ## check order.csv

# ### number of orders and users by each eval_set

# In[12]:


# plt.figure(figsize=(12,8))
color = sns.color_palette()
dist_eval_set=orders.eval_set.value_counts()
sns.barplot(dist_eval_set.index,dist_eval_set.values, alpha=0.8,color=color[1])
plt.ylabel('orders')
plt.xlabel('eval_set type')
plt.title('Number of orders in each set')

plt.show()


# In[13]:


dist_eval_set


# There are 3214,784 orders for the prior set, and the dataset extract the last order of each custormer as train and test dataset, respectively.
# The train set has 131,209 observations and the test dataset has 75,000 observations.

# In[14]:


color = sns.color_palette()
group_ev=orders.groupby('eval_set')
x=[]
y=[]
for name,group in group_ev:
    x.append(name)
    y.append(group.user_id.unique().shape[0])
sns.barplot(x,y, alpha=0.8,color=color[2])
plt.ylabel('users')
plt.xlabel('eval_set type')
plt.title('Number of orders in each set')

plt.show()


# In[15]:


group_ev=orders.groupby('eval_set')

for name,group in group_ev:
    print(name)
    print(group.user_id.unique().shape[0])


# There are 206,209 customers in total. Out of which, the last purchase of 131,209 customers are given as train set and we need to predict for the rest 75,000 customers.

# In[16]:


dist_no_orders=orders.groupby('user_id').order_number.max()
dist_no_orders=dist_no_orders.value_counts()

plt.figure(figsize=(20,8))
sns.barplot(dist_no_orders.index,dist_no_orders.values)
plt.xlabel('orders')
plt.ylabel('users')
plt.title('Frequency of orders by users')
plt.show()


# In[ ]:





# So there are no orders less than 4 and is max capped at 100 as given in the data page.

# # when are orders made?

# In[17]:


dist_d_orders=orders.order_dow.value_counts()

sns.barplot(dist_d_orders.index,dist_d_orders.values, palette=sns.color_palette('Blues_d',7))
plt.xlabel('day of week')
plt.ylabel('orders')
plt.title('Frequency of orders by day of week')

plt.show()


# It looks as though 0 represents Saturday and 1 represents Sunday. Wednesday is then the least popular day to make orders.

# In[19]:


dist_h_orders=orders.order_hour_of_day.value_counts()
plt.figure(figsize=(10,8))
sns.barplot(dist_h_orders.index,dist_h_orders.values, palette=sns.color_palette('Greens_d',24))
plt.xlabel('hour of day')
plt.ylabel('orders')
plt.title('Frequency of orders by hour of day')

plt.show()


# So majority of the orders are made during day time. The 10am hour is the most popular time to make orders, followed by a dip around lunch time and a pickup in the afternoon.Now let us combine the day of week and hour of day to see the distribution.

# In[18]:


grouped=orders.groupby(['order_dow','order_hour_of_day']).order_number.count().reset_index() 
# using reset_index to set order_dow and ordr_h_day as columns, or they would be index
time_orders=grouped.pivot('order_dow', 'order_hour_of_day','order_number')

plt.figure(figsize=(10,5))
sns.heatmap(time_orders, cmap='YlOrRd')
plt.ylabel('Day of Week')
plt.xlabel('Hour of Day')
plt.title('Number of Orders Day of Week vs Hour of Day')
plt.show()


# Saturday afternoon and Sunday morning are the most popular time to make orders.

# # time interval between the orders

# In[20]:


dist_d_prior_orders=orders.days_since_prior_order.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(dist_d_prior_orders.index,dist_d_prior_orders.values, palette=sns.color_palette('Greens_d',31))
plt.xlabel('days of prior order')
plt.ylabel('count')
plt.title('Time interval between orders')

plt.show()


# While the most popular relative time between orders is monthly (30 days), there are "local maxima" at weekly (7 days), biweekly (14 days), triweekly (21 days), and quadriweekly (28 days).
# Looks like customers order once in every week (check the peak at 7 days) or once in a month (peak at 30 days). We could also see smaller peaks at 14, 21 and 28 days (weekly intervals).

# In[ ]:




