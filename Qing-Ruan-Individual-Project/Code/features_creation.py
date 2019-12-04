#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


orders=pd.read_csv('orders.csv')
order_products_prior=pd.read_csv('order_products__prior.csv')
department=pd.read_csv('departments.csv')
aisle=pd.read_csv('aisles.csv')
product=pd.read_csv('products.csv')
order_products_train=pd.read_csv('order_products__train.csv')


# # preprocessing

# ## check order_id for train/prior set and orders set

# In[3]:


# order_id for train
print(order_products_train.order_id.value_counts().shape[0])
print(orders.loc[orders.eval_set=='train'].shape[0])
# order_id for prior
print(order_products_prior.order_id.value_counts().shape[0])
print(orders.loc[orders.eval_set=='prior'].shape[0])


# From above, we can find that the order_id for both train & prior dataset and orders set is correspondont one by one, which is consistent with data description in the kaggle website overview.

# ## missing value

# In[4]:


print(orders.isnull().sum())
orders.index


# There are 206209 missing values in days_since_piror_orders, they means these orders are first order for the user, respectively. According to this meaning, we replace NaN with 0.

# In[5]:


orders=orders.fillna(0)
orders.isnull().sum()


# ## subsetting and merging for datasets

# In[6]:


#merge order_products_prior and orders 
merge_oop=pd.merge(orders, order_products_prior, on='order_id',how='right') #equals to select evalset==0
merge_oop.head()


# In[7]:


#check merge results
print(order_products_prior.loc[order_products_prior['order_id']==2539329])
print(merge_oop.loc[merge_oop['order_id']==2539329])


# As for training, all of the data we used to analyze should come from prior relevant information excluding the last order information for each user, for the next feature selection, we only use 'merge_oop' we created just now instead of orders and order_products_prior. 

# Smillarly, 'order' dataset includes prior orders and last order used for training label. Therefore, we need to exclude information about last order for each user before we using this dataset.

# In[8]:


orders_prior=orders.loc[orders.eval_set=='prior']
orders_prior.head()


# # 2.feature selection

# ## 1.user feature

# ### 1.1 number of orders for each user

# In[9]:


user_f_1=merge_oop.groupby('user_id').order_number.max().reset_index(name='n_orders_users')
user_f_1.head()


# ### 1.2 number of products for each user

# In[10]:


user_f_2=merge_oop.groupby('user_id').product_id.count().reset_index(name='n_products_users')
user_f_2.head()


# ### 1.3 average products of order for each user 

# In[11]:


user_f_2['avg_products_users']=user_f_2.n_products_users/user_f_1.n_orders_users
user_f_2.head()


# ### 1.4 day of one week orderd most for each user

# In[12]:


temp=merge_oop.groupby('user_id')['order_dow'].value_counts().reset_index(name='times_d')
temp.head()


# In[13]:


# find most days ordered by each user
user_f_3=temp.loc[temp.groupby('user_id')['times_d'].idxmax(),['user_id','order_dow']]
user_f_3=user_f_3.rename(columns={'order_dow':'dow_most_user'})   
user_f_3.head()
         


# ### 1.5 time of one day ordered most for each user

# In[14]:


temp=merge_oop.groupby('user_id')['order_hour_of_day'].value_counts().reset_index(name='times_h')
temp.head()


# In[15]:


# find most hours in one day ordered by each user
user_f_4=temp.loc[temp.groupby('user_id')['times_h'].idxmax(),['user_id','order_hour_of_day']]
user_f_4=user_f_4.rename(columns={'order_hour_of_day':'hod_most_user'})   
user_f_4.head()


# ### 1.6 reorder_ratio for each user

# In[16]:


user_f_5=merge_oop.groupby('user_id').reordered.mean().reset_index(name='reorder_ratio_user')
user_f_5.head()


# ### 1.7 shopping frequency for each user

# In[17]:


order_user_group=orders_prior.groupby('user_id')
user_f_6=(order_user_group.days_since_prior_order.sum()/order_user_group.days_since_prior_order.count()).reset_index(name='shopping_freq')
user_f_6.head()


# ### user feature list

# In[18]:


user=pd.merge(user_f_1,user_f_2,on='user_id')
user=user.merge(user_f_3,on='user_id')
user=user.merge(user_f_4,on='user_id')
user=user.merge(user_f_5,on='user_id')
user=user.merge(user_f_6,on='user_id')
user.head()


# ## 2.product feature

# ### 2.1 times of ordered for each product

# In[19]:


prod_f_1=order_products_prior.groupby('product_id').order_id.count().reset_index(name='times_bought_prod')
prod_f_1.head()


# ### 2.2 reordered ratio for each product

# In[20]:


prod_f_2=order_products_prior.groupby('product_id').reordered.mean().reset_index(name='reorder_ratio_prod')
prod_f_2.head()


# ### 2.3 average postions of cart for each product

# In[21]:


prod_f_3=order_products_prior.groupby('product_id').add_to_cart_order.mean().reset_index(name='position_cart_prod')
prod_f_3.head()


# ### 2.4 reordered ratio for each department

# In[22]:


prod_dep=pd.merge(department['department_id'],product[['department_id','product_id']],on='department_id',how='right')
totall_info=pd.merge(prod_dep,prod_f_2,on='product_id',how='right')
totall_info.head()


# In[23]:


group=totall_info.groupby('department_id')
prod_f_4=group.reorder_ratio_prod.mean().reset_index(name='reorder_ratio_dept')

prod_f_4=pd.merge(prod_f_4,totall_info,on='department_id')
del prod_f_4['reorder_ratio_prod']
prod_f_4.drop(['department_id'],axis=1,inplace=True)
prod_f_4.head()


# ### product feature list

# In[24]:


prod=pd.merge(prod_f_1,prod_f_2,on='product_id')
prod=prod.merge(prod_f_3,on='product_id')
prod=prod.merge(prod_f_4,on='product_id')
prod.head()


# ## 3. user & product feature

# ### 3.1 times of one product bought by one user

# In[25]:


user_prd_f_1=merge_oop.groupby(['user_id','product_id']).order_id.count().reset_index(name='times_bought_up')
user_prd_f_1.head()


# ### 3.2 reordered ratio of one product bought by one user

# In[26]:


# number of orders for one user
user_prd_f_2=merge_oop.groupby('user_id').order_number.max().reset_index(name='n_orders_users')
# when the user bought the product for the first time 
temp=merge_oop.groupby(['user_id','product_id']).order_number.min().reset_index(name='first_bought_number')
# merge two datasets
user_prd_f_2=pd.merge(user_prd_f_2,temp,on='user_id')
# how many orders performed after the user bought the product for the first time
user_prd_f_2['order_range']=user_prd_f_2['n_orders_users']-user_prd_f_2['first_bought_number']+1
#reordered ratio 
user_prd_f_2['reorder_ratio_up']=user_prd_f_1.times_bought_up/user_prd_f_2.order_range
user_prd_f_2.head()


# In[27]:


user_prd_f_2=user_prd_f_2.loc[:,['user_id','product_id','reorder_ratio_up']]
user_prd_f_2.head()


# ### 3.3 ratio of one product bought in one user's last four orders

# In[28]:


#Reversing the order number for each product.
merge_oop['order_number_back']=merge_oop.groupby('user_id')['order_number'].transform(max)-merge_oop['order_number']+1
merge_oop.head()


# In[29]:


# select orders where order_number_back <=4
temp1=merge_oop.loc[merge_oop['order_number_back']<=4]
temp1.head()


# In[30]:


# create feature
user_prd_f_3=(temp1.groupby(['user_id','product_id'])['order_number_back'].count()/4).reset_index(name='ratio_last4_orders_up')
user_prd_f_3.head()


# ### user & product feature list

# In[31]:


# merge three features for the user&product list
user_prd=pd.merge(user_prd_f_1,user_prd_f_2,on=['user_id','product_id'])
user_prd=user_prd.merge(user_prd_f_3,on=['user_id','product_id'],how='left')
user_prd.head()


# After checking, we notice that some rows for raio_last4_orders_up have NaN values for our new feature. This happens as there might be products that the customer did not buy on its last four orders. For these cases, we turn NaN values into 0.

# In[32]:


user_prd.ratio_last4_orders_up.fillna(0,inplace=True)


# ## total features list

# In[33]:


total_info=pd.merge(user_prd, user,on='user_id',how='left')
total_info=total_info.merge(prod,on='product_id',how='left')
total_info.head()


# In[34]:


total_info.info()


# In[35]:


total_info.head()


# Totally, we select 14 features to do our models futher. They are'n_orders_users', 'n_products_users', 'avg_products_users',
#        'dow_most_user','hod_most_user','reorder_ratio_user', 'shopping_freq',
#        'product_id', 'times_bought_up', 'reorder_ratio_up',
#        'times_last4_orders_up', 'times_bought_prod', 'reorder_ratio_prod',
#        'position_cart_prod', 'reorder_ratio_dept'.

# In[36]:


total_info.isnull().sum()


# After checking, there is no missing value in this dataset. Then we will use it as observations to build our model.