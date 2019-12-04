#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%-----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
os.chdir("/Users/yuanyuan/Desktop/new data sets/dataset")


#%%-----------------------------------------------------------------------
# open dataset and view structure of each data set
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
train = pd.read_csv('order_products__train.csv')
prior = pd.read_csv('order_products__prior.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
sample = pd.read_csv('sample_submission.csv')

#pd.set_option('display.max_columns',10)
#all_sets = [aisles,departments,train,prior,orders,products,sample]
#for i in all_sets:
#    print(i.head())



#%%-----------------------------------------------------------------------
# Fisrt part: orders.csv
# 1.1 basic information about Orders dataset
    
print(orders.shape)
print(orders.columns)
print(orders.dtypes)
orders.count()


# 'days_since_prior_order': missing value
print(orders['days_since_prior_order'].dtype)
orders['days_since_prior_order'].replace(np.nan,-1)

#%%-----------------------------------------------------------------------

# 1.2 Three sets
color = sns.color_palette()
dist_eval_set=orders.eval_set.value_counts()
sns.barplot(dist_eval_set.index,dist_eval_set.values, alpha=0.8,color=color[1])
plt.ylabel('orders')
plt.xlabel('eval_set type')
plt.title('Number of orders in each set')
plt.show()

dist_eval_set=orders.eval_set.value_counts()
print(dist_eval_set)
# There are  985475 orders for the prior set, and the dataset extract the last order of each custormer as train and test dataset, respectively.
# The train set has 40096 observations and the test dataset has 23004 observations.

# test prior.csv and train.csv 
print((prior.order_id.value_counts().shape[0]) == (orders.loc[orders.eval_set=='prior'].shape[0]))
print((train.order_id.value_counts().shape[0]) == (orders.loc[orders.eval_set=='train'].shape[0]))

# the number of orders in prior.csv and train.csv are the same as those in orders.csv


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
plt.title('Number of users in each set')
plt.show()


group_ev=orders.groupby('eval_set')

for name,group in group_ev:
    print(name)
    print(group.user_id.unique().shape[0])

# There are 63100 users in prior set, 23004 users in test set and 40096 users in train set.

#%%-----------------------------------------------------------------------
# 1.3 Order information
# 1.3.1 Frequency of orders 
dist_no_orders=orders.groupby('user_id').order_number.max()
dist_no_orders=dist_no_orders.value_counts()

plt.figure(figsize=(20,8))
sns.barplot(dist_no_orders.index,dist_no_orders.values)
plt.xlabel('orders')
plt.ylabel('users')
plt.title('Frequency of orders by users')
plt.show()

#boxplot test outliners
sns.boxplot(dist_no_orders.index)
plt.show()
# So there are no orders less than 4 and is max capped at 100 as given in the data page.

#%%-----------------------------------------------------------------------
# 1.3.2 Times of orders
# By day of week
dist_d_orders=orders.order_dow.value_counts()

sns.barplot(dist_d_orders.index,dist_d_orders.values, palette=sns.color_palette('Blues_d',7))
plt.xlabel('day of week')
plt.ylabel('orders')
plt.title('Frequency of orders by day of week')
plt.show()
# It looks as though 0 represents Saturday and 1 represents Sunday. Thursday is then the least popular day to make orders.


#  By hour of day
dist_h_orders=orders.order_hour_of_day.value_counts()
plt.figure(figsize=(10,8))
sns.barplot(dist_h_orders.index,dist_h_orders.values, palette=sns.color_palette('Greens_d',24))
plt.xlabel('hour of day')
plt.ylabel('orders')
plt.title('Frequency of orders by hour of day')
plt.show()
# So majority of the orders are made during day time. The 10am hour is the most popular time to make orders, followed by a dip around lunch time and a pickup in the afternoon. 
# Now let us combine the day of week and hour of day to see the distribution.
 

# By hour in a day
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


#%%-----------------------------------------------------------------------
# 1.3.3 Time interval
dist_d_prior_orders=orders.days_since_prior_order.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(dist_d_prior_orders.index,dist_d_prior_orders.values, palette=sns.color_palette('Greens_d',31))
plt.xlabel('days of prior order')
plt.ylabel('count')
plt.title('Time interval between orders')
plt.show()

# While the most popular relative time between orders is monthly (30 days), there are "local maxima" at weekly (7 days), biweekly (14 days), triweekly (21 days), and quadriweekly (28 days).
# Looks like customers order once in every week (check the peak at 7 days) or once in a month (peak at 30 days). We could also see smaller peaks at 14, 21 and 28 days (weekly intervals).
#%%-----------------------------------------------------------------------









#%%-----------------------------------------------------------------------
# Second part: product.csv (merge products.csv, aisles.csv and departments.csv)
product = products.merge(aisles).merge(departments)
print(product.shape)
print(product.columns)
print(product.dtypes)
pd.options.display.max_columns = None
print(product.head())

#%%-----------------------------------------------------------------------
# 2.1 products in departments and aisles
# 2.1.1 products in departments
grouped = product.groupby("department")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped  = grouped.groupby(['department']).sum()['Total_products'].sort_values(ascending=False)
print(grouped)


plt.xticks(rotation='vertical')
sns.barplot(grouped.index, grouped.values)
plt.ylabel('Number of products', fontsize=13)
plt.xlabel('Departments', fontsize=13)
plt.title("The number of products in each department")
plt.show()

# The most five important departments are personal care, snacks, pantry, beverages and frozen. The number of items from these departments were more than 4,000 times.



# 2.1.2 products in asiles among all departments
grouped2 = product.groupby("aisle")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped2 = grouped2.sort_values(by='Total_products', ascending=False)[:20]
print(grouped2)

grouped2  = grouped2.groupby(['aisle']).sum()['Total_products'].sort_values(ascending=False)
grouped2 = grouped2.drop(labels="missing",axis=0)

plt.xticks(rotation='vertical')
sns.barplot(grouped2.index, grouped2.values)
plt.ylabel('Number of products', fontsize=13)
plt.xlabel('Aisles', fontsize=13)
plt.title("The number of products in each aisle(top 20)")
plt.show()
# The most three important aisles are canding chocolate, ice cream and vitamins sumpplements.



# 2.1.3 products in asiles among each department

#grouped3 = product.groupby(['department','aisle'])
# grouped3 = grouped3['product_id'].aggregate({'Total_products': 'count'}).reset_index()
# print(grouped3)
#
# fig, axes = plt.subplots(7,3, figsize=(20,45),gridspec_kw =  dict(hspace=1.4))
# for (aisle, group), ax in zip(grouped3.groupby(["department"]), axes.flatten()):
#     g = sns.barplot(group.aisle, group.Total_products , ax=ax)
#     ax.set(xlabel = "Aisles", ylabel=" Number of products")
#     g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)
#     ax.set_title(aisle, fontsize=15)

# Each graph shows the number of products in each aisle of different departments.  
    
    
    
    
    
    
#%%-----------------------------------------------------------------------
# 2.3 Oreder of product(merge products+all_order(prior+train)+orders) 
all_order = pd.concat([train, prior], axis=0)
order_flow = orders[['user_id', 'order_id']].merge(all_order[['order_id', 'product_id']]).merge(product)
order_flow.head()

print(order_flow.shape)
print(order_flow.columns)
print(order_flow.dtypes)
order_flow.count()

#%%-----------------------------------------------------------------------
# 2.3.1 Sales in each department(find best selling apartment)
grouped4 = order_flow.groupby("department")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
grouped4.sort_values(by='Total_orders', ascending=False, inplace=True)
print(grouped4)

grouped4  = grouped4.groupby(['department']).sum()['Total_orders'].sort_values(ascending=False)
plt.xticks(rotation='vertical')
sns.barplot(grouped4.index, grouped4.values)
plt.ylabel('Number of Orders', fontsize=13)
plt.xlabel('Departments', fontsize=13)
plt.title('Sales in each department')
plt.show()
# the most three popular departments are produce, dairy eggs and snacks.
#%%-----------------------------------------------------------------------
# 2.3.2 Sales in each aisle(best selling aisle)
grouped5 = order_flow.groupby("aisle")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
grouped5.sort_values(by='Total_orders', ascending=False, inplace=True )
print(grouped5.head(15))

grouped5 = grouped5.groupby(['aisle']).sum()['Total_orders'].sort_values(ascending=False)[:15]
plt.xticks(rotation='vertical')
sns.barplot(grouped5.index, grouped5.values)
plt.ylabel('Number of Orders', fontsize=13)
plt.xlabel('Aisles', fontsize=13)
plt.title('Sales in each aisle')
plt.show()

# the top three best selling aisles are fresh fruits, fresh vegetables and packaged vegetables fruits.
#%%-----------------------------------------------------------------------







#%%----------------------------------------------------------------------- 
# 2.4 Reorder of product
# 2.4.1 the number of reordered products
reord=sum(prior['reordered']==1)
not_reord=sum(prior['reordered']==0)
order_sum = reord + not_reord
reord_pro=reord/order_sum
not_ord_pro=not_reord/order_sum

all_order = pd.concat([train, prior], axis=0)
grouped6 = all_order.groupby("reordered")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped6 = grouped6.groupby(['reordered']).sum()['Total_products'].sort_values(ascending=False)
print(grouped6)
sns.barplot(grouped6.index,grouped6.values)
plt.ylabel('Number of Products', fontsize=13)
plt.xlabel('Reordered or Not Reordered', fontsize=13)
plt.ticklabel_format(style='plain', axis='y')
plt.title("Not reorder vs Reorder")
plt.show()

# conclusion:
# 19126536 products are previously ordered by customers, reordered products take 0.59 % of ordered products.
# 13307953 products are not ordered by customers before, non-reordered products take 0.41 % of ordered products.
#%%-----------------------------------------------------------------------
# 2.4.2 highest reordered rate 
grouped7 = all_order.groupby("product_id")["reordered"].aggregate({'reorder_sum': sum,'total': 'count'}).reset_index()
grouped7['reord_ratio']= grouped7['reorder_sum'] / grouped7['total']
grouped7 = pd.merge(grouped7, product, how='left', on=['product_id'])
grouped8 = grouped7.sort_values(['reord_ratio'],ascending=False).head(10)
print(grouped8)

sns.barplot(grouped8['product_name'],grouped8['reord_ratio'])
plt.ylim([0.85,0.95])
plt.xticks(rotation='vertical')
plt.title('Top 10 reordered rate')
plt.show()

# conclusion: 
# 1.The three products with the highest reordered rate are Raw Veggie Wrappers, Serenity Ultimate Extrema Overnight Pads and Orange Energy Shots.
#%%-----------------------------------------------------------------------
# 2.4.3 department with highest reorder ratio 
grouped9 = grouped7.sort_values(['reord_ratio'],ascending=False)
sns.lineplot(grouped7['department'],grouped7['reord_ratio'])
plt.xticks(rotation='vertical')
plt.title('Reordered ratio in each department')
plt.show()

# A: Personal care has lowest reorder ratio and dairy eggs have highest reorder ratio.
#%%-----------------------------------------------------------------------
# 2.4.4 Relationship between add_to_cart and reordered?

# add_to_cart_order: The sequence of product is added to the cart in each order

prior["add_to_cart_order_mod"] =prior["add_to_cart_order"].copy()
prior["add_to_cart_order_mod"].loc[prior["add_to_cart_order_mod"]>70] = 70
grouped_df = prior.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['add_to_cart_order_mod'].values, grouped_df['reordered'].values, alpha=0.8)
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Add to cart order', fontsize=12)
plt.title("Add to cart order - Reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# t-test
data1 = prior[prior['reordered']==0]['add_to_cart_order']
data2 = prior[prior['reordered']==1]['add_to_cart_order']
print(np.mean(data1))
print(np.mean(data2))
print(stats.ttest_ind(data1,data2))

# conclusion: 
# 1.Orders placed initially in the cart are more likely to be reorderd than one placed later in the cart.
# 2.We did t-test to verify whether the sequence of adding to cart are siginificantly different between reordered products and not reordered products.
# We can conclude from the results showing the p-value is smaller than 0.05 that the sequence of adding to cart significantly influence whether the products being reordered.
#%%-----------------------------------------------------------------------








# # feature engineering

# In[25]:


# prior_orders dataset
prior_orders = orders.merge(order_products_prior, on='order_id', how='inner')
prior_orders.head()


# # user features

# In[26]:


user_f_1=prior_orders.groupby('user_id').order_number.max().reset_index(name='n_orders_users')
user_f_2=prior_orders.groupby('user_id').product_id.count().reset_index(name='n_products_users')
user_f_2['avg_products_users']=user_f_2.n_products_users/user_f_1.n_orders_users
temp=prior_orders.groupby('user_id')['order_dow'].value_counts().reset_index(name='times_d')
user_f_3=temp.loc[temp.groupby('user_id')['times_d'].idxmax(),['user_id','order_dow']]
user_f_3=user_f_3.rename(columns={'order_dow':'dow_most_user'})   
temp=prior_orders.groupby('user_id')['order_hour_of_day'].value_counts().reset_index(name='times_h')
user_f_4=temp.loc[temp.groupby('user_id')['times_h'].idxmax(),['user_id','order_hour_of_day']]
user_f_4=user_f_4.rename(columns={'order_hour_of_day':'hod_most_user'})   
user_f_5=prior_orders.groupby('user_id').reordered.mean().reset_index(name='reorder_ratio_user')
order_user_group=prior_orders.groupby('user_id')
user_f_6=(order_user_group.days_since_prior_order.sum()/order_user_group.days_since_prior_order.count()).reset_index(name='shopping_freq')


# In[27]:


user=pd.merge(user_f_1,user_f_2,on='user_id')
user=user.merge(user_f_3,on='user_id')
user=user.merge(user_f_4,on='user_id')
user=user.merge(user_f_5,on='user_id')
user=user.merge(user_f_6,on='user_id')
user.head()


# In[28]:


user.shape


# # product features

# In[29]:


prod_f_1=order_products_prior.groupby('product_id').order_id.count().reset_index(name='times_bought_prod')
prod_f_2=order_products_prior.groupby('product_id').reordered.mean().reset_index(name='reorder_ratio_prod')
prod_f_3=order_products_prior.groupby('product_id').add_to_cart_order.mean().reset_index(name='position_cart_prod')
prod_dep=pd.merge(departments['department_id'],products[['department_id','product_id']],on='department_id',how='right')
totall_info=pd.merge(prod_dep,prod_f_2,on='product_id',how='right')

group=totall_info.groupby('department_id')
prod_f_4=group.reorder_ratio_prod.mean().reset_index(name='reorder_ratio_dept')

prod_f_4=pd.merge(prod_f_4,totall_info,on='department_id')
del prod_f_4['reorder_ratio_prod']


# In[67]:


prod_f_2.head()


# In[30]:


prod=pd.merge(prod_f_1,prod_f_2,on='product_id')
prod=prod.merge(prod_f_3,on='product_id')
prod=prod.merge(prod_f_4,on='product_id')
prod.head()


# In[31]:


prod.shape


# # user * product features

# In[32]:


user_prd_f_1=prior_orders.groupby(['user_id','product_id']).order_id.count().reset_index(name='times_bought_up')

# number of orders for one user
user_prd_f_2=prior_orders.groupby('user_id').order_number.max().reset_index(name='n_orders_users')
# when the user bought the product for the first time 
temp=prior_orders.groupby(['user_id','product_id']).order_number.min().reset_index(name='first_bought_number')
# merge two datasets
user_prd_f_2=pd.merge(user_prd_f_2,temp,on='user_id')
# how many orders performed after the user bought the product for the first time
user_prd_f_2['order_range']=user_prd_f_2['n_orders_users']-user_prd_f_2['first_bought_number']+1
#reordered ratio 
user_prd_f_2['reorder_ratio_up']=user_prd_f_1.times_bought_up/user_prd_f_2.order_range
user_prd_f_2=user_prd_f_2.loc[:,['user_id','product_id','reorder_ratio_up']]

#Reversing the order number for each product.
prior_orders['order_number_back']=prior_orders.groupby('user_id')['order_number'].transform(max)-prior_orders['order_number']+1

temp1=prior_orders.loc[prior_orders['order_number_back']<=4]
user_prd_f_3=(temp1.groupby(['user_id','product_id'])['order_number_back'].count()/4).reset_index(name='ratio_last4_orders_up')


# In[33]:


user_prd=pd.merge(user_prd_f_1,user_prd_f_2,on=['user_id','product_id'])
user_prd=user_prd.merge(user_prd_f_3,on=['user_id','product_id'],how='left')
user_prd.head()


# In[34]:


user_prd.ratio_last4_orders_up.fillna(0,inplace=True)


# In[35]:


user_prd.shape


# # merge all features

# In[36]:


data = user_prd.merge(user, on='user_id', how='left')
data.head()


# In[37]:


data = data.merge(prod, on='product_id', how='left')
data.head()


# In[38]:


data.shape


# # add features into train dataset

# In[39]:


orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head(10)


# In[40]:


data = data.merge(orders_future, on='user_id', how='left')
data.head(10)


# In[41]:


data = data[data.eval_set=='train']
data.head()


# In[42]:


data = data.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data['reordered'] = data['reordered'].fillna(0)
data.head(15)


# In[43]:


# data_train = data_train.drop(['eval_set_x', 'order_id_x', 'eval_set_y', 'order_id_y'], axis=1)
# data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data = data.drop(['department_id', 'eval_set', 'order_id'], axis=1)


# In[44]:


data.head(20)


# In[45]:


data.shape


# In[46]:


data = data.set_index(['user_id', 'product_id'])
data.head(15)


# # get features and target

# In[47]:


X, y = data.drop('reordered', axis=1), data.reordered


# In[48]:


X.head()


# In[49]:


y.value_counts()


# In[50]:


from imblearn.over_sampling import RandomOverSampler

# RandomOverSampler (with random_state=0)
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

reordered = 'reordered'
pd.DataFrame(data=y, columns=[reordered])[reordered].value_counts()


# # Logistic Model

# In[78]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[79]:


print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)


# In[80]:


conf_matrix = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(conf_matrix, index=['0','1'], columns=['0','1'] )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


# # Decision Tree

# In[51]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[131]:


clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[132]:


y_pred_gini = clf_gini.predict(X_test)


# In[133]:


print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)


# In[134]:


conf_matrix = confusion_matrix(y_test, y_pred_gini)
df_cm = pd.DataFrame(conf_matrix)

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


# # Random Forest

# In[53]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)


# In[59]:


# plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.drop('reordered', axis=1).columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()


# In[60]:


# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[137]:


from sklearn.metrics import roc_auc_score
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)


# In[138]:


conf_matrix = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(conf_matrix)

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()


# # Naive Bayes

# In[140]:


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[141]:


print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)


# In[142]:


conf_matrix = confusion_matrix(y_test, y_pred)
#class_names = total_info_train['reordered'].unique()

df_cm = pd.DataFrame(conf_matrix, index=['0','1'], columns=['0','1'] )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()


# # KNN

# In[72]:


# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)


# In[73]:


y_pred = clf.predict(X_test)


# In[74]:


print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)


# In[75]:


conf_matrix = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(conf_matrix)

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()



