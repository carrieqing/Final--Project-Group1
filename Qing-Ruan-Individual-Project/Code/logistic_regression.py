# -*- coding: utf-8 -*-
# @Time    : 11/26/19 8:28 AM
# @Author  : Qing
# # 3.creating train and test dataset

# ### set reordered as independent columns

# In[37]:


#select order_id for train by orders.csv
orders_y=orders.loc[((orders.eval_set=='train') | (orders.eval_set=='test')),['user_id','order_id','eval_set']]
orders_y.head()


# In[38]:


# add order_id to the total_info
total_info=pd.merge(total_info,orders_y,on='user_id',how='left')
total_info.head()


# ## create train dataset

# In[39]:


# select reordered in order_products_train as our dependent variable
total_info_train=total_info[total_info.eval_set=='train']
total_info_train=pd.merge(total_info_train,order_products_train[['product_id','order_id','reordered']],on=['order_id','product_id'],how='left')
total_info_train.head()
#total_info=total_info.drop(['reordered_x','reordered_y','add_to_cart_order'],axis=1)


# We add 'reordered' to total_info, and this column in original order_products_train.csv means for the train order(last order for each user), if the product has been bought before.After merged, it means if the product of certain user has been bought before. dataset and the same t. Therefore, only reorder==1 columns have been selected and reorder==Nan means the product has not been selected in the user last order.

# In[40]:


# fill Nan with 0
total_info_train['reordered'].fillna(0,inplace=True) ## inplace decides whether modify original dataset


# In[41]:


# drop order_id
total_info_train.drop(['order_id','eval_set'],axis=1,inplace=True)
total_info_train.head()


# In[42]:


# reset 'user_id' and 'product_id'index
total_info_train=total_info_train.set_index(['user_id','product_id'])
total_info_train.head()


# ### create test dataset

# In[43]:


# select test part
total_info_test=total_info[total_info.eval_set=='test']
total_info_test.head()


# In[44]:


# reset 'user_id' and 'product_id'index
total_info_test=total_info_test.set_index(['user_id','product_id'])
total_info_test.head()


# In[45]:


# only select features
total_info_test.drop(['order_id','eval_set'],axis=1,inplace=True)
total_info_test.head()


# In[46]:


total_info_train.shape,total_info_test.shape


# ## 4.model building

# ## 4.1logistic regression

# In[47]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# In[48]:


X=total_info_train.drop('reordered',axis=1)
y=total_info_train.reordered
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,stratify=y)


# In[49]:


y.value_counts()


# In[50]:


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[51]:


clf = LogisticRegression()
clf.fit(X_train, y_train)


# In[52]:


clf.coef_


# In[53]:


# make predictions# predicton on test
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)


# In[54]:


print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")


# In[66]:


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
