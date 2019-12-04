#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: zixuan

"""
#%%-----------------------------------------------------------------------
### EDA
#%%-----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
os.chdir("/Users/yuanyuan/Desktop/DATS_6103/project/DM_data sets")


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









#%%-----------------------------------------------------------------------
### Naive Bayes
#%%-----------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt




from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)


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

#%%-----------------------------------------------------------------------










#%%-----------------------------------------------------------------------
### GUI
#%%-----------------------------------------------------------------------
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import tkinter.font as tkFont
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        # self.geometry('600x400')
        self.createWidgets()
        self.color = sns.color_palette()

    def createWidgets(self):
        self.title("Presentation")
        tabControl = ttk.Notebook(self)
        self.tab1 = tk.Frame(tabControl)
        self.createTab1()
        tabControl.add(self.tab1, text = 'EDA')
        self.tab2 = tk.Frame(tabControl)
        self.createTab2()
        tabControl.add(self.tab2, text = 'Features')
        self.tab3 = tk.Frame(tabControl)
        self.createTab3()
        tabControl.add(self.tab3, text = 'Models')
        tabControl.pack(expand=1, fill="both")

    def addmenu(self, Menu):
        Menu(self)

    def createTab1(self):
        topframe = tk.Frame(self.tab1, height=80)
        #contentframe = tk.Frame(tab1)
        topframe.pack(side = tk.TOP)
        #contentframe.pack(side = tk.TOP)

        OrdersButton = tk.Menubutton(topframe, text = 'Orders')
        #OrdersButton.pack()
        ordersMenu = tk.Menu(OrdersButton, tearoff = False)
        ordersMenu.add_command(label = 'EDA1', command = self.oeda1)
        ordersMenu.add_command(label = 'EDA2', command = self.oeda2)
        ordersMenu.add_command(label = 'EDA3', command = self.oeda3)
        ordersMenu.add_command(label = 'EDA4', command = self.oeda4)
        OrdersButton.config(menu=ordersMenu)

        ProductsButton = tk.Menubutton(topframe, text = 'Products')
        #ProductsButton.pack()
        productsMenu = tk.Menu(ProductsButton, tearoff = False)
        productsMenu.add_command(label = 'EDA1', command = self.peda1)
        productsMenu.add_command(label = 'EDA2', command = self.peda2)
        productsMenu.add_command(label = 'EDA3', command = self.peda3)
        productsMenu.add_command(label = 'EDA4', command = self.peda4)
        ProductsButton.config(menu=productsMenu)

        OrdersButton.grid(row=0, column=0, sticky=tk.W)
        ProductsButton.grid(row=0, column=1)

        fig1 = Figure(figsize=(5, 4), dpi=100)
        self.ax1 = fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.canvas1._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        fig2 = Figure(figsize=(5, 4), dpi=100)
        self.ax2 = fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.tab1)
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)


    def createTab2(self):
        self.ft1 = tkFont.Font(size = 24)
        tk.Label(self.tab2, text='Select Feature:').pack()
        self.features = ttk.Combobox(self.tab2, values=list(data.columns))
        self.features.pack()
        tk.Button(self.tab2, text='Show first fifteen values', command=self.show_feature).pack()
        self.featureText = tk.Text(self.tab2, font = self.ft1)
        self.featureText.pack()

    def show_feature(self):
        identifier = self.features.get()  # get option
        self.featureText.delete(1.0, tk.END)  # empty widget to print new text
        self.featureText.insert(tk.END, str(data[identifier][0:15]))

    def createTab3(self):
        self.ft2 = tkFont.Font(size = 18)
        tk.Label(self.tab3, text='Select Model:').pack()
        self.models = ttk.Combobox(self.tab3, values=['Logistic Regression', 'k-NearestNeighbor', 'Random Forest'])
        self.models.pack()
        tk.Button(self.tab3, text='Show Result', command=self.show_result).pack()
        fig3 = Figure(figsize=(5, 4), dpi=100)
        self.ax1 = fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(fig3, master=self.tab3)
        self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas3._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.modelText = tk.Text(self.tab3)
        self.modelText.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        # self.modelText.pack(side = tk.LEFT)

        # self.text.pack(side = tk.LEFT)

    def show_result(self):
        modelName = self.models.get()
        if modelName == 'Logistic Regression':

            self.modelText.delete(1.0, tk.END)
            self.modelText.insert(tk.END, "Classification Report:\n" + str(classification_report(y_test, LRy_pred)) + "\n" + "Accuracy : " + str((accuracy_score(y_test, LRy_pred) * 100)) + "\n" + "ROC_AUC : " + str((roc_auc_score(y_test, LRy_pred_score[:, 1]) * 100)))

            self.canvas3.get_tk_widget().destroy()
            self.canvas3._tkcanvas.destroy()
            fig = self.draw_cm(LRy_pred)
            self.canvas3 = FigureCanvasTkAgg(fig, master=self.tab3)
            self.canvas3.draw()
            self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas3._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        elif modelName == 'k-NearestNeighbor':

            self.modelText.delete(1.0, tk.END)
            self.modelText.insert(tk.END, "Classification Report:\n" + str(
                classification_report(y_test, KNNy_pred)) + "\n" + "Accuracy : " + str(
                (accuracy_score(y_test, KNNy_pred) * 100)) + "\n" + "ROC_AUC : " + str(
                (roc_auc_score(y_test, KNNy_pred_score[:, 1]) * 100)))

            self.canvas3.get_tk_widget().destroy()
            self.canvas3._tkcanvas.destroy()
            fig = self.draw_cm(KNNy_pred)
            self.canvas3 = FigureCanvasTkAgg(fig, master=self.tab3)
            self.canvas3.draw()
            self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas3._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        else:

            self.modelText.delete(1.0, tk.END)
            self.modelText.insert(tk.END, "Classification Report:\n" + str(
                classification_report(y_test, RFy_pred)) + "\n" + "Accuracy : " + str(
                (accuracy_score(y_test, RFy_pred) * 100)) + "\n" + "ROC_AUC : " + str(
                (roc_auc_score(y_test, RFy_pred_score[:, 1]) * 100)))

            self.canvas3.get_tk_widget().destroy()
            self.canvas3._tkcanvas.destroy()
            fig = self.draw_cm(RFy_pred)
            self.canvas3 = FigureCanvasTkAgg(fig, master=self.tab3)
            self.canvas3.draw()
            self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas3._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def draw_cm(self, y_pred):
        conf_matrix = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(conf_matrix, index=['0', '1'], columns=['0', '1'])

        f, ax = plt.subplots(figsize=(5, 4))
        hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                         yticklabels=df_cm.columns, xticklabels=df_cm.columns)
        hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        return f

    def create_oeda1(self):
        f, ax = plt.subplots(figsize = (5, 4))
        dist_eval_set = orders.eval_set.value_counts()
        sns.barplot(dist_eval_set.index, dist_eval_set.values, alpha=0.8, color=self.color[1])
        plt.ylabel('orders')
        plt.xlabel('eval_set type')
        plt.title('Number of orders in each set')
        return f

    def create_oeda2(self):
        f, ax = plt.subplots(figsize=(5, 4))
        group_ev = orders.groupby('eval_set')
        x = []
        y = []
        for name, group in group_ev:
            x.append(name)
            y.append(group.user_id.unique().shape[0])
        sns.barplot(x, y, alpha=0.8, color=self.color[2])
        plt.ylabel('users')
        plt.xlabel('eval_set type')
        plt.title('Number of users in each set')
        return f

    def create_oeda3(self):
        f, ax = plt.subplots(figsize=(5, 4))
        self.dist_no_orders = orders.groupby('user_id').order_number.max()
        self.dist_no_orders = self.dist_no_orders.value_counts()

        sns.barplot(self.dist_no_orders.index, self.dist_no_orders.values)
        plt.xlabel('orders')
        plt.ylabel('users')
        plt.title('Frequency of orders by users')
        return f

    def create_oeda4(self):
        f, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(self.dist_no_orders.index)
        return f

    def create_oeda5(self):
        f, ax = plt.subplots(figsize=(5, 4))
        self.dist_d_orders = orders.order_dow.value_counts()

        sns.barplot(self.dist_d_orders.index, self.dist_d_orders.values, palette=sns.color_palette('Blues_d', 7))
        plt.xlabel('day of week')
        plt.ylabel('orders')
        plt.title('Frequency of orders by day of week')
        return f

    def create_oeda6(self):
        f, ax = plt.subplots(figsize=(5, 4))
        dist_h_orders = orders.order_hour_of_day.value_counts()
        sns.barplot(dist_h_orders.index, dist_h_orders.values, palette=sns.color_palette('Greens_d', 24))
        plt.xlabel('hour of day')
        plt.ylabel('orders')
        plt.title('Frequency of orders by hour of day')
        return f

    def create_oeda7(self):
        f, ax = plt.subplots(figsize=(5, 4))
        grouped = orders.groupby(['order_dow', 'order_hour_of_day']).order_number.count().reset_index()
        # using reset_index to set order_dow and ordr_h_day as columns, or they would be index
        time_orders = grouped.pivot('order_dow', 'order_hour_of_day', 'order_number')
        sns.heatmap(time_orders, cmap='YlOrRd')
        plt.ylabel('Day of Week')
        plt.xlabel('Hour of Day')
        plt.title('Number of Orders Day of Week vs Hour of Day')
        return f

    def create_oeda8(self):
        f, ax = plt.subplots(figsize=(5, 4))
        dist_d_prior_orders = orders.days_since_prior_order.value_counts()
        sns.barplot(dist_d_prior_orders.index, dist_d_prior_orders.values, palette=sns.color_palette('Greens_d', 31))
        plt.xlabel('days of prior order')
        plt.ylabel('count')
        plt.title('Time interval between orders')
        return f

    def create_peda1(self):
        f, ax = plt.subplots(figsize=(5, 4))
        grouped = product.groupby("department")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
        grouped = grouped.groupby(['department']).sum()['Total_products'].sort_values(ascending=False)
        plt.xticks(rotation='vertical')
        sns.barplot(grouped.index, grouped.values)
        plt.ylabel('Number of products', fontsize=13)
        plt.xlabel('Departments', fontsize=13)
        plt.title("The number of products in each department")
        return f

    def create_peda2(self):
        f, ax = plt.subplots(figsize=(5, 4))
        grouped2 = product.groupby("aisle")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
        grouped2 = grouped2.sort_values(by='Total_products', ascending=False)[:20]

        grouped2 = grouped2.groupby(['aisle']).sum()['Total_products'].sort_values(ascending=False)

        plt.xticks(rotation='vertical')
        sns.barplot(grouped2.index, grouped2.values)
        plt.ylabel('Number of products', fontsize=13)
        plt.xlabel('Aisles', fontsize=13)
        plt.title("The number of products in each aisle(top 20)")
        return f

    def create_peda3(self):
        f, ax = plt.subplots(figsize=(5, 4))
        grouped4 = order_flow.groupby("department")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
        grouped4.sort_values(by='Total_orders', ascending=False, inplace=True)

        grouped4 = grouped4.groupby(['department']).sum()['Total_orders'].sort_values(ascending=False)
        plt.xticks(rotation='vertical')
        sns.barplot(grouped4.index, grouped4.values)
        plt.ylabel('Number of Orders', fontsize=13)
        plt.xlabel('Departments', fontsize=13)
        plt.title('Sales in each department')
        return f

    def create_peda4(self):
        f, ax = plt.subplots(figsize=(5, 4))
        grouped5 = order_flow.groupby("aisle")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
        grouped5.sort_values(by='Total_orders', ascending=False, inplace=True)
        grouped5 = grouped5.groupby(['aisle']).sum()['Total_orders'].sort_values(ascending=False)[:15]
        plt.xticks(rotation='vertical')
        sns.barplot(grouped5.index, grouped5.values)
        plt.ylabel('Number of Orders', fontsize=13)
        plt.xlabel('Aisles', fontsize=13)
        plt.title('Sales in each aisle')
        return f

    def create_peda5(self):
        f, ax = plt.subplots(figsize=(5, 4))
        grouped6 = all_order.groupby("reordered")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
        grouped6 = grouped6.groupby(['reordered']).sum()['Total_products'].sort_values(ascending=False)

        sns.barplot(grouped6.index, grouped6.values)
        plt.ylabel('Number of Products', fontsize=13)
        plt.xlabel('Reordered or Not Reordered', fontsize=13)
        plt.ticklabel_format(style='plain', axis='y')
        plt.title("Not reorder vs Reorder")
        return f

    def create_peda6(self):
        f, ax = plt.subplots(figsize=(5, 4))
        self.grouped7 = all_order.groupby("product_id")["reordered"].aggregate({'reorder_sum': sum, 'total': 'count'}).reset_index()
        self.grouped7['reord_ratio'] = self.grouped7['reorder_sum'] / self.grouped7['total']
        self.grouped7 = pd.merge(self.grouped7, product, how='left', on=['product_id'])
        grouped8 = self.grouped7.sort_values(['reord_ratio'], ascending=False).head(10)

        sns.barplot(grouped8['product_name'], grouped8['reord_ratio'])
        plt.ylim([0.85, 0.95])
        plt.xticks(rotation='vertical')
        plt.title('Top 10 reordered rate')
        return f

    def create_peda7(self):
        f, ax = plt.subplots(figsize=(5, 4))
        grouped9 = self.grouped7.sort_values(['reord_ratio'], ascending=False)
        sns.lineplot(grouped9['department'], grouped9['reord_ratio'])
        plt.xticks(rotation='vertical')
        plt.title('Reordered ratio in each department')
        return f

    def create_peda8(self):
        f, ax = plt.subplots(figsize=(5, 4))
        prior["add_to_cart_order_mod"] = prior["add_to_cart_order"].copy()
        prior["add_to_cart_order_mod"].loc[prior["add_to_cart_order_mod"] > 70] = 70
        grouped_df = prior.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()
        sns.pointplot(grouped_df['add_to_cart_order_mod'].values, grouped_df['reordered'].values, alpha=0.8)
        plt.ylabel('Reorder ratio', fontsize=12)
        plt.xlabel('Add to cart order', fontsize=12)
        plt.title("Add to cart order - Reorder ratio", fontsize=15)
        plt.xticks(rotation='vertical')
        return f

    def oeda1(self):
        self.canvas1.get_tk_widget().destroy()
        self.canvas1._tkcanvas.destroy()
        fig1 = self.create_oeda1()
        self.canvas1 = FigureCanvasTkAgg(fig1, master = self.tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = 1)
        self.canvas1._tkcanvas.pack(side = tk.LEFT, fill = tk.BOTH, expand = 1)

        self.canvas2.get_tk_widget().destroy()
        self.canvas2._tkcanvas.destroy()
        fig2 = self.create_oeda2()
        self.canvas2 = FigureCanvasTkAgg(fig2, master = self.tab1)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def oeda2(self):
        self.canvas1.get_tk_widget().destroy()
        self.canvas1._tkcanvas.destroy()
        fig1 = self.create_oeda3()
        self.canvas1.get_tk_widget().pack_forget()
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas1._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas2.get_tk_widget().destroy()
        self.canvas2._tkcanvas.destroy()
        fig2 = self.create_oeda4()
        self.canvas2.get_tk_widget().pack_forget()
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.tab1)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def oeda3(self):
        self.canvas1.get_tk_widget().destroy()
        self.canvas1._tkcanvas.destroy()
        fig1 = self.create_oeda5()
        self.canvas1.get_tk_widget().pack_forget()
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas1._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas2.get_tk_widget().destroy()
        self.canvas2._tkcanvas.destroy()
        fig2 = self.create_oeda6()
        self.canvas2.get_tk_widget().pack_forget()
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.tab1)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def oeda4(self):
        self.canvas1.get_tk_widget().destroy()
        self.canvas1._tkcanvas.destroy()
        fig1 = self.create_oeda7()
        self.canvas1.get_tk_widget().pack_forget()
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas1._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas2.get_tk_widget().destroy()
        self.canvas2._tkcanvas.destroy()
        fig2 = self.create_oeda8()
        self.canvas2.get_tk_widget().pack_forget()
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.tab1)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def peda1(self):
        self.canvas1.get_tk_widget().destroy()
        self.canvas1._tkcanvas.destroy()
        fig1 = self.create_peda1()
        self.canvas1.get_tk_widget().pack_forget()
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas1._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas2.get_tk_widget().destroy()
        self.canvas2._tkcanvas.destroy()
        fig2 = self.create_peda2()
        self.canvas2.get_tk_widget().pack_forget()
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.tab1)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def peda2(self):
        self.canvas1.get_tk_widget().destroy()
        self.canvas1._tkcanvas.destroy()
        fig1 = self.create_peda3()
        self.canvas1.get_tk_widget().pack_forget()
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas1._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas2.get_tk_widget().destroy()
        self.canvas2._tkcanvas.destroy()
        fig2 = self.create_peda4()
        self.canvas2.get_tk_widget().pack_forget()
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.tab1)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def peda3(self):
        self.canvas1.get_tk_widget().destroy()
        self.canvas1._tkcanvas.destroy()
        fig1 = self.create_peda5()
        self.canvas1.get_tk_widget().pack_forget()
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas1._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas2.get_tk_widget().destroy()
        self.canvas2._tkcanvas.destroy()
        fig2 = self.create_peda6()
        self.canvas2.get_tk_widget().pack_forget()
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.tab1)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def peda4(self):
        self.canvas1.get_tk_widget().destroy()
        self.canvas1._tkcanvas.destroy()
        fig1 = self.create_peda7()
        self.canvas1.get_tk_widget().pack_forget()
        self.canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas1._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas2.get_tk_widget().destroy()
        self.canvas2._tkcanvas.destroy()
        fig2 = self.create_peda8()
        self.canvas2.get_tk_widget().pack_forget()
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.tab1)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas2._tkcanvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

class MyMenu():
    def __init__(self, root):
        self.root = root
        self.menubar = tk.Menu(self.root)
        # add menu to root
        root.config(menu=self.menubar)

        self.myMenu = tk.Menu(self.menubar)
        self.myMenu.add_command(label = "About", command = self.about)

        self.myMenu.add_separator()
        self.myMenu.add_command(label = "quit", command = root.quit)

        # add menu to Menu
        self.menubar.add_cascade(label = "Menu", menu = self.myMenu)

    def about(self):
        pass

def LoadAndPreprocess():
    global aisles
    global departments
    global train
    global prior
    global orders
    global products
    global product
    global all_order
    global order_flow
    global data
    global X_train
    global X_test
    global y_train
    global y_test
    global LRy_pred
    global LRy_pred_score
    global KNNy_pred
    global KNNy_pred_score
    global RFy_pred
    global RFy_pred_score

    aisles = pd.read_csv('aisles.csv')
    departments = pd.read_csv('departments.csv')
    train = pd.read_csv('order_products__train.csv')
    prior = pd.read_csv('order_products__prior.csv')
    orders = pd.read_csv('orders.csv')
    products = pd.read_csv('products.csv')
    data = pd.read_csv('data.csv')
    product = products.merge(aisles).merge(departments)
    all_order = pd.concat([train, prior], axis=0)
    order_flow = orders[['user_id', 'order_id']].merge(all_order[['order_id', 'product_id']]).merge(product)
    Xdata, Ydata = data.drop('reordered', axis=1), data.reordered
    ros = RandomOverSampler(random_state=0)
    Xdata, Ydata = ros.fit_sample(Xdata, Ydata)
    X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.3, random_state=100)
    LRy_pred, LRy_pred_score = LRModel()
    KNNy_pred, KNNy_pred_score = KNNModel()
    RFy_pred, RFy_pred_score = RFModel()

def featureSelection():

    prior_orders = orders.merge(prior, on='order_id', how='inner')
    prior_orders.head()

    user_f_1 = prior_orders.groupby('user_id').order_number.max().reset_index(name='n_orders_users')
    user_f_2 = prior_orders.groupby('user_id').product_id.count().reset_index(name='n_products_users')
    user_f_2['avg_products_users'] = user_f_2.n_products_users / user_f_1.n_orders_users
    temp = prior_orders.groupby('user_id')['order_dow'].value_counts().reset_index(name='times_d')
    user_f_3 = temp.loc[temp.groupby('user_id')['times_d'].idxmax(), ['user_id', 'order_dow']]
    user_f_3 = user_f_3.rename(columns={'order_dow': 'dow_most_user'})
    temp = prior_orders.groupby('user_id')['order_hour_of_day'].value_counts().reset_index(name='times_h')
    user_f_4 = temp.loc[temp.groupby('user_id')['times_h'].idxmax(), ['user_id', 'order_hour_of_day']]
    user_f_4 = user_f_4.rename(columns={'order_hour_of_day': 'hod_most_user'})
    user_f_5 = prior_orders.groupby('user_id').reordered.mean().reset_index(name='reorder_ratio_user')
    order_user_group = prior_orders.groupby('user_id')
    user_f_6 = (order_user_group.days_since_prior_order.sum() / order_user_group.days_since_prior_order.count()).reset_index(name='shopping_freq')

    user = pd.merge(user_f_1, user_f_2, on='user_id')
    user = user.merge(user_f_3, on='user_id')
    user = user.merge(user_f_4, on='user_id')
    user = user.merge(user_f_5, on='user_id')
    user = user.merge(user_f_6, on='user_id')

    prod_f_1 = prior.groupby('product_id').order_id.count().reset_index(name='times_bought_prod')
    prod_f_2 = prior.groupby('product_id').reordered.mean().reset_index(name='reorder_ratio_prod')
    prod_f_3 = prior.groupby('product_id').add_to_cart_order.mean().reset_index(name='position_cart_prod')
    prod_dep = pd.merge(departments['department_id'], products[['department_id', 'product_id']], on='department_id', how='right')
    totall_info = pd.merge(prod_dep, prod_f_2, on='product_id', how='right')

    group = totall_info.groupby('department_id')
    prod_f_4 = group.reorder_ratio_prod.mean().reset_index(name='reorder_ratio_dept')

    prod_f_4 = pd.merge(prod_f_4, totall_info, on='department_id')
    del prod_f_4['reorder_ratio_prod']

    prod = pd.merge(prod_f_1, prod_f_2, on='product_id')
    prod = prod.merge(prod_f_3, on='product_id')
    prod = prod.merge(prod_f_4, on='product_id')

    user_prd_f_1 = prior_orders.groupby(['user_id', 'product_id']).order_id.count().reset_index(name='times_bought_up')

    # number of orders for one user
    user_prd_f_2 = prior_orders.groupby('user_id').order_number.max().reset_index(name='n_orders_users')
    # when the user bought the product for the first time
    temp = prior_orders.groupby(['user_id', 'product_id']).order_number.min().reset_index(name='first_bought_number')
    # merge two datasets
    user_prd_f_2 = pd.merge(user_prd_f_2, temp, on='user_id')
    # how many orders performed after the user bought the product for the first time
    user_prd_f_2['order_range'] = user_prd_f_2['n_orders_users'] - user_prd_f_2['first_bought_number'] + 1
    # reordered ratio
    user_prd_f_2['reorder_ratio_up'] = user_prd_f_1.times_bought_up / user_prd_f_2.order_range
    user_prd_f_2 = user_prd_f_2.loc[:, ['user_id', 'product_id', 'reorder_ratio_up']]

    # Reversing the order number for each product.
    prior_orders['order_number_back'] = prior_orders.groupby('user_id')['order_number'].transform(max) - prior_orders[
        'order_number'] + 1

    temp1 = prior_orders.loc[prior_orders['order_number_back'] <= 4]
    user_prd_f_3 = (temp1.groupby(['user_id', 'product_id'])['order_number_back'].count() / 4).reset_index(
        name='ratio_last4_orders_up')

    user_prd = pd.merge(user_prd_f_1, user_prd_f_2, on=['user_id', 'product_id'])
    user_prd = user_prd.merge(user_prd_f_3, on=['user_id', 'product_id'], how='left')

    user_prd.ratio_last4_orders_up.fillna(0, inplace=True)

    data = user_prd.merge(user, on='user_id', how='left')
    data = data.merge(prod, on='product_id', how='left')
    orders_future = orders[((orders.eval_set == 'train') | (orders.eval_set == 'test'))]
    orders_future = orders_future[['user_id', 'eval_set', 'order_id']]
    data = data.merge(orders_future, on='user_id', how='left')
    data = data[data.eval_set == 'train']
    data = data.merge(train[['product_id', 'order_id', 'reordered']], on=['product_id', 'order_id'], how='left')
    data['reordered'] = data['reordered'].fillna(0)
    data = data.drop(['department_id', 'eval_set', 'order_id'], axis=1)
    data = data.set_index(['user_id', 'product_id'])
    #data.to_csv("data.csv")
    return data

def LRModel():
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_score = clf.predict_proba(X_test)
    return y_pred, y_pred_score

def KNNModel():
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_score = clf.predict_proba(X_test)
    return y_pred, y_pred_score

def RFModel():
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_score = clf.predict_proba(X_test)
    return y_pred, y_pred_score

def main():
    app = Application()
    app.addmenu(MyMenu)
    app.mainloop()

if __name__ == '__main__':

    LoadAndPreprocess()
    #featureSelection()
    main()
    
    
    
    
