# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:50:48 2020

@author: rramn
"""

#Enabling Intellisense
#%config IPCompleter.greedy=True
#Importing Relevant packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import Imputer
from missingpy import MissForest
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
# Data extraction and exploratory Analysis
rawdat=pd.read_csv(r'C:\Users\rramn\Documents\Kaggle_projects\Telecom_churn\cell2celltrain.csv')

#Data visualization
df_vis=pd.DataFrame(rawdat)
type(df_vis)
#print(np.dtype(t.iloc[0]))
nrow,ncol=df_vis.shape
colnames=list(df_vis)
mp.figure(num=None,figsize=(6*8,8*ncol),dpi=80)
for i in range(ncol):
    mp.subplot(ncol,8,i+1)
    t=df_vis.iloc[:,i]
    if(np.issubdtype(type(t.iloc[0]),np.number)):
        t.hist()
    else:
        df_vis.iloc[:,i]=df_vis.iloc[:,i].astype('category')
        valcount=t.value_counts()
        valcount.plot.bar()
    mp.ylabel('counts')
    mp.title(f'{colnames[i]} (column {i})')
    mp.xticks(rotation=90)
    #mp.title()   
mp.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
mp.show()
df_vis.dtypes
#df_vis['HandsetPrice']=df_vis['HandsetPrice'].astype('int')
#df_vis.dtypes

#Data cleaning
df_vis.info()
sns.heatmap(df_vis.isnull(),cbar=False)
df_vis.isnull().sum()
df_vis.isna().sum()

df_vis['HandsetPrice']=df_vis['HandsetPrice'].replace('Unknown',-1)
df_vis['HandsetPrice']
df_vis['HandsetPrice']=df_vis['HandsetPrice'].astype('int')
df_vis['HandsetPrice']=df_vis['HandsetPrice'].replace(-1,np.NaN)
df_vis['HandsetPrice']

#Missing values imputation
temp=df_vis
temp=temp.fillna(temp.mean())
imputer=CategoricalImputer(missing_values='NaN', strategy='most_frequent')
imputer.fit_transform(temp['ServiceArea'])
temp=temp.apply(lambda x: x.fillna(x.value_counts().index[0]))
temp.isna().sum()

#Predicting Churn index
#Label encoding categorical variables
cl_df=temp
cl_df.isna().sum()
y_cl=cl_df.iloc[:,1]
cl_df=cl_df.drop(['CustomerID'],axis=1)
x_cl=pd.DataFrame(cl_df.iloc[:,1:])
cat_columns=x_cl.select_dtypes(['category']).columns
cat_columns
x_cl[cat_columns]=x_cl[cat_columns].apply(lambda x: x.cat.codes)
y_cl=y_cl.cat.codes

#Feature selection and importance
#
#rf=RandomForestClassifier(min_samples_split=2)
#rf.fit(xcl_train,ycl_train)
#predcl=rf.predict(xcl_test)
#metrics.accuracy_score(ycl_test,predcl)

# Feature selection Lasso regression
lr=SelectFromModel(LogisticRegression(penalty="l1"),max_features=40)
lr.fit(x_cl,y_cl)

lr_fea=lr.get_support()
lr_list=x_cl.loc[:,lr_fea].columns.tolist()
lr_list
#Feature importance using Random forests
rf=RandomForestClassifier(n_estimators=100,min_samples_split=4)
rf.fit(x_cl,y_cl)
imp=rf.feature_importances_
ind=np.argsort(imp)
mp.figure(1)
mp.barh(range(len(ind)),imp[ind],align='center')
mp.yticks(range(len(ind)),x_cl[ind])
mp.show()
x_cl[ind]
# Stacking model
xcl_train,xcl_test,ycl_train,ycl_test=train_test_split(x_cl,y_cl,test_size=0.3)
#Support vector classifier
#svm=SVC(C=5, probability=True,gamma='auto')
#svm.fit(xcl_train,ycl_train)
#          
#
lr=LogisticRegressionCV(cv=10)
lr.fit(x_cl,y_cl)
metrics.accuracy_score(y_cl,lr.predict(x_cl))
metrics.roc_auc_score(y_cl,lr.predict(x_cl))
#nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(xcl_train,ycl_train)
metrics.accuracy_score(ycl_test,gb.predict(xcl_test))  
#rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)
