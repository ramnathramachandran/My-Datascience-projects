# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 23:02:44 2020

@author: rramn
"""
#Importig Relevant packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

#Objective

#Prdict Number of shares an article can get
#Classifying the articles into different categories.. Which category should be published maximum for higher number of shares?
#For different categories of articles what should be their min and max content length
#
#

# Data extraction and exploratory Analysis

dat=pd.read_csv(r'C:\Users\rramn\Documents\projects\Mashable_data\OnlineNewsPopularity.csv')

mash_df=pd.DataFrame(dat)

mash_df.shape
mash_df.dtypes
#mash_df.timedelta
mash_df.head()

#Dropping duplicate rows and the URL column
mash_df=mash_df.drop_duplicates()
mash_df=mash_df.drop(['url'],axis=1)
mash_df.head()

#Checking for missing values
mash_df.isnull().sum()
mash_df.describe()

# Exploratory analysis using visualization
mash_df['shares'].max()-mash_df['shares'].min()
sns.scatterplot(mash_df['n_tokens_content'],mash_df['shares'])
sns.scatterplot(mash_df['n_tokens_title'],mash_df['shares'])

#correlation analysis using a plot
mp.figure(figsize=(40,30))
cor=mash_df.corr(method='pearson')
sns.heatmap(cor)
mp.show()

#Since some of the variables are highly correlated PCA would be a feasible to solution to reduce the dimension of the data

X=mash_df.drop('shares',1)
Y=mash_df['shares']

#PCA analysis
#scaler=StandardScaler()
#scaler.fit(X)
#
#pca_d=scaler.transform(X)
#
#pca=PCA(svd_solver='full')
#X_new=pca.fit_transform(X)
#X_or=pca.inverse_transform(X_new)

min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X_t = pd.DataFrame(x_scaled)

x_train,x_test,y_train,y_test=train_test_split(X_t,Y,test_size=0.2,random_state=0)

#Stochastic gradient descent

lr=linear_model.SGDRegressor(max_iter=1000,penalty='l1')
lr.fit(x_train,y_train)
lr.coef_
pred=lr.predict(x_test)
np.sqrt(metrics.mean_squared_error(y_test,pred))

#Random forest regressor

rf=RandomForestRegressor(min_samples_split=9)
rf.fit(x_train,y_train)
pred_rf=rf.predict(x_test)
np.sqrt(metrics.mean_squared_error(y_test,pred_rf))


# SGD with transformed PCA data

#min_max_scaler = MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(X_or)
#X_or = pd.DataFrame(x_scaled)
#xtrainp,xtestp,ytrainp,ytestp=train_test_split(X_or,Y,test_size=0.2,random_state=0)
#lr=linear_model.SGDRegressor(max_iter=1000)
#lr.fit(xtrainp,ytrainp)
#pred=lr.predict(xtestp)
#np.sqrt(metrics.mean_squared_error(ytestp,pred))

#Lasso with CV
reg=linear_model.LassoCV(cv=10,random_state=0).fit(x_train,y_train)
reg.score(x_train,y_train)
predls=reg.predict(x_test)
np.sqrt(metrics.mean_squared_error(y_test,predls))
reg.coef_

#Neural net

nnr=MLPRegressor(hidden_layer_sizes=(20,),activation='relu')
nnr.fit(x_train,y_train)
np.sqrt(metrics.mean_squared_error(y_test,nnr.predict(x_test)))

#Feature selection using Recursive elimination

mod=linear_model.LinearRegression()
rfe=RFECV(mod,cv=5)
rfl=rfe.fit(X_t,Y)
rfl.n_features_
rfl.ranking_
rfl.support_

scr=cross_val_score(lr,X_t,Y,scoring='neg_mean_squared_error',cv=10)
scr
a=scr
a=a*-1

np.average(np.sqrt(a))
metrics.SCORERS.keys() # To get the list of supported scores



#Classification of articles based on the category

mash_cl=mash_df
mash_cl.shape

#Relabeling the category columns and merging into a single column

lifestyle=mash_cl[mash_cl['data_channel_is_lifestyle']==1]
lifestyle['dat_ch']=1

entertainment=mash_cl[mash_cl['data_channel_is_entertainment']==1]
entertainment['dat_ch']=2

bus=mash_cl[mash_cl['data_channel_is_bus']==1]
bus['dat_ch']=3

soc=mash_cl[mash_cl['data_channel_is_socmed']==1]
soc['dat_ch']=4

tech=mash_cl[mash_cl['data_channel_is_tech']==1]
tech['dat_ch']=5

world=mash_cl[mash_cl['data_channel_is_world']==1]
world['dat_ch']=6

nonclas=mash_cl[(mash_cl['data_channel_is_world']==0) & (mash_cl['data_channel_is_tech']==0) & (mash_cl['data_channel_is_socmed']==0) & (mash_cl['data_channel_is_bus']==0) & (mash_cl['data_channel_is_lifestyle']==0) & (mash_cl['data_channel_is_entertainment']==0)]
nonclas['dat_ch']=0

frame=[lifestyle,entertainment,bus,soc,tech,world,nonclas]
mash_cl=pd.concat(frame,ignore_index=True)

mash_cl=mash_cl.drop(['data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world'],axis=1)
mash_cl['dat_ch']

Xcl=mash_cl.drop('dat_ch',1)
Ycl=mash_cl['dat_ch']

#Randomforest
rfc=RandomForestClassifier(n_estimators=100)
acu=cross_val_score(rfc,Xcl,Ycl,scoring='accuracy',cv=10)
np.average(acu)

#SVM

min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(Xcl)
Xcl_t=pd.DataFrame(x_scaled)

svc=SVC(gamma='auto')
svc.fit(Xcl_t,Ycl)
metrics.accuracy_score(Ycl,svc.predict(Xcl_t))