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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import chi2_contingency
from sklearn import linear_model
from sklearn import cluster 
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

# Data extraction and exploratory Analysis
rawdat=pd.read_csv(r'C:\Users\rramn\Documents\Kaggle_projects\Telecom_churn\cell2celltrain.csv')

df_vis=pd.DataFrame(rawdat)
df_vis['MonthlyRevenue'].max()
type(df_vis)
#print(np.dtype(t.iloc[0]))
nrow,ncol=df_vis.shape
colnames=list(df_vis)
mp.figure(num=None,figsize=(6*8,8*ncol),dpi=80)
for i in range(ncol):
    mp.subplot(ncol,8,i+1)
    t=df_vis.iloc[:,i]
    if(np.issubdtype(type(t.iloc[0]),np.number)):
        #t.hist()
        sns.boxplot(t)
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


#Bivariate analysis
mp.figure(figsize=(40,30))
crdf=df_vis.select_dtypes(exclude='category')
cor=crdf.corr(method='pearson')
sns.heatmap(cor)
mp.show()



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
df_vis.Churn.value_counts()

#Missing values imputation
temp=df_vis
temp=temp.fillna(temp.mean())
imputer=CategoricalImputer(missing_values='NaN', strategy='most_frequent')
imputer.fit_transform(temp['ServiceArea'])
temp=temp.apply(lambda x: x.fillna(x.value_counts().index[0]))
temp.isna().sum()

#Statistical analysis
nr,nc=temp.shape
for j in range(nc):
    if((temp.iloc[:,j].dtype!=np.int64) & (temp.iloc[:,j].dtype!=np.number)):
        xx=temp.iloc[:,j]
        yy=temp['Churn']
        ct=pd.crosstab(xx,yy) 
        ch=chi2_contingency(ct)
        print('Chisquare result for',temp.columns[j],'is  ',ch)
    else:
        print('Not a category')        

#Predicting Churn index
#Label encoding categorical variables
temp.dtypes
cl_df=temp
cl_df.isna().sum()
cl_df=cl_df.drop(['CustomerID'],axis=1)

#cl_df['Churn']=cl_df['Churn'].cat.codes
cl_df['Churn']=LabelEncoder().fit_transform(cl_df.Churn)
cl_df.Churn.value_counts()
for n,v in cl_df.iteritems():
    if((cl_df[n].dtype!=np.int64) & (cl_df[n].dtype!=np.number) & (n!='Churn')):
        mean_encode=cl_df.groupby(n)['Churn'].mean()
        print(mean_encode)
        cl_df.loc[:,n]=cl_df[n].map(mean_encode)

cl_df.dtypes
x_cl=pd.DataFrame(cl_df.iloc[:,1:])
y_cl=cl_df.iloc[:,0]

#Logistic Regression significance test

lrr=LogisticRegression(penalty="l1")
lrr.fit(lrdf,y_cl)

#cat_columns=x_cl.select_dtypes(['category']).columns
#cat_columns
#x_cl[cat_columns]=x_cl[cat_columns].apply(lambda x: x.cat.codes)
#y_cl=y_cl.cat.codes

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
imp_feat=pd.Series(rf.feature_importances_, index=x_cl.columns)
imp_feat.nlargest(20).plot(kind='barh')
ind=np.argsort(imp)
rf.feature_importances_

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


#Predicting Monthly Revenue
reg_df=temp
cl_df['Churn']
reg_df['Revenue_loss']=cl_df.Churn * reg_df['MonthlyRevenue']
reg_df['Revenue_loss']
reg_df=reg_df.drop(['MonthlyRevenue'],axis=1)
#Casting categorical types 
for n,v in reg_df.iteritems():
    if((reg_df[n].dtype!=np.int64) & (reg_df[n].dtype!=np.number)):
        mean_encode=reg_df.groupby(n)['Revenue_loss'].mean()
        print(mean_encode)
        reg_df.loc[:,n]=reg_df[n].map(mean_encode)

reg_df.dtypes
y_reg=reg_df['Revenue_loss']
x_reg=pd.DataFrame(reg_df.drop(['CustomerID','Revenue_loss'],axis=1))
x_reg.shape
#scaler=StandardScaler()
#x_reg=scaler.fit_transform(x_reg)

min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x_reg)
x_reg = pd.DataFrame(x_scaled)
#Stochastic gradient Descent and Neural Networks

sgdr= linear_model.LinearRegression()
sgdr.fit(x_reg,y_reg)
np.sqrt(metrics.mean_squared_error(y_reg,sgdr.predict(x_reg)))


nnr=MLPRegressor(hidden_layer_sizes=(20,),activation='relu')
nnr.fit(x_reg,y_reg)
metrics.mean_squared_error(y_reg,nnr.predict(x_reg))

#Customer segmentation
clus_df=reg_df
scaler=StandardScaler()
scale_df=scaler.fit_transform(clus_df)


#Choosing the best k
cluster_range=[2,3,4,5,6,7,8,9,10]
distortions=[]
for k in cluster_range:
    clus=cluster.KMeans(n_clusters=k,max_iter=100)
    clus.fit(scale_df)
    distortions.append(sum(np.min(cdist(scale_df,clus.cluster_centers_,'euclidean'), axis=1))/scale_df.shape[0])
    
mp.plot(cluster_range,distortions,'bx-')
mp.xlabel('k')
mp.ylabel('Distortion')
mp.title('Elbow method showing optimal k')
mp.show()

#Finding silhouette measure with best k
clusk=cluster.KMeans(n_clusters=5)
clusk.fit(scale_df)
metrics.silhouette_score(scale_df,clusk.labels_)
pred=clusk.transform(scale_df)
clust=pd.DataFrame(pred)
clust.shape
sns.barplot(x=clust,y=df_vis['MonthlyMinutes'])

