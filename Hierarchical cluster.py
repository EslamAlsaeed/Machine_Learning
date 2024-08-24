# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:39:05 2023

@author: Eslam
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

datasheet=pd.read_csv('Mall_Customers.csv')
x=datasheet.iloc[:,3:].values
#y=datasheet.iloc[:,4:].values

"""

from sklearn.impute import SimpleImputer
imputer =SimpleImputer(missing_values=np.nan,strategy='mean')
x=imputer.fit_transform(x)

#"""

"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
e=x
label_encoder=LabelEncoder()
e=label_encoder.fit_transform(e)
one_hot=OneHotEncoder(sparse=(False))
e=one_hot.fit_transform(e)
x=np.delete(x, 4,axis=1)
x=np.append(e.astype(int), values=x,axis=1)

#"""

"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=(0))

#"""

"""

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#"""




import scipy.cluster.hierarchy as sch

#dengrogram =sch.dendgrogram(sch.linkage(x,method='ward'))
dengrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram ')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

y_kmean=hc.fit_predict(x)

plt.scatter(x[y_kmean==0,0],x[y_kmean==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[y_kmean==1,0],x[y_kmean==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[y_kmean==2,0],x[y_kmean==2,1],s=100,c='green',label='Cluster 2')
plt.scatter(x[y_kmean==3,0],x[y_kmean==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(x[y_kmean==4,0],x[y_kmean==4,1],s=100,c='magenta',label='Cluster 5')
#plt.scatter(k_mean.cluster_centers_[:,0],k_mean.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('Cluster of customers')
plt.xlabel('Annal Income')
plt.ylabel('speend Score')
plt.legend()
plt.show()
