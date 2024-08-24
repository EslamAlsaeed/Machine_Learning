# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:24:19 2023

@author: Eslam
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:16:45 2023

@author: Eslam
"""


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
datasheet=pd.read_csv('Mall_Customers.csv')
x=datasheet.iloc[:,2:4].values
y=datasheet.iloc[:,4:].values


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
x=imputer.fit_transform(x)

"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
e=x[:,:1]
label_hot=LabelEncoder()
e=label_hot.fit_transform(e)
one_encoder=OneHotEncoder(sparse=(False))
e=e.reshape(len(e),1)
e=one_encoder.fit_transform(e)
x=np.delete(x, 0,axis=1)
x=np.append(e.astype(int), values=x,axis=1)

#"""
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,5):
    k_mean=KMeans(n_clusters=i,init='k-means++',random_state=(42))
    k_mean.fit(x)
    wcss.append(k_mean.inertia_)
    
plt.plot(range(1,5), wcss)
plt.title('the Elbow method')
plt.xlabel('Number of Cluster')
plt.ylabel('wcss')
plt.show()
k_mean=KMeans(n_clusters=i,init='k-means++',random_state=(42))

y_kmean=k_mean.fit_predict(x)


plt.scatter(x[y_kmean==0,0],x[y_kmean==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[y_kmean==1,0],x[y_kmean==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[y_kmean==2,0],x[y_kmean==2,1],s=100,c='green',label='Cluster 2')
plt.scatter(x[y_kmean==3,0],x[y_kmean==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(x[y_kmean==4,0],x[y_kmean==4,1],s=100,c='magenta',label='Cluster 5')
plt.scatter(k_mean.cluster_centers_[:,0],k_mean.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('Cluster of customers')
plt.xlabel('Annal Income')
plt.ylabel('speend Score')
plt.legend()
plt.show()


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

"""

from sklearn.ensemble import RandomForestClassifier 

classifier=RandomForestClassifier(n_estimators=10,random_state=(0),criterion='entropy')
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

#"""


"""

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)


from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K_mean clustering (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#"""