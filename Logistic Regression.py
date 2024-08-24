# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:52:12 2023

@author: Eslam
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
datasheet =pd.read_csv('Social_Network_Ads.csv')
x=datasheet.iloc[:,1:4].values
y=datasheet.iloc[:,4:].values



from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
x[:,2:]=imputer.fit_transform(x[:,2:]).astype(int)



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
e=x[:,0]
label_encoder=LabelEncoder()
e=label_encoder.fit_transform(e)
one_hot_encoder=OneHotEncoder(sparse=(False))
e=e.reshape(len(e),1)
e=one_hot_encoder.fit_transform(e)
x=np.delete(x,0,axis=1)
x=np.append(e.astype(int), values=x,axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x[:,2:],y,test_size=0.2,random_state=(0))

#"""

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
#y_train=sc.fit_transform(y_train)
#y_test=sc.fit_transform(y_test)

#"""


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
#x_train, y_train = load_iris(return_X_y=True)
classifer=LogisticRegression(random_state=(0))

classifer.fit(x_train,y_train)
y_pred=classifer.predict(x_test)
classifer.predict_proba(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifer.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifer.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""

plt.scatter(x_train, y_train, color='red')
plt.plot(x,y_pred,color='blue')
plt.title('logistic Regression')
plt.xlabel('Age')
plt.ylabel('Estimated')
plt.show()

#"""