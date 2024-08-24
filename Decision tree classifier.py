# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:56:18 2023

@author: Eslam
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
datasheet=pd.read_csv('Social_Network_Ads.csv')
x=datasheet.iloc[:,:2].values
y=datasheet.iloc[:,2:].values


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
x=imputer.fit_transform(x)

"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
e=x
label_hot=LabelEncoder()
e=label_hot.fit_transform(e)
one_encoder=OneHotEncoder(sparse=(False))
e=e.reshape(len(e),1)
e=one_encoder.fit_transform(e)
x=np.delete(x, 3,axis=1)
x=np.append(e.astype(int), values=x,axis=1)

#"""

#"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=(0))

#"""

#"""

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#"""

from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion='entropy',random_state=(0))
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)


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
plt.title('Decision tree classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
