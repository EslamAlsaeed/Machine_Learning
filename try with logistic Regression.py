# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:45:30 2023

@author: Eslam
"""

from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import make_classification
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
datasheet =pd.read_csv('Social_Network_Ads.csv')
x=datasheet.iloc[:,1:4].values
y=datasheet.iloc[:,4:].values

"""
x,y=make_classification(
    n_samples=400,
    n_features=2,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=2,
    n_redundant=0,
    n_repeated=0
    )
#"""

"""
plt.scatter(x,y,c=y,cmap=('rainbow'))
plt.title('show function')
plt.show()

#"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x[:,1:],y,test_size=0.2,random_state=(0))

#"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_train_test = sc.transform(x_test)
#"""
from sklearn.linear_model import LogisticRegression
classifer=LogisticRegression(random_state=(0))
classifer.fit(x_train,y_train)
y_pred=classifer.predict(x_test)

from sklearn.metrics import confusion_matrix ,accuracy_score
cm=confusion_matrix(y_test, y_pred)
ac=accuracy_score(y_test, y_pred)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


