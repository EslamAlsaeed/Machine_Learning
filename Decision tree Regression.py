# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:56:40 2023

@author: Eslam
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

datasheet =pd.read_csv('Position_Salaries.csv')
x=datasheet.iloc[:,1:2].values
y=datasheet.iloc[:,2:]


"""

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
y=sc.fit_transform(y)

"""
from sklearn.tree import DecisionTreeRegressor
regessive=DecisionTreeRegressor(random_state=(0))
regessive.fit(x,y)

pred=regessive.predict(np.array([[6.5]]))


"""
x_grad=np.arange(max(x),min(x),0.1)
x_grad=x_grad.reshape(len(x_grad), 1)
"""
plt.scatter(x, y, color='red')
plt.plot(x,regessive.predict(x),color='blue')
plt.title('Decision tree Regression')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()
