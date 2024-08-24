# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 21:03:21 2023

@author: Eslam
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

datasheet =pd.read_csv('Position_Salaries.csv')
x=datasheet.iloc[:,1:2].values
y=datasheet.iloc[:,2:]


from sklearn.ensemble import RandomForestRegressor
regessive=RandomForestRegressor(n_estimators=10,random_state=(0))
regessive.fit(x, y)

pred=regessive.predict(np.array([[6.5]]))

x_grad=np.arange(min(x),max(x),0.1)
x_grad=x_grad.reshape(len(x_grad), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grad,regessive.predict(x_grad),color='blue')
plt.title('Random forst  Regression')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()