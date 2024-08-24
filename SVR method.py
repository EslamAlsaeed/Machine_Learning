# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:12:53 2023

@author: Eslam
"""
#import libary
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#import data 

datasheet =pd.read_csv('Position_Salaries.csv')
x=datasheet.iloc[:,1:2].values
y=datasheet.iloc[:,2:].values

#scaler for data 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
y=sc.fit_transform(y)

#new method for fitting curve to relation

from sklearn.svm  import SVR

regressor =SVR(kernel='rbf')
regressor.fit(x,y )

#test number which required


u=regressor.predict(sc.transform(np.array([[6.5]])))
u=u.reshape(len(u),1)
pred=sc.inverse_transform(u)
#print(pred)

#drawn of method

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Salaries vs level SVR')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()


