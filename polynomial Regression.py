# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:04:34 2023

@author: Eslam
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

datasheet =pd.read_csv('Position_Salaries.csv')
x=datasheet.iloc[:,1:2].values
y=datasheet.iloc[:,2:].values

from sklearn.impute import SimpleImputer
imputer =SimpleImputer(missing_values=np.nan,strategy='mean')
x=imputer.fit_transform(x)


"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x[:,1:2]=sc.fit_transform(x[:,1:2])
y=sc.fit_transform(y)
"""


"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
e=x[:,0]
label_encoder=LabelEncoder()
e=label_encoder.fit_transform(e)
One_hot_encoder=OneHotEncoder(sparse=(False))
e=e.reshape(len(e),1)
e=One_hot_encoder.fit_transform(e)
x=np.delete(x,0,axis=1)
x=np.append(e.astype(int), values=x.astype(int),axis=1) 
"""


"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=(0))
"""



from sklearn.linear_model import LinearRegression
regressive=LinearRegression()
x_linear=regressive.fit(x,y)
#y_predit=regressive.predict(x)

from sklearn.preprocessing import PolynomialFeatures
 
polynom=PolynomialFeatures(degree=4)
x_pol=polynom.fit_transform(x)

linear_reg_2=LinearRegression()
linear_reg_2.fit(x_pol,y)

#drawn linear eqution 

plt.scatter(x,y,color='red')
plt.plot(x,x_linear.predict(x),color='blue' )
plt.title('linear Relation ')
plt.xlabel('Level ')
plt.ylabel('Salary')
plt.show()

#drawn polynomial eqution
plt.scatter(x,y,color='red')
plt.plot(x, linear_reg_2.predict(x_pol))
plt.title('Polynomial Relation')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


#drawn with more resloutions

x_resloution=np.arange(min(x),max(x),0.1)
x_resloution=x_resloution.reshape(len(x_resloution),1)
plt.scatter(x,y,color='red')
plt.plot(x_resloution,linear_reg_2.predict(polynom.fit_transform(x_resloution)), color='blue')
plt.title('Plonymial with more Resloution')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#reslut of linear model

print(regressive.predict(np.array([[6.5]])))

#reslut of polnmial model
print(linear_reg_2.predict(polynom.fit_transform(np.array([[6.5]]))))
