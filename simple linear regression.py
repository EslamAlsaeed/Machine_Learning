# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:28:19 2023

@author: Eslam
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
 
# importing data 
 
datasheet = pd.read_csv('Salary_Data.csv')
x=datasheet.iloc[:,:1].values
y=datasheet.iloc[:,1:].values


# missing data 


"""from sklearn.impute import SimpleImputer  
imputer =SimpleImputer(missing_values=np.nan ,strategy='mean')
x=imputer.fit_transform(x)"""

#don't need it here 



#test and training


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=(0)) 

"""
#preprocessing data for scaler 

from sklearn.preprocessing import StandardScaler

sc_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_train=sc_x.fit_transform(x_test)
y_train=sc_x.fit_transform(y_train)
y_test=sc_x.fit_transform(y_test)

"""

#fitting simple linear regression  to the traingin set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  #varible for save function 
regressor.fit(x_train , y_train) #corresion between to varible


#predicting the test set results 

y_pred = regressor.predict(x_test)
y_pred_train =regressor.predict(x_train)  #for training 



#drawn function  for training

plt.scatter(x_train,y_train,color='red') #for axais y and x
plt.plot(x_train, y_pred_train ,color='blue')  #for slope 
plt.title('Salary vs Experience (training set)')  #for title upper drawn 
plt.xlabel('Years of Experience') #for title of axias x
plt.ylabel('Salary') #for title of axias y 
plt.show() #for drawn


#drawn function for test 

plt.scatter(x_test,y_test ,color ='red')
plt.plot(x_train, y_pred_train ,color='blue')
plt.title('Salary vs Experience (test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()