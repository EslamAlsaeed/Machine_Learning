# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:45:37 2023

@author: Eslam
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dataseet=pd.read_csv('50_Startups.csv')
x=dataseet.iloc[:,:4].values
y=dataseet.iloc[:,4:].values

#"""
from sklearn.impute import  SimpleImputer
imputer =SimpleImputer(missing_values=np.nan , strategy='mean')
x[:,:3]=imputer.fit_transform(x[:,:3])
 
#"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
e=x[:,3]
label_encoder=LabelEncoder()
e=label_encoder.fit_transform(e)
one_hot_encoder=OneHotEncoder(sparse=(False))
e=e.reshape(len(e),1)
e=one_hot_encoder.fit_transform(e)
x=np.append(e.astype(int),values=x,axis=1)
x=np.delete(x, 6,axis=1)
#x=pd.get_dummies(x[:,3],columns=('State'))
#"""
x=x[:,1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=(0))

"""

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
y_train=sc_x.fit_transform(y_train)
y_test=sc_x.fit_transform(y_test)

"""


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train, y_train)
y_predt=regression.predict(x_test)


#for backward  Elimtition


import statsmodels.formula.api as sm
import statsmodels.api as sa
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#x=np.append(arr=np.ones((50,1)).astype(int), values=x ,axis=1 )
x=sa.add_constant(x).astype(int) #for add constant columns for prosessor mastike of method
#important to be integer
x_polt=x[:,[0,1,2,3,4,5]] #for preper data 
regression_ols=sa.OLS(endog=y,exog=x_polt).fit()  #for sing data 
our=regression_ols.summary() #for show data as sl
#print(our)

#second iteration
x_polt=x[:,[0,1,3,4,5]]
regression_ols=sa.OLS(endog=y,exog=x_polt).fit()
our=regression_ols.summary()
#print(our)

#threed iteration 

x_polt=x[:,[0,3,4,5]]
regression_ols=sa.OLS(endog=y,exog=x_polt).fit()
#print(regression_ols.summary())

#four iteration

x_polt=x[:,[0,3,5]]
regression_ols=sa.OLS(endog=y,exog=x_polt).fit()
#print(regression_ols.summary())

#five iteration

x_polt=x[:,[0,3]]
regression_ols=sa.OLS(endog=y,exog=x_polt).fit()
#print(regression_ols.summary())

