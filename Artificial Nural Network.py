# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:51:41 2023

@author: Eslam
"""
#

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13:].values



from sklearn.preprocessing import LabelEncoder , OneHotEncoder
e=x[:,1]
f=x[:,2]
label_enconder=LabelEncoder()
e=label_enconder.fit_transform(e)
f=label_enconder.fit_transform(f)
one_encoder=OneHotEncoder(sparse=(False))
e=e.reshape(len(e),1)
f=f.reshape(len(f),1)
e=one_encoder.fit_transform(e)
#f=one_encoder.fit_transform(f)
x=np.delete(x,1,axis=1 )
x=np.delete(x,1,axis=1 )
x=np.append(f.astype(int), x,axis=1)
x=np.append(e.astype(int), x,axis=1)
x=np.delete(x,0,axis=1 )

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=(0))

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)




#import keras 
from keras.models import Sequential
from keras.layers import Dense

#Intialsation ANN

classifier=Sequential()

#Adding input layer and first hidden layer

classifier.add(Dense(output_dim=6,init='uniform',activation='rule',input_dim=11))

#Adding second hidden layer

classifier.add(Dense(output_dim=6,init='uniform',activation='rule'))

#Output layer

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ANN

classifier.compile(optimizer='adam',loss=('binary_crossentropy'),metrics=('accuracy'))


classifier.fit(x_train,y_train,batch_size=(10),nb_epoch=100)

y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)


