import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

#for prepering data we do next 
#and x for input y for output

dataset = pd.read_csv('Data.csv')
x= dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values
#f=dataset.iloc[0:,:].values

#for missing data we do next

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan , strategy= 'mean' )
x[:,1:3]= imputer.fit_transform(x[:,1:3])
#print(x)


#for one hot encoder input data 


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
e=x[:,0]
label_encoder_x=LabelEncoder()
e=label_encoder_x.fit_transform(e)
one_input=OneHotEncoder(sparse=(False))
e=e.reshape(len(e),1)
e=one_input.fit_transform(e)






#for one hot encoder output data
label_encoder_y=LabelEncoder()
out=label_encoder_y.fit_transform(y)
one_out=OneHotEncoder(sparse=(False))
out=out.reshape(len(out),1)
one_out=one_out.fit_transform(out)
y=out



#splitting the data to training  
#


from sklearn.model_selection import train_test_split
x_train,x_test, y_train , y_test =train_test_split(x,y ,test_size=0.2 , random_state=(0))


#feature scalling


from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler() #varibal for function
x_train=sc_x.fit_transform(x_train[:,1:]) #scal for x_train
x_test=sc_x.transform(x_test[:,1:])  #scal for x_test 
y_train=sc_x.fit_transform(y_train)  #scal for y_train
y_test=sc_x.fit_transform(y_test)    #scal fro y_test







#l=pd.Series(list(x[:,0:1]))
#print(l)
#print(pd.get_dummies(l))
#print(x[:,0:1])
#s=pd.Series(list('abca'))
#print(pd.get_dummies(s))
