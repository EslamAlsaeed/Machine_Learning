# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:22:30 2023

@author: Eslam
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter=('\t'),quoting=3)

import re 
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review= re.sub('[^a-zA-Z]','',dataset['Review'][i])
    review=review.lower()
    review= review.split()
    ps= PorterStemmer()
   # review=[ps.stem(word) for word in review if not word in set(stopwords.word('english'))]
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=(0))

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entrop',random_state=(0))
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

#"""