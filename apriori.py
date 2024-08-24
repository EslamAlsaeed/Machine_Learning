# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:11:45 2023

@author: Eslam
"""


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

datasheet=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
#x=datasheet.iloc[:,3:].values
#y=datasheet.iloc[:,4:].values

transcacrion=[]
for i in range(0,7501):
    transcacrion.append([str(datasheet.values[i,j]) for j in range(0,20)]) 
    
from apriori import apriori
rules =apriori(transcacrion,min_confidence=0.2,min_lift=3,min_length=2)
list(rules)
