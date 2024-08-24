# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:56:42 2023

@author: Eslam
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

import random
n=10000
d=10
ads_selected=[]
number_of_reward_1=[0]*d
number_of_reward_0=[0]*d
total_reward=0
for n in range(0,n): 
  ad=0
  max_random=0  
  for i in range(0,d):
     random_beta= random.betavariate(number_of_reward_1[i]+1,number_of_reward_0[i]+1 )
     if random_beta >max_random:
         max_random=random_beta
         ad=i
     ads_selected.append(ad)
     reward=dataset.values[n,ad]
     if reward==1:
         number_of_reward_1[ad]=number_of_reward_1[ad]+1
     else:
         number_of_reward_0[ad] =number_of_reward_0[ad]+1
     total_reward=total_reward+reward  
plt.hist(ads_selected)
plt.title('mistogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Numbers of times each ad was selected')
plt.show()                 