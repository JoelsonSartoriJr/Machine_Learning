#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:03:53 2019

@author: ubuntu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

dataset = "UCB/Ads_CTR_Optimisation.csv"
df = pd.read_csv(dataset)

N = 10000
d = 10
numbers_of_selections = [0]*d
sums_of_reward = [0]*d
ads_selected = []
total_reward = 0

for n in range(0, N):
    add = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            averange_reward = sums_of_reward[i]/ numbers_of_selections[i]
            delta_i = math.sqrt(3/2* math.log(n+1)/numbers_of_selections[i])
            upper_bound = averange_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] +=1
    reward =  df.values[n, ad]
    sums_of_reward[ad] += reward
    total_reward += reward
    
plt.hist(ads_selected)
plt.title("Ads selections")
plt.xlabel("Ads")
plt.ylabel("Numbers of times each ad as selected")
plt.show()