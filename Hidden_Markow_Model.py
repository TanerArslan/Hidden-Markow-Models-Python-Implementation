#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:27:10 2020

@author: tanerarslan
"""

import numpy as np 
import pandas as pd

#prior knowledge
px0 = [0.4, 0.6]

#transition probability
theta = np.array([[0.8, 0.2], [0.1, 0.9]])

#observation probability
phi = {'CO' : [0.1, 0.3], 
       'CR' : [0.2, 0.0],
       'SL' : [0.4, 0.3],
       'SO' : [0.0, 0.3],
       'W' : [0.3, 0.1]}

#convert into data frame
phi = pd.DataFrame(data=phi)

#iterations
T = 7
#observations 
y = ["SL","SO", "CO", "CR", "SL", "SL","SL"]

def HMM(px0, theta, phi, y, T):
    
    x = 0
    for t in range(T):
        if t < len(y):
            x += 1
            
            #the first observation
            if x == 1:
                ### prediction
                x_S = (px0[0] * theta[0][0]) + (px0[1] * theta[1][0])
                x_H = 1.00 - x_S
                
                ###bayes update
                #observed y index
                index_Y = phi.columns.get_loc(y[x-1]) 
                
                x_S_Y = (x_S * phi.iloc[0, index_Y]) / ((x_S * phi.iloc[0, index_Y]) + (x_H * phi.iloc[1, index_Y])) 
                x_H_Y = (x_H * phi.iloc[1, index_Y]) / ((x_S * phi.iloc[0, index_Y]) + (x_H * phi.iloc[1, index_Y]))

                
               #print("T", x, ":")
               #print("Sad:" , x_S_Y)
               #print("Happy:", x_H_Y)

            # second or later observations    
            else:
                ### prediction
                x_S = (x_S_Y * theta[0][0]) + (x_H_Y * theta[1][0])
                x_H = 1.00 - x_S
            
                ### bayes update
                
                #observed y index
                index_Y = phi.columns.get_loc(y[x-1]) 
                
                x_S_Y = (x_S * phi.iloc[0, index_Y]) / ((x_S * phi.iloc[0, index_Y]) + (x_H * phi.iloc[1, index_Y])) 
                x_H_Y = (x_H * phi.iloc[1, index_Y]) / ((x_S * phi.iloc[0, index_Y]) + (x_H * phi.iloc[1, index_Y]))

                
                #print("T", x, ":")
                #print("Sad:" , x_S_Y)
                #print("Happy:", x_H_Y)
                
        else:
            x += 1
            
            ### prediction
            x_S_Y = (x_S_Y * theta[0][0]) + (x_H_Y * theta[1][0])
            x_H_Y = 1.00 - x_S
            
            #print("T", x, ":")
            #print("Sad:" , x_S_Y)
            #print("Happy:", x_H_Y)
            
    return 

HMM(px0 = px0, theta=theta, phi=phi, y= y, T=7)