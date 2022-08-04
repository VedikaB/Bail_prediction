# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:50:39 2019

@author: vedika barde
"""

import numpy as np
import matplotlib.pyplot as mp
def estimate_coefficient(x,y):
    n=np.size(x)
    mean_x,mean_y=np.mean(x),np.mean(y)
    
    ss_xy=np.sum(y*x)-n*mean_x*mean_y
    ss_xx=np.sum(x*x)-n*mean_x*mean_x
    
    b_1=(ss_xy)/(ss_xx)
    b_0=(mean_y)-(b_1*mean_x)
    return(b_0,b_1)
    
def plot_line(x,y,b):
    mp.scatter(x,y,color="m",marker="o",s=30)
    y_pred=b[1]*x+b[0]
    mp.plot(x,y_pred,color="g")
    mp.show()
    
def main():
    x=np.array([10,9,2,15,10,16,11,16])
    y=np.array([95,80,10,50,45,98,38,93])
    b=estimate_coefficient(x,y)
    print("Estimated values of coefficient\nb_0:{}\nb_1:{}".format(b[0],b[1]))
    plot_line(x,y,b)
if __name__ == "__main__": 
    main() 