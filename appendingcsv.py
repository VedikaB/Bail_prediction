# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:25:45 2019

@author: vedika barde
"""
import csv
row = ['384','CPC','438','MALE','0','0','0','1','1','0','0','VERY SERIOUS','0','0','STABLE','0','1','0','0','0','Refused']

with open('C:/Users/vedika barde/Desktop/BE_pro/Bailfinal.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)