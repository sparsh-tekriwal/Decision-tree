#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:25:07 2020

@author: sparshtekriwal
"""

import sys
import csv
import numpy as np


def inspection(train_data, metrics_out):



    labels = train_data[:,-1]

    value_counts = {}

    for x in labels:
        if x in value_counts.keys():
            value_counts[x]+=1
        else:
            value_counts[x]=1

    value_counts= sorted(value_counts.items(), key= lambda x: x[1], reverse=True)


    gini_impurity = 1
    for x in value_counts:
        gini_impurity = gini_impurity - (x[1]/len(labels))**2
    error = ( len(labels) - value_counts[0][1] )/len(labels)

    with open(metrics_out, 'w') as f:
        f.write("gini_impurity: %s\nerror: %s" %(gini_impurity, error))



if __name__ == '__main__':

    input = sys.argv[1]
    output = sys.argv[2]
    train_data_as_list  = []

    with open(input) as tsvfile:
        reader = csv.reader(tsvfile, delimiter= '\t')
        for row in reader:
            train_data_as_list.append(row)
    train_data=np.array(train_data_as_list[1:])

    inspection(train_data, output)



