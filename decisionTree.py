#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:00:54 2020

@author: sparshtekriwal
"""
# imports
import sys
import csv
import numpy as np

# reads data with attribute names
def read_file(file_name):

    data_as_list = []

    with open(file_name) as tsvfile:
        reader = csv.reader(tsvfile, delimiter= '\t')
        for row in reader:
            data_as_list.append(row)

    return np.array(data_as_list)


# accepts data without attribute names
def calculate_gini_impurity(data):

    labels = data[:,-1]
    value_counts = {}

    for x in labels:
        if x in value_counts.keys():
            value_counts[x]+=1
        else:
            value_counts[x]=1

    #this is a list of tuples
    sorted_value_counts= sorted(value_counts.items(), key= lambda x: x[1], reverse=True)

    gini_impurity = 1

    for x in sorted_value_counts:
        gini_impurity = gini_impurity - (x[1]/len(labels))**2


    return gini_impurity


# accepts data without attributes
def calculate_gini_gain(data, split_index):
    gini_root = calculate_gini_impurity(data)
    unique_values = set(data[:, split_index])
    gini_subroot = 0

    for x in unique_values:
        subset = data[data[:,split_index]==x]
        prob_subset=subset.shape[0]/data.shape[0]
        gini_subroot += prob_subset * calculate_gini_impurity(subset)
    gini_gain = gini_root - gini_subroot
    return gini_gain


#accepts only data without attributes
def find_max_gini_gain_index(data):

    index_with_max_gini_gain = -1
    max_gain = 0

    for x in range(data.shape[1]-1):
        gini_gain = calculate_gini_gain(data, x)
        if gini_gain > max_gain:
            max_gain = gini_gain
            index_with_max_gini_gain = x
    return index_with_max_gini_gain


# accepts data without attributes and returns true or false
def is_pure(data):
    if (len( set( data[:,-1] )) ==1 ):
        return True
    return False


# accepts data without attributes and returns majority
def majority_vote(data):
    uniques_values, value_counts = np.unique(data[:,-1], return_counts=True)
    if(len(value_counts)==2 and (value_counts[0]==value_counts[1])):
        return uniques_values[1]
    return uniques_values[np.argmax(value_counts)]


## partition and drop column # accepts only data and not attributes , returns values as well
def partition(data, attributes, split_index):
    partitions=[]
    values = []
    for x in np.unique(data[:,split_index]):
        partitions.append( np.vstack( (attributes, data[data[:,split_index]==x])  )  )
        values.append (x)
    return np.delete(partitions[0],split_index,1), np.delete(partitions[1],split_index,1), values[0], values[1]


# To print as formated in tree
def print_class_counts(node, unique_labels):
    data = node.data

    print("[" + str(np.sum(data[:,-1]==unique_labels[0])) + " " +unique_labels[0] + " /" +
            str(np.sum(data[:,-1]==unique_labels[1])) + " " + unique_labels[1] + "]")

class TreeNode:

    def __init__(self, data):
        self.leftNode = None
        self.rightNode = None
        self.data = data[1:]
        self.attributes = data[0]
        self.depth = 0
        self.best_split_index = None
        self.unique_labels = self.calculate_unique_labels()
        self.val = majority_vote(self.data)
        self.rootNodeAttributes = data[0]
#        self.indexThatThisChildWasSplitOn = None
        self.valueThatThisChildHadDuringSplit = None

    def split(self):
        self.best_split_index = find_max_gini_gain_index(self.data)
        if (self.best_split_index != -1):
            self.leftNode = TreeNode( partition(self.data, self.attributes, self.best_split_index)[0] )
            self.rightNode = TreeNode( partition(self.data, self.attributes, self.best_split_index)[1] )

            #print(self.rightNode.data.shape, self.leftNode.data.shape)
            self.leftNode.valueThatThisChildHadDuringSplit = partition(self.data, self.attributes, self.best_split_index)[2]
            self.rightNode.valueThatThisChildHadDuringSplit = partition(self.data, self.attributes, self.best_split_index)[3]

            self.leftNode.depth = self.depth + 1
            self.rightNode.depth = self.depth + 1

            self.leftNode.rootNodeAttributes = self.rootNodeAttributes
            self.rightNode.rootNodeAttributes = self.rootNodeAttributes

#            self.leftNode.indexThatThisChildWasSplitOn = self.best_split_index
#            self.rightNode.indexThatThisChildWasSplitOn = self.best_split_index


    def calculate_unique_labels(self):
        return np.unique(self.data[:,-1])

#####################


def decisionTree(node: TreeNode , max_depth):
    #For Printing Initial Class counts of tree. For graphical purpose
    if(node.depth==0):
        print_class_counts(node, node.unique_labels)

    if( (is_pure(node.data) == False) and (node.depth < max_depth) ):
        node.split()
        if (node.best_split_index== -1):
            return

        print(("| " * node.leftNode.depth) + node.attributes[node.best_split_index] + " = " +
              node.leftNode.valueThatThisChildHadDuringSplit + ": " , end =" ")
        print_class_counts(node.leftNode, node.unique_labels)
#        print(node.best_split_index)

        decisionTree(node.leftNode, max_depth)

        print(("| " * node.rightNode.depth) + node.attributes[node.best_split_index] + " = " +
              node.rightNode.valueThatThisChildHadDuringSplit + ": " , end =" ")
        print_class_counts(node.rightNode, node.unique_labels)

        decisionTree(node.rightNode, max_depth)

    return


def classify(node, row):

    if(node.leftNode == None or node.rightNode == None):
        return node.val

    elif (row[node.best_split_index] == node.leftNode.valueThatThisChildHadDuringSplit) :
#        print(node.leftNode.valueThatThisChildHadDuringSplit)
        row = np.delete(row,node.best_split_index)
        return classify(node.leftNode, row)

    else:
        row = np.delete(row,node.best_split_index)
        return classify(node.rightNode, row)
    return node.val





def calculate_error(true_labels, predicted_labels):
    error_count=0
    for i in range(len(true_labels)):
        if(true_labels[i]!=predicted_labels[i]):
            error_count+=1
    return error_count/len(true_labels)






if __name__ == '__main__':

    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    train_data = read_file(train_input)
    test_data = read_file(test_input)

    node1 = TreeNode(train_data)
    decisionTree(node1, max_depth)

    train_labels = []
    for row in train_data[1:]:
        train_labels.append(classify(node1, row))

#    print(train_labels)
    test_labels = []
    for row in test_data[1:]:
        test_labels.append(classify(node1, row))

    train_error = calculate_error(train_data[1:,-1],train_labels)
    test_error = calculate_error(test_data[1:,-1],test_labels)



    with open(train_out, 'w') as f:
        for item in train_labels:
            f.write("%s\n" % item)

    with open(test_out, 'w') as f:
        for item in test_labels:
            f.write("%s\n" % item)

    with open(metrics_out, 'w') as f:
        f.write("error(train): %s\nerror(test): %s" %(train_error, test_error))












