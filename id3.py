#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
from math import log

import node


train=None
varnames=None
test=None
testvarnames=None
root=None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):

    if p == 0 or p == 1:
        return 0
    p1 = 1 - p
    s = - p * log(p, 2) - p1 * log(p1, 2)

    return s
	
	
# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    p = py / (total * 1.0)
    s = entropy(p)

    if pxi == 0:
        p2 = (py - py_pxi) / ((total - pxi) * 1.0)
        s2 = entropy(p2)
        gain = s - ((total - pxi) / (total * 1.0)) * s2
        return gain

    p1 = py_pxi / (pxi * 1.0)
    s1 = entropy(p1)
    if pxi == total:
        gain = s - (pxi / (total * 1.0)) * s1
    else:
        p2 = (py - py_pxi) / ((total - pxi) * 1.0)
        s2 = entropy(p2)
        gain = s - (pxi / (total * 1.0)) * s1 - ((total - pxi) / (total * 1.0)) * s2


    return gain

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable	
	
	
	
# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
		data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, model_file):
    f = open(model_file, 'w+')
    root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):

    #py_pxi, pxi, py, total
    py_pxi = 0
    pxi = 0
    py = 0
    total = len(data)
    for i in data:
        if i[len(data[0])-1] == 1:
            py += 1

    guess = py / (total * 1.0)
    if guess == 1:
        return node.Leaf(varnames, 1)
    elif guess == 0:
        return node.Leaf(varnames, 0)

    if len(varnames) == 1:
        if guess > 0.5 :
            return node.Leaf(varnames, 1)
        else:
            return node.Leaf(varnames, 0)

    gain = 0;

    for i in range(len(varnames) - 1):
        for j in data:
            if j[i] == 1:
                pxi += 1
            if j[i] == 1 and j[-1] == 1:
                py_pxi += 1
        if infogain(py_pxi, pxi, py, total) > gain :
            gain = infogain(py_pxi, pxi, py, total)
            index = i
        py_pxi = 0
        pxi = 0

    if gain == 0:
        if guess > 0.5:
            return node.Leaf(varnames, 1)
        else:
            return node.Leaf(varnames, 0)

    # divide the data
    data0 = []
    data1 = []

    for i in range(len(data)):
        if data[i][index] == 0:
            list = data[i]
            del list[index]
            data0.append(list)
        else:
            list = data[i]
            del list[index]
            data1.append(list)

    # delete root from the varnames and data
    new_varnames = []
    for i in range(len(varnames)):
        if i != index:
            new_varnames.append(varnames[i])

    return node.Split(varnames, index, build_tree(data0, new_varnames), build_tree(data1, new_varnames))


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
	global train
	global varnames
	global test
	global testvarnames
	global root
	(train, varnames) = read_data(trainS)
	(test, testvarnames) = read_data(testS)
	modelfile = modelS
	#print train, '\n\n\n\n\n', varnames

	# build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
	root = build_tree(train, varnames)

	print_model(root, modelfile)
	
def runTest():
	correct = 0
	# The position of the class label is the last element in the list.
	yi = len(test[0]) - 1
	for x in test:
		# Classification is done recursively by the node class.
        # This should work as-is.
		pred = root.classify(x)
		if pred == x[yi]:
			correct += 1
	acc = float(correct)/len(test)
	return acc	
	
	
# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
		print 'Usage: id3.py <train> <test> <model>'
		sys.exit(2)
    loadAndTrain(argv[0],argv[1],argv[2]) 
                    
    acc = runTest()             
    print "Accuracy: ",acc                      

if __name__ == "__main__":
    main(sys.argv[1:])
