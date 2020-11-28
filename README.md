Programming a Decision Tree 

The program learns a decision tree with a specified maximum depth, prints the decision tree in a specified format, predicts the labels of the training and testing examples, and calculates training and testing errors.

The tree uses Gini index as the splitting criteria.

Execution - Type on command line - python decisionTree.py [args...]

Where above [args...] is a placeholder for six command-line arguments : train_input test_input max_depth train_out test_out metrics_out. These arguments are described below:

train input : path to the training input .tsv file
test input : path to the test input .tsv file
max depth : maximum depth to which the tree will be built
train out : path of output .labels file to which the predictions on the training data will be written
test out : path of output .labels file to which the predictions on the test data will be written
metrics out : path of the output .txt file to which metrics such as train and test error will be written
eg - python decisionTree.py train.tsv test.tsv 2 train.labels test.labels metrics.txt

Output -

[15 D /13 R] | feature_1 = y: [13 D /1 R] | | feature_2 = y: [13 D /0 R] | feature_2 = n: [0 D /1 R] | feature_1 = n: [2 D /12 R] | | feature_2 = y: [2 D /7 R] | feature_2 = n: [0 D /5 R]
