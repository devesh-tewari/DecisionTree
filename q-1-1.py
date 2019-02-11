#Part 1: Decision Tree for categorical data

import pandas
import numpy
import pprint
from numpy import log2 as log

csv_path = raw_input("Enter path to input CSV file: ")
dataset = pandas.read_csv(csv_path)

#split data into train data and validation data
a = numpy.split(dataset, [int(.8 * len(dataset.index))])
data = a[0].reset_index()
validation_data = a[1].reset_index()

#We inititalize X and Y as the attributes and label respectively for the decision tree
X = dataset.keys()[[5,7,8,9]]
Y = 'left'

#The following function 'choose_best_attribute()' takes a set of data as input and
#returns the best attribute which has maximum information gain. The calculations are
#based on entropy.
def choose_best_attribute(data, x):

    if len(x) == 1:
        return x[0]

    notLeft_count = len(data[data[Y] == 0])
    left_count = len(data[data[Y] == 1])
    entropy = 0.0

    if left_count == 0 or notLeft_count == 0:
        #in this case entropy_val = 0
        pass
    else:
        q = float(left_count) / (left_count + notLeft_count)
        entropy = - ( q*log(q) + (1-q)*log(1-q) )

    max_info_gain = float(-99999999999)

    for att in x:
        entropy_split = 0.0
        attValue = numpy.unique(data[att])

        for value in attValue:
            subdata = data[data[att] == value]
            notLeft_count_split = len(subdata[subdata[Y] == 0])
            left_count_split = len(subdata[subdata[Y] == 1])

            if left_count_split == 0 or notLeft_count_split == 0:
                #in this case entropy_split_val = 0
                continue
            else:
                q = float(left_count_split) / (left_count_split + notLeft_count_split)

            entropy_split_val = - ( q*log(q) + (1-q)*log(1-q) )
            weight = float( len(subdata.index) ) / len(data.index)

            entropy_split += weight * entropy_split_val

        info_gain = entropy - entropy_split

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            bestAtt = att

    if max_info_gain <= 0.0001:  #threshold for information gain
        return 'noBest'

    return bestAtt

#This function 'check_purity()' is used to calculate the purity of labels in a dataset.
#Along with the purity, it also returns the majority label value.
def check_purity(y):
    left_count = 0
    notLeft_count = 0

    for value in y:
        if value == 1:
            left_count = left_count + 1
        else:
            notLeft_count = notLeft_count + 1

    if left_count == 0:
        return 1, 0
    elif notLeft_count == 0:
        return 1, 1

    impurity = 0.0
    q = float(left_count) / (left_count + notLeft_count)

    impurity = - ( q*log(q) + (1-q)*log(1-q) )

    purity = 1 - impurity

    if left_count > notLeft_count:
        return purity, 1
    else:
        return purity, 0


#The function below actually builds the decision tree recursively using dictionary
#keys as nodes. It stops when purity of a node increases some threshold or when there
#are no attributes left with good information gain. It makes sense to drop an atrribute
#once it has been used in a branch, so attributes occur only once in a branch.
def build_decision_tree(data, x, tree = None):

    #if pure enough
    purity, majority = check_purity(data[Y])

    if purity > 0.65:
        return majority

    #get an attribute with maximum information gain
    bestAtt = choose_best_attribute(data, x)
    if bestAtt == 'noBest':
        return majority

    if tree == None:
        tree = {}
        tree[bestAtt] = {}

    #take distinct values of that attribute
    attValue = numpy.unique(dataset[bestAtt])

    #for all categorical values keep groing tree
    for value in attValue:
        subdata = data[data[bestAtt] == value]

        tree[bestAtt][value] = build_decision_tree(subdata, x.drop(bestAtt)) #Calling the function recursively

    return tree


decision_tree = build_decision_tree(data, X)
pprint.pprint(decision_tree)


#We now calculate the performance measures for the decision tree using the validation data.
#In our case we want high recall because we do not want an employee to leave the company
#when we have no clue that he/she would leave.
def predict(row, tree):

    for nodes in tree.keys():
        value = row[nodes]
        value = value.tolist()
        value = value[0]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(row, tree)
        else:
            prediction = tree
            break

    return prediction


def calculate_performance(validation_data, tree):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(validation_data.index)):
        row = validation_data.iloc[[i]][Y]
        row = row.tolist()
        row = row[0]
        if predict(validation_data.iloc[[i]], tree) == 1:
            if row == 1:
                TP += 1
            else:
                FP += 1
        else:
            if row == 0:
                TN += 1
            else:
                FN += 1

    accuracy = float(TP + TN) / (TP + TN + FP + FN)

    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP) / (TP + FN)

    F1measure = 2.0 / ( (1/recall) + (1/precision) )

    print ("Validation Results:\n")
    print ("accuracy = " + str(accuracy))
    print ("precision = " + str(precision))
    print ("recall = " + str(recall))
    print ("F1 measure = " + str(F1measure))


calculate_performance(validation_data, decision_tree)


def predict_test(test_set, tree):
    print ("\nPredictions:")
    for i in range(len(test_set.index)):
        print (predict(test_set.iloc[[i]], tree))

do_test = raw_input("\nProvide test data? (y/n): ")

if do_test == 'y' or do_test == 'Y':
    csv_path = raw_input("Enter path to test CSV file: ")
    test_set = pandas.read_csv(csv_path)

    predict_test(test_set, decision_tree)
