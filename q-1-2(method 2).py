#Part 2 (2nd method): Decision Tree for categorical as well as numerical data

import pandas
import numpy
import pprint
from numpy import log2 as log
from matplotlib import pyplot as plt

csv_path = raw_input("Enter path to input CSV file: ")
dataset = pandas.read_csv(csv_path)

#split data into train data and validation data
splitted = numpy.split(dataset, [int(.8 * len(dataset.index))])
data = splitted[0].reset_index()
validation_data = splitted[1].reset_index()

X = dataset.keys()[[0,1,2,3,4,5,7,8,9]]
Y = 'left'

vi = dataset.keys()[1]
filtered = dataset[(dataset[vi] >= 0) & (dataset[vi] < 380)]
yy = filtered[Y]
xx = filtered[vi]
#plt.scatter(xx, yy)
#plt.show()

x1 ,y1 = numpy.unique(filtered[filtered[Y] == 0][vi], return_counts=True)
x2 ,y2 = numpy.unique(filtered[filtered[Y] == 1][vi], return_counts=True)

plt.xlabel(vi)
plt.ylabel('frequency')
plt.title('Plot for Visualizing splitting points')
plt.scatter(x1, y1, label='left=0', color='green')
plt.scatter(x2, y2, label='left=1', color='red')
plt.legend()
plt.show()


def Range_set(att):
    #print att
    if att == 'satisfaction_level':
        return [[0,0.0,0.12],[1,0.12,0.36],[2,0.36,0.47],[3,0.47,0.92],[4,0.92,1.1]]

    if att == 'last_evaluation':
        return [[0,0.0,0.45],[1,0.45,0.58],[2,0.58,0.77],[3,0.77,1.1]]

    if att == 'number_project':
        return [[0,0,3],[1,3,6],[2,6,100]]

    if att == 'average_montly_hours':
        return [[0,0,127],[1,127,162],[2,162,217],[3,217,288],[4,288,1000]]

    if att == 'time_spend_company':
        return [[0,0,3],[1,3,4],[2,4,5],[3,5,7],[4,7,100]]


def is_categorical(att):
    if att in dataset.keys()[[5,6,7,8,9]]:
        return True
    else:
        return False


def choose_best_attribute(data, x, measure='entropy'):
    if len(x) == 1:
        return x[0]

    notLeft_count = len(data[data[Y] == 0])
    left_count = len(data[data[Y] == 1])
    impurity = 0.0
    if left_count == 0 or notLeft_count == 0:
        #in this case entropy_val = 0
        return 'noBest'
    else:
        q = float(left_count) / (left_count + notLeft_count)
        if measure == 'entropy':
            impurity = - ( q*log(q) + (1-q)*log(1-q) )
        if measure == 'gini':
            impurity = 4 * q * (1-q)
        if measure == 'misclassification':
            impurity = 2 * min(q, 1-q)

    max_info_gain = float(-99999999999)

    for att in x:
        impurity_split = 0.0

        if is_categorical(att):
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

                if measure == 'entropy':
                    impurity_split_val = - ( q*log(q) + (1-q)*log(1-q) )
                if measure == 'gini':
                    impurity_split_val = 4 * q * (1-q)
                if measure == 'misclassification':
                    impurity_split_val = 2 * min(q, 1-q)

                weight = float( len(subdata.index) ) / len(data.index)

                impurity_split += weight * impurity_split_val

        else:
            for Range in Range_set(att):
                subdata = data[(data[att] >= Range[1]) & (data[att] < Range[2])]
                notLeft_count_split = len(subdata[subdata[Y] == 0])
                left_count_split = len(subdata[subdata[Y] == 1])

                if left_count_split == 0 or notLeft_count_split == 0:
                    #in this case entropy_split_val = 0
                    continue
                else:
                    q = float(left_count_split) / (left_count_split + notLeft_count_split)

                if measure == 'entropy':
                    impurity_split_val = - ( q*log(q) + (1-q)*log(1-q) )
                if measure == 'gini':
                    impurity_split_val = 4 * q * (1-q) #scaled to [0,1]
                if measure == 'misclassification':
                    impurity_split_val = 2 * min(q, 1-q) #scaled to [0,1]

                weight = float( len(subdata.index) ) / len(data.index)

                impurity_split += weight * impurity_split_val

        #print entropy_split, att
        info_gain = impurity - impurity_split
        #print info_gain, att
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            bestAtt = att

    #print ""

    if max_info_gain <= 0:
        return 'noBest'

    #print ""
    #print max_info_gain, bestAtt
    #print ""
    return bestAtt


def check_purity(y, measure = 'entropy'):
    left_count = 0
    notLeft_count = 0

    for value in y:
        if value == 1:
            left_count = left_count + 1
        else:
            notLeft_count = notLeft_count + 1

    if notLeft_count == 0 and notLeft_count == 0:
        #in this case we return 1 as majority because we want high recall
        return 1, 1
    elif left_count == 0:
        return 1, 0
    elif notLeft_count == 0:
        return 1, 1

    impurity = 0.0
    q = float(left_count) / (left_count + notLeft_count)

    if measure == 'entropy':
        impurity = - ( q*log(q) + (1-q)*log(1-q) )
    if measure == 'gini':
        impurity = 4 * q * (1-q)
    if measure == 'misclassification':
        impurity = 2 * min(q, 1-q)

    purity = 1 - impurity
    #print purity

    if left_count > notLeft_count:
        return purity, 1

    else:
        return purity, 0


def build_decision_tree(data, x, measure = 'entropy', tree = None):

    #if pure enough
    purity, majority = check_purity(data[Y], measure)
    #print purity, majority
    if purity > 0.65: #gini:0.75, mis:0.85
        return majority

    #get an attribute with maximum information gain
    bestAtt = choose_best_attribute(data, x, measure)
    if bestAtt == 'noBest':
        return majority

    if tree == None:
        tree = {}
        tree[bestAtt] = {}

    if is_categorical(bestAtt):
        #for all categorical values keep growing tree
        attValue = numpy.unique(dataset[bestAtt])
        for value in attValue:
            subdata = data[data[bestAtt] == value]
            #if len(x) == 1:
            #   tree[bestAtt][value] = majority
            #else:
            tree[bestAtt][value] = build_decision_tree(subdata, x.drop(bestAtt), measure) #Calling the function recursively

    else:
        #divide the numerical data in specified range(s)
        #print Range_set(bestAtt)
        for Range in Range_set(bestAtt):
            subdata = data[(data[bestAtt] >= Range[1]) & (data[bestAtt] < Range[2])]
            #if len(x) == 1:
            #    tree[bestAtt][Range[0]] = majority
            #else:
            tree[bestAtt][Range[0]] = build_decision_tree(subdata, x.drop(bestAtt), measure)

    return tree

#print type(X)
decision_tree = build_decision_tree(data, X)
#pprint.pprint(decision_tree)

def predict(inst,tree):
    for nodes in tree.keys():
        value = inst[nodes]
        value = value.tolist()
        value = value[0]
        #print nodes
        if not is_categorical(nodes):
            ranges = Range_set(nodes)
            for i in range(len(ranges)):
                if value >= ranges[i][1] and value < ranges[i][2]:
                    value = i
        #print nodes, value
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;

    return prediction


def calculate_performance(validation_data):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(validation_data.index)):
        row = validation_data.iloc[[i]][Y]
        row = row.tolist()
        row = row[0]
        if predict(validation_data.iloc[[i]], decision_tree) == 1:
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

    print ("accuracy =", accuracy)
    print ("precision =", precision)
    print ("recall =", recall)
    print ("F1 measure =", F1measure)

calculate_performance(validation_data)


def predict_test(test_set, tree):
    print ("\nPredictions:")
    for i in range(len(test_set.index)):
        print (predict(test_set.iloc[[i]], tree))

do_test = raw_input("\nProvide test data? (y/n): ")

if do_test == 'y' or do_test == 'Y':
    csv_path = raw_input("Enter path to test CSV file: ")
    test_set = pandas.read_csv(csv_path)

    predict_test(test_set, decision_tree)
