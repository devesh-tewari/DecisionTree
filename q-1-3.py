#Part 3: Effectiveness of Misclassification rate, Gini, Entropy as impurity measures

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
X = dataset.keys()[[0,1,2,3,4,5,7,8,9]]
Y = 'left'

#The function is_categorical() takes an attribute name as input and returns true if the 
#atrribute is categorical, otherwise returns false.
def is_categorical(att):
    if att in dataset.keys()[[5,6,7,8,9]]:
        return True
    else:
        return False

    
#The function Range_set() is used to decide the best spliiting point in a numeric attribute.
#It calculates the information gain for all unique values of an attribue in a dataset and 
#tries to split the node in terms of values greater than this value and less than this value.
#Then it returns the splitting value at which information gain is maximum.
def Range_set(att, data, measure = 'entropy'):
    
    notLeft_count = len(data[data[Y] == 0])
    left_count = len(data[data[Y] == 1])
    impurity = 0.0
    
    q = float(left_count) / (left_count + notLeft_count)
    if measure == 'entropy':
        impurity = - ( q*log(q) + (1-q)*log(1-q) )
    elif measure == 'gini':
        impurity = 4 * q * (1-q) #scaled to [0,1]
    elif measure == 'misclassification':
        impurity = 2 * min(q, 1-q) #scaled to [0,1]
            
    attValue = numpy.unique(data[att])
    attValue.sort()
    max_info_gain = 0.0
    max_at = attValue[0] + 0.0005
    i = 0
    while i < len(attValue) - 1:
        impurity_split = 0.0
        
        mid = float( attValue[i] + attValue[i+1] ) / 2
        i += 1
        left_data = data[data[att] < mid]
        right_data = data[data[att] >= mid]
        
        notLeft_count_split = len(left_data[left_data[Y] == 0])
        left_count_split = len(left_data[left_data[Y] == 1])

        if left_count_split == 0 or notLeft_count_split == 0:
            #in this case entropy_split_val = 0
            continue
        else:
            q = float(left_count_split) / (left_count_split + notLeft_count_split)

        if measure == 'entropy':
            impurity_split_val = - ( q*log(q) + (1-q)*log(1-q) )
        elif measure == 'gini':
            impurity_split_val = 4 * q * (1-q) #scaled to [0,1]
        elif measure == 'misclassification':
            impurity_split_val = 2 * min(q, 1-q) #scaled to [0,1]

        weight = float( len(left_data.index) ) / len(data.index)

        impurity_split += weight * impurity_split_val
        
        
        notLeft_count_split = len(right_data[right_data[Y] == 0])
        left_count_split = len(right_data[right_data[Y] == 1])

        if left_count_split == 0 or notLeft_count_split == 0:
            #in this case entropy_split_val = 0
            continue
        else:
            q = float(left_count_split) / (left_count_split + notLeft_count_split)

        if measure == 'entropy':
            impurity_split_val = - ( q*log(q) + (1-q)*log(1-q) )
        elif measure == 'gini':
            impurity_split_val = 4 * q * (1-q) #scaled to [0,1]
        elif measure == 'misclassification':
            impurity_split_val = 2 * min(q, 1-q) #scaled to [0,1]

        weight = float( len(right_data.index) ) / len(data.index)

        impurity_split += weight * impurity_split_val
        
        info_gain = impurity - impurity_split
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_at = mid
    
    return [ str(attValue[0]) + '~' + str(max_at),  str(max_at) + '~' + str(attValue[len(attValue)-1]) ]


#The choose_best_attribute() function can handle numeric atrributes.
#If numerical data is chosen as best attribute, it will now return a list of ranges 
#to split the data.
def choose_best_attribute(data, x, measure='entropy'):
    Ranges = []
    ranges = []
    if len(x) == 1:
        return x[0], Ranges
    
    notLeft_count = len(data[data[Y] == 0])
    left_count = len(data[data[Y] == 1])
    impurity = 0.0
    if left_count == 0 or notLeft_count == 0:
        #in this case entropy_val = 0
        return 'noBest', Ranges
    else:
        q = float(left_count) / (left_count + notLeft_count)
        if measure == 'entropy':
            impurity = - ( q*log(q) + (1-q)*log(1-q) )
        elif measure == 'gini':
            impurity = 4 * q * (1-q)
        elif measure == 'misclassification':
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
                elif measure == 'gini':
                    impurity_split_val = 4 * q * (1-q)
                elif measure == 'misclassification':
                    impurity_split_val = 2 * min(q, 1-q)
                    
                weight = float( len(subdata.index) ) / len(data.index)
            
                impurity_split += weight * impurity_split_val
        
        else:
            ranges = Range_set(att, data, measure)
            Range = ranges[0].split('~')
            subdata = data[(data[att] >= float(Range[0])) & (data[att] < float(Range[1]))]
            notLeft_count_split = len(subdata[subdata[Y] == 0])
            left_count_split = len(subdata[subdata[Y] == 1])

            if left_count_split == 0 or notLeft_count_split == 0:
                #in this case entropy_split_val = 0
                continue
            else:
                q = float(left_count_split) / (left_count_split + notLeft_count_split)

            if measure == 'entropy':
                impurity_split_val = - ( q*log(q) + (1-q)*log(1-q) )
            elif measure == 'gini':
                impurity_split_val = 4 * q * (1-q) #scaled to [0,1]
            elif measure == 'misclassification':
                impurity_split_val = 2 * min(q, 1-q) #scaled to [0,1]

            weight = float( len(subdata.index) ) / len(data.index)

            impurity_split += weight * impurity_split_val

            Range = ranges[1].split('~')
            subdata = data[(data[att] >= float(Range[0])) & (data[att] <= float(Range[1]))]
            notLeft_count_split = len(subdata[subdata[Y] == 0])
            left_count_split = len(subdata[subdata[Y] == 1])

            if left_count_split == 0 or notLeft_count_split == 0:
                #in this case entropy_split_val = 0
                continue
            else:
                q = float(left_count_split) / (left_count_split + notLeft_count_split)

            if measure == 'entropy':
                impurity_split_val = - ( q*log(q) + (1-q)*log(1-q) )
            elif measure == 'gini':
                impurity_split_val = 4 * q * (1-q) #scaled to [0,1]
            elif measure == 'misclassification':
                impurity_split_val = 2 * min(q, 1-q) #scaled to [0,1]

            weight = float( len(subdata.index) ) / len(data.index)

            impurity_split += weight * impurity_split_val
        
        info_gain = impurity - impurity_split
        
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            bestAtt = att
            Ranges = ranges
    
    if max_info_gain <= 0.0001:
        return 'noBest', Ranges
    
    return bestAtt, Ranges


def check_purity(y, measure = 'entropy'):
    left_count = 0
    notLeft_count = 0
    
    for value in y:
        if value == 1:
            left_count = left_count + 1
        else:
            notLeft_count = notLeft_count + 1
            
    if left_count + notLeft_count < 5:
        if left_count > notLeft_count:
            return 1, 1
        else:
            return 1, 0

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
    
    
#If the best attribute chosen is numeric,
#Then it will be called recursively for the range below the best split and above the best split.
#For numerical data, insted of string a value in the node of decision tree, we store the range 
#in the format "start~end" where the range is [start,end]
def build_decision_tree(data, x, measure = 'entropy', tree = None): 
    
    #if pure enough
    purity, majority = check_purity(data[Y], measure)
    
    if measure == 'entropy' and purity > 0.7:
        return majority
    if measure == 'gini' and purity > 0.75:
        return majority
    if measure == 'misclassification' and purity > 0.97:
        return majority

    #get an attribute with maximum information gain
    bestAtt, Range_str = choose_best_attribute(data, x, measure)
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
            tree[bestAtt][value] = build_decision_tree(subdata, x.drop(bestAtt), measure) #Calling the function recursively 
    
    else:
        #divide the numerical data in specified range(s)
        Range = Range_str[0].split('~')
        subdata = data[(data[bestAtt] >= float(Range[0])) & (data[bestAtt] < float(Range[1]))]
        
        tree[bestAtt][Range_str[0]] = build_decision_tree(subdata, x, measure)
        
        Range = Range_str[1].split('~')
        subdata = data[(data[bestAtt] >= float(Range[0])) & (data[bestAtt] <= float(Range[1]))]
        
        tree[bestAtt][Range_str[1]] = build_decision_tree(subdata, x, measure)
            
    return tree



#If the attribute is numeric, in the predict() function, we need to check if the row value lies between the range.
def predict(row, tree):
    for nodes in tree.keys():
        value = row[nodes]
        #print value
        value = value.tolist()
        value = value[0]
        #print nodes
        if not is_categorical(nodes):
            found = False
            for Range_str in tree[nodes].keys():
                #print Range_str
                Range = Range_str.split('~')
                if value >= float(Range[0]) and value <= float(Range[1]):
                    value = Range_str
                    found = True
            
            if not found:
                Range_str = tree[nodes].keys()[0]
                if value < float(Range[0]):
                    value = Range_str
                
                Range_str = tree[nodes].keys()[len(tree[nodes].keys())-1]
                if value > float(Range[1]):
                    value = Range_str
                    
        #print nodes, value
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(row, tree)
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
        
    print ("Validation Results:\n")
    print ("accuracy = " + str(accuracy))
    print ("precision = " + str(precision))
    print ("recall = " + str(recall))
    print ("F1 measure = " + str(F1measure))


decision_tree = build_decision_tree(data, X, 'gini')
print ("\nGINI INDEX:\n")
calculate_performance(validation_data)

decision_tree = build_decision_tree(data, X, 'entropy')
print ("\n\nENTROPY:\n")
calculate_performance(validation_data)

decision_tree = build_decision_tree(data, X, 'misclassification')
print ("\n\nMISCLASSIFICATION RATE:\n")
calculate_performance(validation_data)