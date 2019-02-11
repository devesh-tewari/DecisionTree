#Part 4: Visualizing two attributes of training data on a 2D plot taking

import pandas

csv_path = raw_input("Enter path to csv file: ")
dataset = pandas.read_csv(csv_path)

import matplotlib.pyplot as plt

att1 = 'satisfaction_level'
att2 = 'last_evaluation'
Y = 'left'

plt.figure(figsize=(15,15))
x = dataset[dataset[Y] == 1][att1]
y = dataset[dataset[Y] == 1][att2]

plt.scatter(x, y, label = 'left=1', color = 'red', s = 30)

x = dataset[dataset[Y] == 0][att1]
y = dataset[dataset[Y] == 0][att2]
plt.scatter(x, y, label = 'left=0', color = 'green', s = 10)

plt.xlabel(att1) 
plt.ylabel(att2)
plt.title('Plot for two major attributes') 
plt.legend()
plt.show()