import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

label = []
feature = [[],[],[]]
for i in range(1000):
    X1 = random.randint(0,100)
    feature[0].append(X1)
    
    X2 = random.randint(0,200)
    feature[1].append(X2)
    
    X3 = random.randint(0,300)
    feature[2].append(X3)
    
    Y = 5 * X1 + 3 * X2 + 1.5 * X3 + 6
    label.append(Y)

    data = {'feature1': feature[0], 'feature2': feature[1], 'feature3': feature[2], 'labels': label}  

df = pd.DataFrame(data)
standarized_df = df.copy()

standarized_df.feature1 = preprocessing.normalize([standarized_df.feature1])[0]
standarized_df.feature2 = preprocessing.normalize([standarized_df.feature2])[0]
standarized_df.feature3 = preprocessing.normalize([standarized_df.feature3])[0]

def plot(fig, ax, X, Y, xLabel, yLabel):
    ax.scatter(X, Y)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    plt.show()

#fig, ax = plt.subplots()
#plot(fig, ax, standarized_df.feature1, standarized_df.labels, "Feature 1", "label")

#fig, ax = plt.subplots()
#plot(fig, ax, standarized_df.feature2, standarized_df.labels, "Feature 2", "label")

#fig, ax = plt.subplots()
#plot(fig, ax, standarized_df.feature3, standarized_df.labels, "Feature 3", "label")




X_train, X_test, y_train, y_test = train_test_split(
    standarized_df.iloc[:,:-1], standarized_df.labels, test_size=0.33, random_state=42)



m = len(X_train)

def cost(X, y, weights):
    predictions = X.dot(weights)
    subtraction = np.subtract(predictions, y)
    squared = np.square(subtraction)
    cost = 1 / (2 * m) * np.sum(squared)
    return cost


def gradient_descent(X, y, weights, Alpha, iterations):
    CostPerIteration = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = X.dot(weights)
        subtraction = np.subtract(predictions, y)
        sum_delta = (Alpha / m) * X.transpose().dot(subtraction)
        weights = weights - sum_delta
        CostPerIteration[i] = cost(X, y, weights) 
        
    return weights, CostPerIteration

weights = np.zeros(3)
iterations = 10000
Alpha = 0.15
weights, CostPerIteration = gradient_descent(X_train, y_train, weights, Alpha, iterations)

fig, ax = plt.subplots()
plot(fig, ax, CostPerIteration, np.arange(iterations), "Cost per iteration", "iterations")

def predict(X):
    return X.dot(weights)

prediction = predict(X_test)
cost = cost(X_test, y_test, weights)

print("\nCost: ", cost)
print("\n weights:")
print(weights)