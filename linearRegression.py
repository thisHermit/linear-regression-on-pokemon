"""
An Example of a Linear Regression model.

Here i am taking an example from https://www.kaggle.com/alopez247/pokemon
to find a relation between variable "Total" and "HP".

"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import os

data = pd.read_csv("./pokemon_alopez247.csv")
d = {"Total": data['Total'],
     "HP": data['HP']}
smallData = pd.DataFrame(d)
test = smallData.values
epsilon = 0.001
print(data)


def compute_error_for_line(b, m, points):
    """Return the Error for Line given the points."""
    totalError = 0
    for i in range(0, len(points)):
        x = test[i, 0]
        y = test[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    """Return the new b and m points."""
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        error = y - ((m_current * x) + b_current)
        b_gradient += -(2 / N) * error
        m_gradient += -(2 / N) * x * error
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def main():
    """Return and plot function here."""
    plt.figure(num=None, figsize=(20, 10), dpi=80,
               facecolor='w', edgecolor='k')
    plt.axis([0, 260, 0, 780])
    plt.ylabel("Total")
    plt.xlabel("HP")
    plt.scatter(test[:, [0]], test[:, [1]], c='r', s=1)

    m = 0.3
    b = -30
    x = np.arange(800)
    y = m * x + b
    for i in range(30):
        error = compute_error_for_line(b, m, test)
        print("error :", error)
        if(error > epsilon):
            y = m * x + b
            plt.plot(x, y)
            b, m = step_gradient(b, m, test, 0.0001)
            print("b , m :", b, ",", m)
            plt.pause(0.01)

    plt.show()

    plt.pause(0.001)
#    for i in range(200):
#        y = np.random.random() * 200
#        plt.scatter(i, y)
#        plt.pause(0.0001)
#    while True:
#        plt.pause(0.0001)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
