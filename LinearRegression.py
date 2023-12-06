import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data():
    df = pd.read_csv('Data/Housing.csv')
    data, target = np.array(df[['area']], 'f'), np.array(df[['price']], 'f')

    # Normalizing data
    for i in range(data.shape[1]):
        data[:, i] = ((data[:, i] - np.mean(data[:, i])) / np.std(data[:, i]))
    return data, target


def h(x, theta):
    return np.matmul(x, theta)


def cost_func(y, x, theta):
    m = len(y)
    return (1 / (2 * m)) * (h(x, theta) - y).T @ (h(x, theta) - y)


def gradient_descent(x, theta, num_iter, y, alpha):
    costs = []
    m = len(x)
    for i in range(num_iter):
        gradient = (1 / m) * x.T @ (h(x, theta) - y)
        costs.append(cost_func(y, x, theta)[0])
        theta = theta - alpha * gradient
    return theta, costs


def linear_regression():
    x, y = read_data()
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    theta = np.zeros((x.shape[1], 1))
    alpha = 0.1
    num_iter = 100
    theta_opt, costs = gradient_descent(x, theta, num_iter, y, alpha)
    plt.plot(x[:, 1], h(x, theta_opt), label='Regression Fit', color='orange')
    plt.scatter(x[:, 1], y, label='Data')
    plt.legend()
    # plt.plot(list(range(1, num_iter + 1)), costs)
    plt.show()


linear_regression()
