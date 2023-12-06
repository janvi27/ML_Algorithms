import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
pd.options.mode.chained_assignment = None


def read_data():
    df = pd.read_csv('Data/Invistico_Airline.csv')
    data, y = df.drop(columns=['satisfaction']), df[['satisfaction']]
    return data, y


def clean_data(x, y):
    y.loc[y['satisfaction'] == 'satisfied', 'satisfaction'] = 1
    y.loc[y['satisfaction'] == 'dissatisfied', 'satisfaction'] = 0

    for col in x.select_dtypes(include=['float64', 'int64']):
        x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())

    for col in x.select_dtypes(include='object'):
        unique = x[col].unique()
        encode = {k: v for k, v in zip(unique, range(len(unique)))}
        x.replace({col: encode}, inplace=True)
    x.fillna(x.median(), inplace=True)
    return np.array(x, 'f'), np.array(y, 'int')


def train_test_split(x, y):
    mask = np.ones((len(x), 1), dtype=bool)
    rand = sample(range(len(x)), int(0.7 * len(x)))
    mask[rand] = False

    X_train, y_train, X_test, y_test = x[~mask], y[~mask], x[mask], y[mask]
    return X_train, y_train, X_test, y_test


def h(x, theta):
    return 1 / (1 + np.exp(- np.matmul(x, theta)))


def cost_function(x, theta, y):
    m = x.shape[0]
    return (-1 / m) * (y * np.log(h(x, theta)) + (1 - y) * np.log(1 - h(x, theta)))


def gradient_descent(x, theta, y, num_iter, alpha):
    costs = []
    m = x.shape[0]
    for i in range(num_iter):
        h_x = h(x, theta)
        gradient = (1 / m) * x.T @ (h_x - y)
        costs.append(cost_function(x, theta, y)[0])
        theta = theta - alpha * gradient
    return theta, costs


def metric(y, y_pred):
    return np.sum(y == y_pred) / len(y)


def logistic_regression():
    x, y = read_data()
    X_train, y_train, X_test, y_test = train_test_split(x, y)
    X_train, y_train = clean_data(X_train, y_train)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    theta = np.zeros((X_train.shape[1], 1))
    num_iter = 5000
    alpha = 0.01

    theta_opt, costs = gradient_descent(X_train, theta, y_train, num_iter, alpha)
    y_pred = h(X_train, theta_opt)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    print('Training accuracy: ', metric(y_train, y_pred))

    X_test, y_test = clean_data(X_test, y_test)
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    y_pred = h(X_test, theta_opt)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    print('Test accuracy: ', metric(y_test, y_pred))
    plt.plot(list(range(num_iter)), costs)
    plt.show()

logistic_regression()
