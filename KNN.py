import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import sample
pd.options.mode.chained_assignment = None


def read_data():
    df = pd.read_csv('Data/Invistico_Airline.csv')
    X, y = df.drop(columns=['satisfaction']), df[['satisfaction']]

    for col in X.select_dtypes(include='object'):
        unique = X[col].unique()
        encode = {k: v for k, v in zip(unique, range(len(unique)))}
        X.replace({col: encode}, inplace=True)
    y.loc[y['satisfaction'] == 'satisfied', 'satisfaction'] = 1
    y.loc[y['satisfaction'] == 'dissatisfied', 'satisfaction'] = 0

    return X, y


def scale(X, y):
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    X.fillna(X.mean(), inplace=True)
    return np.array(X, 'f'), np.array(y, 'int')


def split(X, y):
    mask = np.ones((len(X), 1), dtype=bool)
    rand = sample(range(len(X)), int(0.7 * len(X)))
    mask[rand] = False

    X_train, y_train, X_test, y_test = X[~mask], y[~mask], X[mask], y[mask]
    return X_train, y_train, X_test, y_test


def assign_cluster(X, mu, cluster):
    # Calculate which cluster for each data point
    for i, row in enumerate(X):
        dist = np.zeros((len(mu), 1))
        for idx, m in enumerate(mu):
            dist[idx, 0] = np.linalg.norm(row - mu[idx])
        cluster[i, 0] = np.argmin(dist)
    return cluster


def EM(mu, X, num_iter):
    cluster = np.zeros((len(X), 1))
    for i in range(num_iter):
        if i % 10 == 0:
            print('Iteration: ', i)
        # Assign points to the nearest cluster
        cluster = assign_cluster(X, mu, cluster)
        # Reassign means based on new clustering
        for j in range(len(mu)):
            mu[j] = np.mean(X[np.where(cluster == j)[0], :], axis=0)
    return mu


def metric(y, y_pred):
    return sum(np.array(y == y_pred)) / len(y)


def knn():
    X, y = read_data()
    X_train, y_train, X_test, y_test = split(X, y)
    X_train, y_train = scale(X_train, y_train)
    X_test, y_test = scale(X_test, y_test)

    means = np.zeros((2, len(X_train[0])))

    for i in range(len(X_train[0])):
        y0, y1 = np.array(y_train == 0).reshape(-1,), np.array(y_train == 1).reshape(-1,)
        means[0, i] = np.random.normal(X_train[y0, i].mean(), X_train[y0, i].std())
        means[1, i] = np.random.normal(X_train[y1, i].mean(), X_train[y1, i].std())
    means = EM(means, X_train, 50)

    y_pred = assign_cluster(X_test, means, np.zeros((len(X_test), 1)))
    accuracy = metric(y_test, y_pred)
    print('Accuracy: ', accuracy[0])


knn()
