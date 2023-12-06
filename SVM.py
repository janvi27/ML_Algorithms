import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample


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


def boundary(x, theta):
    return np.matmul(x, theta)

