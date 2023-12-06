import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import sample
pd.options.mode.chained_assignment = None

sns.set_theme()
sns.set_style("darkgrid")


def read_data():
    df = pd.read_csv('Data/Invistico_Airline.csv')
    data, y = df.drop(columns=['satisfaction']), df[['satisfaction']]
    return data, y


def train_test_split(x, y):
    mask = np.ones((len(x), 1), dtype=bool)
    rand = sample(range(len(x)), int(0.7 * len(x)))
    mask[rand] = False

    X_train, y_train, X_test, y_test = x[~mask], y[~mask], x[mask], y[mask]
    return X_train, y_train, X_test, y_test


def clean_data(x, y):
    y.loc[y['satisfaction'] == 'satisfied', 'satisfaction'] = 1
    y.loc[y['satisfaction'] == 'dissatisfied', 'satisfaction'] = 0

    for col in x.select_dtypes(include='object'):
        unique = x[col].unique()
        encode = {k: v for k, v in zip(unique, range(len(unique)))}
        x.replace({col: encode}, inplace=True)
    x.fillna(x.median(), inplace=True)
    return x, y


def likelihood_calc(x, y):
    cont_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    fig, axes = plt.subplots(2, 4)
    y0 = y.loc[:, 'satisfaction'] == 0
    y1 = y.loc[:, 'satisfaction'] == 1
    L0, L1 = {}, {}  # Dictionaries to store conditional probabilities for each feature given y
    for i, col in enumerate(cont_cols):
        density = axes[0][i].hist(x.loc[y0[y0].index, col])
        L0[col] = [density[0] / len(y0[y0].index),
                   pd.arrays.IntervalArray.from_arrays(density[1][:-1], density[1][1:], closed='both')]
        density = axes[1][i].hist(x.loc[y1[y1].index, col])
        L1[col] = [density[0] / len(y1[y1].index),
                   pd.arrays.IntervalArray.from_arrays(density[1][:-1], density[1][1:], closed='both')]
        axes[0][i].set_title(col)
        axes[1][i].set_xlabel('Value')
    disc_cols = [col for col in x.columns if col not in cont_cols]
    for i, col in enumerate(disc_cols):
        densities = x.loc[y0[y0].index, col].value_counts()
        L0[col] = dict(densities / len(y0[y0].index))
        densities = x.loc[y1[y1].index, col].value_counts()
        L1[col] = dict(densities / len(y1[y1].index))
    fig.supylabel('Frequency')
    fig.tight_layout(pad=1.5)
    return L0, L1


def find_prob(col, x_val, L):
    if col in ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        mask = L[col][1].contains(x_val)
        return L[col][0][mask][0] if any(mask) else 0.1
    else:
        return L[col][x_val] if x_val in L[col] else 0.1


def metric(y_pred, y):
    return np.sum(y_pred == y) / len(y)


def naive_bayes():
    x, y = read_data()
    X_train, y_train, X_test, y_test = train_test_split(x, y)
    X_train, y_train = clean_data(X_train, y_train)
    y0, y1 = y_train.loc[:, 'satisfaction'] == 0, y_train.loc[:, 'satisfaction'] == 1
    y0, y1 = len(y0[y0].index) / len(y_train), len(y1[y1].index) / len(y_train)
    L0, L1 = likelihood_calc(X_train, y_train)

    y_pred = []
    X_test, y_test = clean_data(X_test, y_test)
    for index, row in X_test.iterrows():
        p0, p1 = y0, y1
        for col in X_test.columns:
            p0 *= find_prob(col, row[col], L0)  # Calculate likelihood using conditional independence
            p1 *= find_prob(col, row[col], L1)
        prediction = 0 if p0 >= p1 else 1
        y_pred.append(prediction)
    print('Accuracy: ', metric(np.array(y_pred), np.array(y_test['satisfaction'])))


naive_bayes()
