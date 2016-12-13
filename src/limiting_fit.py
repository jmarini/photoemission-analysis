#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division

import os

import pandas as pd

import numpy as np

import scipy

import statsmodels.api as sm

import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

from linear_fit import LinearFit


def cluster_data(data, columns):
    """ Perform k-means clustering (N=2) for the specified columns.

        Filters on the cluster with the higher mean threshold energy.
    """
    X = data[columns].values
    sX = scale(X)

    estimator = KMeans(n_clusters=2)
    estimator.fit(sX)
    Z = estimator.predict(sX)

    data['cluster'] = Z
    cluster = data.groupby(['cluster']).mean().sort_values(by='Eth', ascending=False).index[0]

    return data[data.cluster==cluster]


def iterative_fit(data, xcol, ycol, delta=1e-3, max_iterations=10):
    """ Perform iterative OLS fit to find the limiting relationship in the data.
    """
    condition = None
    last = 0.

    for n in range(max_iterations):
        if condition is not None:
            df = data[condition]
        else:
            df = data

        x = df[xcol].values
        X = sm.add_constant(x)
        y = df[ycol].values

        model = sm.OLS(y, X)
        results = model.fit()
        fit = LinearFit(x, y, results)
        if np.abs(last - fit.x_intercept) <= delta:
            print('Converged in {} iterations'.format(n))
            break
        last = fit.x_intercept

        condition = data[ycol] > (fit.yp(xp=data[xcol]) - fit.confidence_interval(xp=data[xcol]))

    return fit


if __name__ == '__main__':
    data = pd.read_csv('data/interim/photoemission.csv')

    data['logQE'] = np.log10(data.QE)
    data.loc[data.logQE==-np.inf, 'logQE'] = -6
    data['sqrtQE'] = np.sqrt(data.QE)
    data['QE3'] = np.power(data.QE, 1. / 3.)
    data['slope2'] = np.power(data.slope, 2.)
    data['slope3'] = np.power(data.slope, 3.)

    df = cluster_data(data, ['Eth', 'logQE'])

    xerr = np.array([df.err_minus.values, df.err_plus.values])

    ycol = 'slope'
    fit = iterative_fit(df, 'Eth', ycol)

    fig, ax = plt.subplots(1, 1)
    ymax = df[ycol].max() * 1.1
    df.plot(x='Eth', y=ycol, kind='scatter', xerr=xerr, c='k', ylim=(0, ymax), xlim=(4, 5), ax=ax)
    ax.set_xlabel(r'$E_{th}$')
    ax.set_ylabel(r'$m^3$')
    fit.plot_fit(np.arange(4, 5, 0.01), axis=ax, ci=False)
    plt.show()
