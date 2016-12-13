#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division

import os

import pandas as pd

import numpy as np

import scipy

import statsmodels.api as sm

from linear_fit import LinearFit


def read_data(filename):
    """ Reads in raw photoemission spectra and performs initial transformation
        and cleaning.
    """
    data = pd.read_csv(filename, sep='[\t,]+', header=None, engine='python',
                       na_values=['XXX.XXX'], names=['nm', 'QE'])

    if data.nm.min() < 10.:  # first column is energy, not wavelength
        data['E'] = data.nm
        data['nm'] = 1240. / data.E  # conversion of energy to wavelength
    else:
        data['E'] = 1240. / data.nm  # conversion of wavelength to energy

    data['QE'] = data.QE.clip_lower(0.)
    data['QE3'] = np.power(data.QE, 1. / 3.)  # QE^(1/3) so we can use a linear fit

    return data


def find_linear(df, window_length=15):
    """ Fits linear portion of spectra using a rolling mean of `window_length`
        and takes the n best fits as candidates.
    """
    if window_length % 2 == 0:
        raise ValueError('The window must be odd')
    if window_length < 5:
        raise ValueError('The window must be at least 5')

    stats = {
        'i': [],
        'slope': [],
        'threshold': [],
        'r2': [],
        'stddev': [],
    }
    fits = []

    n = (window_length - 1) // 2
    for mid in range(n, len(df) - n):
        low, high = mid - n, mid + n + 1
        window = df.iloc[low:high]

        x = window.E.values
        X = sm.add_constant(x)
        y = window.QE3.values

        model = sm.OLS(y, X)
        results = model.fit()

        fit = LinearFit(x, y, results)

        fits.append(fit)
        stats['i'].append(mid)
        stats['slope'].append(fit.slope)
        stats['threshold'].append(fit.x_intercept)
        stats['r2'].append(fit.rsquared)
        stats['stddev'].append(fit.stddev)

    stats = pd.DataFrame(stats)

    # we are looking for the line with
    #   a) the highest R^2
    #   b) the highest slope

    max_slope = stats.slope.max()
    stats['best'] = stats.r2 + stats.slope / max_slope
    maxima = scipy.signal.argrelmax(stats.best.values)[0]
    best_fit = stats.iloc[maxima].best.idxmax()

    return fits[best_fit]


def qe_at(df, energy):
    """ Extracts QE at the specified energy.
    """
    return df[df.E >= energy].sort_values(by='E', ascending=True).iloc[0].QE


def extract_sample_id(filename):
    """ Extracts the sample id from the filename.


        Filenames are in the format 'ldbs<measurement id>_<sample id>.txt'.
    """
    return os.path.splitext(filename)[0].split('_')[1]


def read_all_data(directory, skip_files=None):
    """ Read all data in the raw directory and save as interim data.
    """
    skip_files = skip_files or []
    delta = 0.5
    data = {'s': [], 'slope': [], 'Eth': [], 'err_minus': [], 'err_plus': [], 'QE': []}
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if '_' not in file:
                continue
            try:
                sample = extract_sample_id(file)
            except:
                print('Error: {}'.format(file))
                continue

            if sample in skip_files:
                print('Skipping {}'.format(file))
                continue

            filename = os.path.join(directory, file)
            df = read_data(filename)
            f = find_linear(df, window_length=11)
            qe = qe_at(df, f.x_intercept + delta)

            data['s'].append(sample)
            data['Eth'].append(f.x_intercept)
            em, ep = np.abs(f.prediction_interval(yp=0))[::-1]
            data['slope'].append(f.slope)
            data['err_minus'].append(em)
            data['err_plus'].append(ep)
            data['QE'].append(qe)
    return pd.DataFrame(data)


if __name__ == '__main__':
    pe_data = read_all_data('data/raw/', skip_files=['s0492d', 'g2037', 's0508b'])
    pe_data.to_csv('data/interim/photoemission.csv', index=False,
                   columns=['s', 'QE', 'slope', 'Eth', 'err_minus', 'err_plus'])
