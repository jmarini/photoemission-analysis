# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division

import pandas as pd

import numpy as np

import scipy

import statsmodels.api as sm


class LinearFit(object):

    def __init__(self, x, y, fit):
        self.x = x
        self.X = sm.add_constant(x)
        self.y = y
        self.fit = fit

        self.b = fit.params
        self.y_intercept, self.slope = fit.params
        self.x_intercept = -self.y_intercept / self.slope

        self.xm = np.mean(x)
        self.Sxx = np.sum(np.power(self.x - self.xm, 2))
        self.ym = np.mean(y)
        self.n = fit.nobs
        self.dof = fit.df_resid
        self.rsquared = fit.rsquared
        self.stddev = np.sqrt(fit.ssr / fit.df_resid)

    def yp(self, xp):
        """Calculates line over given x range."""
        return np.dot(sm.add_constant(xp), self.b)

    def tstar(self, alpha):
        """Statistical t-test for confidence intervals."""
        return scipy.stats.distributions.t.ppf(1.0 - alpha / 2.0, self.dof)

    def confidence_interval(self, xp=None, yp=None, alpha=0.05):
        """Calculates confidence interval for either x or y values."""
        if not (xp is None) ^ (yp is None):
            raise ValueError('Only one of [xp, yp] must be specified.')

        if xp is not None:
            return self._y_ci(xp, alpha=alpha)
        if yp is not None:
            return self._x_ci(yp, alpha=alpha)

    def prediction_interval(self, xp=None, yp=None, alpha=0.05):
        """Calculates prediction interval for either x or y values."""
        if not (xp is None) ^ (yp is None):
            raise ValueError('Only one of [xp, yp] must be specified.')

        if xp is not None:
            return self._y_pi(xp, alpha=alpha)
        if yp is not None:
            return self._x_pi(yp, alpha=alpha)

    def plot_fit(self, xp, axis, alpha=0.05, ci=True):
        """Plots the fit over given x range."""
        yp = self.yp(xp)

        if ci:
            ci = self.confidence_interval(xp, alpha=alpha)
            pi = self.prediction_interval(xp, alpha=alpha)
            axis.fill_between(xp, yp - pi, yp + pi, color='k', alpha=0.1)
            axis.fill_between(xp, yp - ci, yp + ci, color='k', alpha=0.2)
        axis.plot(xp, yp, c='k', ls=':')

    def _y_ci(self, xp, alpha=0.05):
        return (self.tstar(alpha) * self.stddev
                * np.sqrt((1. / self.n) + (np.power(xp - self.xm, 2) / self.Sxx)))

    def _y_pi(self, xp, alpha=0.05):
        return (self.tstar(alpha) * self.stddev
                * np.sqrt(1. + (1. / self.n) + (np.power(xp - self.xm, 2) / self.Sxx)))

    def _x_ci(self, yp, alpha=0.05):
        """Inverse regression, after Sec 3.2 from Draper & Smith 1998."""
        xp = (yp - self.b[0]) / self.b[1]
        dx = xp - self.xm
        ts = self.tstar(alpha) * self.stddev
        b1 = self.b[1]
        g = (ts / b1) ** 2 / self.Sxx

        left = dx * g
        denom = (1. - g)
        right = (ts / b1) * np.sqrt(dx ** 2 / self.Sxx + denom / self.n)

        return np.array([(left - right) / denom, (left + right) / denom])

    def _x_pi(self, yp, alpha=0.05):
        """Inverse regression, after Sec 3.2 from Draper & Smith 1998."""
        xp = (yp - self.b[0]) / self.b[1]
        dx = xp - self.xm
        ts = self.tstar(alpha) * self.stddev
        b1 = self.b[1]
        g = (ts / b1) ** 2 / self.Sxx

        left = dx * g
        denom = (1. - g)
        right = (ts / b1) * np.sqrt(dx ** 2 / self.Sxx + denom / self.n + denom)

        return np.array([(left - right) / denom, (left + right) / denom])
