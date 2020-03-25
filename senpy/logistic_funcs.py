# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:29:17 2020

@author: alexc
"""

import numpy as np
from scipy.special import expit

STABILITY = 1E-8

def z(x, mu, sigma):
    """
    Returns the z-location of the given stimulus levels
    """
    return (x - mu) / sigma
    
def prob(z):
    """
    Cumulative distribution function (CDF) for a logistic distribution
    """
    return expit(z)
    

def pred(pts, theta, inverted):
    """
    Returns the predictive probability at given pts under an assumed latent logistic distribution.
    """
    mu, sigma = theta
    P = prob(z(pts, mu, sigma))
    if inverted:
        return np.maximum(1-P, 0.0)
    else:
        return P
    

def cost(theta, X, Y):
    """
    Cost function under an assumed latent logistic distribution.

    Parameters
    ----------
    theta : array
        theta[0] is mu
        theta[1] is sigma
    X : column vector
        The tested stimulus level(s)
    Y : column vector with values [0 or 1]
        The response

    Returns
    -------
    numeric
        The cost, or negative log likelihood, of the data under theta.

    """
    mu = theta[0]
    sigma = theta[1]
    p_values = prob(z(X, mu, sigma))
    term_1 = -1 * Y * np.log(p_values + STABILITY)
    term_2 = -1 * (1 - Y) * np.log(1 - p_values + STABILITY)
    return np.sum(term_1 + term_2)

def cost_deriv(theta, X, Y):
    """
    Derivative with respect to theta of the cost function under an assumed latent logistic distribution.

    Parameters
    ----------
    theta : array-like [1x2]
        theta[0] is mu
        theta[1] is sigma
    X : column vector
        The tested stimulus level(s)
    Y : column vector with values [0 or 1]
        The response

    Returns
    -------
    array-like
        The gradient vector [1x2] of cost with respect to theta.

    """
    mu = theta[0]
    sigma = theta[1]
    z_values = z(X, mu, sigma)
    p_values = prob(z_values)
    q_values = 1-p_values
    term_1 = Y * q_values / sigma
    term_2 = -1 * (1-Y) * p_values / sigma
    term_3 = Y * q_values * z_values / sigma
    term_4 = -1 * (1-Y) * p_values * z_values / sigma
    return np.sum(np.hstack((term_1 + term_2, term_3 + term_4)), axis=0)

def opt_config(X):
    low = np.min(X)
    high = np.max(X)
    sigma_0 = (high - low)/4
    sigma_low  = max(sigma_0/2, .00001)
    sigma_high = sigma_0 + sigma_0/2
    bounds = [(None, None), (.00001, None)]
    return [low, high, sigma_low, sigma_high, bounds]

def estimate_names(latex=False):
    if latex:
        return ['$\mu$', '$\sigma$']
    else:
        return ['mu', 'sigma']

def Hessian(X, y, mu, sigma):
    z_values = z(X, mu, sigma)
    P = prob(z_values)
    Q = 1 - P
    exp = np.nan_to_num(np.exp(-z_values))
    
    t = -1 * exp * (P/sigma)**2
    t1 = y*Q - (1-y)*P
    
    a_11 = np.sum(t)
    a_12 = np.sum(t*z_values + (1/sigma**2) * t1)
    a_22 = np.sum(t*z_values**2 + (2*z_values/sigma**2) * t1)
    
    return np.array([[a_11, a_12], [a_12, a_22]])

def cdf_deriv(x_pts, mu, sigma):
    """
    Derivative of the latent distribution cdf with respect to mu and sigma.

    Parameters
    ----------
    x_pts : array [n_pts x 1]
        An array of the levels at which to compute the gradient.
    mu : numeric
        The center of the normal distribution.
    sigma : numeric (strictly positive)
        The standard deviation of the normal distribution.

    Returns
    -------
    array [n_pts x 2]
        The gradient of the cdf at n points with respect to mu and sigma.

    """
    z_values = z(x_pts, mu, sigma)
    exp = np.nan_to_num(np.exp(-z_values))
    P = prob(z_values)
    t = exp * P**2 * (-1/sigma)
    dmu = t
    dsig = t * z_values
    return np.hstack((dmu, dsig))

def expected_info(X, mu, sigma):
    z_values = z(X, mu, sigma)
    exp = np.nan_to_num(np.exp(-z_values))
    p = prob(z_values)
    
    a_11 = (p/sigma)**2 * exp
    a_12 = np.sum(a_11 * z_values)
    a_22 = np.sum(a_11 * z_values**2)
    info = np.array([[np.sum(a_11), a_12], [a_12, a_22]])
    return info

function_dictionary = {'cost': cost,
                       'cost_deriv': cost_deriv,
                       'opt_config': opt_config,
                       'pred': pred,
                       'estimate_names': estimate_names,
                       'Hessian': Hessian,
                       'cdf_deriv': cdf_deriv,
                       'info': expected_info}
