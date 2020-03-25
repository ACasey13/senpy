# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:29:17 2020

@author: alexc
"""

import numpy as np

STABILITY = 1E-8
    
def prob(x, alpha, beta):
    """
    Cumulative distribution function (CDF) for a log-logistic distribution
    """
    return np.minimum(np.maximum(1./(1+(x/alpha)**-beta),0),1)

def pred(pts, theta, inverted):
    """
    Returns the predictive probability at given pts under an assumed latent log-logistic distribution.
    """
    alpha, beta = theta
    P = prob(pts, alpha, beta)
    if inverted:
        return 1.0 - P
    else:
        return P

def cost(theta, X, Y):
    """
    Cost function under an assumed latent log-logistic distribution.

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
    alpha = theta[0]
    beta = theta[1]
    p_values = prob(X, alpha, beta)
    term_1 = -1 * Y * np.log(p_values + STABILITY)
    term_2 = -1 * (1 - Y) * np.log(1 - p_values + STABILITY)
    return np.sum(term_1 + term_2)

def cost_deriv(theta, X, Y):
    """
    Derivative with respect to theta of the cost function under an assumed latent log-logistic distribution.

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
    alpha = theta[0]
    beta = theta[1]
    P = prob(X, alpha, beta)
    Q = 1-P
    t1 = Y*Q - (1-Y)*P
    term_1 = (beta/alpha) * t1
    term_2 = -1 * np.log(X/alpha + STABILITY) * t1
    
    return np.sum(np.hstack((term_1, term_2)), axis=0)

def opt_config(X):
    low = max(np.min(X), 0.00001)
    high = np.max(X)
    if high <= low: high = low+1
    sigma_0 = (high - low)/4
    sigma_low  = max(sigma_0/2, .00001)
    sigma_high = sigma_0 + sigma_0/2
    bounds = [(.00001, None), (.00001, None)]
    return [low, high, sigma_low, sigma_high, bounds]

def estimate_names(latex=False):
    if latex:
        return [r'$\alpha$', r'$\beta$']
    else:
        return ['alpha', 'beta']

def Hessian(X, y, alpha, beta):
    P = prob(X, alpha, beta)
    Q = 1 - P
    
    t = P * Q
    t1 = y*Q - (1-y)*P
    
    a_11 = np.sum(-t * (beta/alpha)**2 + (beta/alpha**2)*t1)
    a_12 = np.sum(+t * (beta/alpha) * np.log(X/alpha+STABILITY) - (1/alpha)*t1) 
    a_22 = np.sum(-t * np.log(X/alpha + STABILITY)**2)
    
    return np.array([[a_11, a_12], [a_12, a_22]])

def cdf_deriv(x_pts, alpha, beta):
    """
    Derivative of the latent distribution cdf with respect to alpha and beta.

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
        The gradient of the cdf at n points with respect to alpha and beta.
    """
    P = prob(x_pts, alpha, beta)
    Q = 1 - P
    t = P * Q
    dalpha = -1 * (beta/alpha) * t
    dbeta = np.log(x_pts/alpha + STABILITY) * t
    return np.hstack((dalpha, dbeta))

def expected_info(X, alpha, beta):
    P = prob(X, alpha, beta)
    Q = 1 - P
    A = P * Q
    a_11 = np.sum((beta/alpha)**2 * A)
    a_12 = np.sum(-1*(beta/alpha)*np.log(X/alpha+STABILITY)*A)
    a_22 = np.sum(2*np.log(X/alpha+STABILITY)*A)
    info = np.array([[a_11, a_12], [a_12, a_22]])
    return info

function_dictionary = {'cost': cost,
                       'cost_deriv': cost_deriv,
                       'opt_config': opt_config,
                       'pred': pred,
                       'estimate_names': estimate_names,
                       'Hessian': Hessian,
                       'cdf_deriv': cdf_deriv,
                       'info': expected_info}
