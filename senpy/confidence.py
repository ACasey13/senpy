# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:14:24 2020

@author: alexc
"""

import scipy.stats as st
import numpy as np
from scipy.stats import chi2
from skimage import measure

class HomogeneousResult(Exception):
    pass

def _get_dist(latent, theta):
    if latent == 'normal':
        dist = st.norm(*theta)
    elif latent == 'logistic':
        dist = st.logistic(*theta)
    elif latent == 'log-logistic':
        dist = st.fisk(c=theta[1], scale=theta[0])
    return dist

def _determine_outcome(stimulus, threshold, inverted):
    result = np.zeros(threshold.shape)
    if inverted:
        result[stimulus <= threshold] = 1
    else:
        result[stimulus >= threshold] = 1
    return result

def parametric_bootstrap(model, pts, num_samples, CI_level):
    X = model.X
    CI_level = np.array(CI_level).flatten()
    
    dist = _get_dist(model.latent, model.theta)
    draw = dist.rvs((X.shape[0], num_samples))
    sampled_outcomes = _determine_outcome(X, draw, model.inverted).T
    
    predictions = []
    for y in sampled_outcomes:
        try:
            model.fit(X, y)
        except HomogeneousResult:
            continue
        predictions.append(model.pred(pts, model.theta, model.inverted).flatten())
    predictions = np.array(predictions)
    
    lb = np.atleast_2d(np.quantile(predictions, (1-CI_level)/2, axis=0)).T
    ub = np.atleast_2d(np.quantile(predictions, 1-(1-CI_level)/2, axis=0)).T
    return lb, ub

def nonparametric_bootstrap(model, pts, num_samples, CI_level):
    idxs = np.arange(model.X.shape[0])
    X = model.X.flatten()
    Y = model.Y.flatten()
    CI_level = np.array(CI_level).flatten()
    
    draw = np.random.choice(idxs, size=(num_samples, X.size))
    X = X[draw]
    Y = Y[draw]
    
    predictions = []
    for x_gen, y_gen in zip(X , Y):
        try:
            model.fit(x_gen.reshape((-1,1)), y_gen)
        except HomogeneousResult:
            continue
        predictions.append(model.pred(pts, model.theta, model.inverted).flatten())
    predictions = np.array(predictions)
    
    lb = np.atleast_2d(np.quantile(predictions, (1-CI_level)/2, axis=0)).T
    ub = np.atleast_2d(np.quantile(predictions, 1-(1-CI_level)/2, axis=0)).T
    return lb, ub

def delta(model, pts, num_samples, CI_level, preds):
    if model.inverted:
        Y = np.logical_not(model.Y).astype(int)
    else: 
        Y = model.Y
    pts = pts.reshape((-1,1))
    CI_level = np.array(CI_level).flatten()
    H = model.Hessian(model.X, Y, model.theta[0], model.theta[1])
    I = np.linalg.pinv(-1*H)
    jac = model.cdf_deriv(pts, model.theta[0], model.theta[1])
    w, _ = np.linalg.eig(H)
    
    if np.prod(w) > 0 and H[0,0] < 0:
        pass
    else:
        raise Exception("""According to the Hessian of the likelihood, maximum likelihood has not been achieved!
Hessian:\n{}""".format(H))
    
    z_factor = -st.norm.ppf((1.-CI_level)/2.)
    sigma_bound = np.sqrt(np.sum(np.matmul(jac, I) * jac, axis=1)).reshape((-1,1))
    lb = preds - z_factor * sigma_bound
    ub = preds + z_factor * sigma_bound
    lb[lb < 0] = 0
    ub[ub > 1] = 1
    return lb, ub

def increase_bounds(bounds, t1, t2):
    #print('t values')
    #print(t1)
    #print(t2)
    
    t1_low, t1_high, t2_low, t2_high, limits = bounds
    
    def check_bound(val, limit, direction):
        if limit is None:
            return val
        if direction == 'lower':
            return max(val, limit)
        elif direction == 'uppper':
            return min(val, limit)
        else:
            raise ValueError('Bounds check failed. Limit or direction not understood.')
    
    t1_change = (t1_high - t1_low)
    t2_change = (t2_high - t2_low)
    new_bounds = bounds.copy()
    
    if t1 == 'both':
        new_bounds[0] = check_bound(t1_low - t1_change/2, limits[0][0], 'lower')
        new_bounds[1] = check_bound(t1_high + t1_change/2, limits[0][1], 'upper')
    elif t1 == 'lower':
        new_bounds[0] = check_bound(t1_low - t1_change, limits[0][0], 'lower')
    elif t1 == 'upper':
        new_bounds[1] = check_bound(t1_high + t1_change, limits[0][1], 'upper')
    if t2 == 'both':
        new_bounds[2] = check_bound(t2_low - t2_change/2, limits[1][0], 'lower')
        new_bounds[3] = check_bound(t2_high + t2_change/2, limits[1][1], 'upper')
    elif t2 == 'lower':
        new_bounds[2] = check_bound(t2_low - t2_change, limits[1][0], 'lower')
    elif t2 == 'upper':
        new_bounds[3] = check_bound(t2_high + t2_change, limits[1][1], 'upper')
    
    return new_bounds

def interp_bounds(val, int_0, int_1, n):
    return (val / (n-1)) * (int_1 - int_0) + int_0

def get_contours(vals, level, bounds, n):
    if len(n) == 1:
        n1 = n
        n2 = n
    else:
        n1 = n[0]
        n2 = n[1]
    cntrs = measure.find_contours(vals, level)
    for ix, cntr in enumerate(cntrs):
        x = interp_bounds(cntr[:,[1]], bounds[0], bounds[1], n1)
        y = interp_bounds(cntr[:,[0]], bounds[2], bounds[3], n2)
        cntrs[ix] = np.hstack((x,y))
    return cntrs

def map_likelihood_ratio(model, bounds, n, levels, max_iter, field=False,
                         counter=1):
    if isinstance(n, int):
        n = [n,]
    n = list(n)
    if len(n) == 1:
        n = n*2
        
    t1_interval = np.linspace(bounds[0], bounds[1], n[0])
    t2_interval = np.linspace(bounds[2], bounds[3], n[1])
        
    test_t1, test_t2 = np.meshgrid(t1_interval, t2_interval)
    
    if model.inverted:
        Y = np.logical_not(model.Y).astype(int)
    else:
        Y = model.Y
        
    log_likelihoods = []
    for test_theta in zip(test_t1.flatten(), test_t2.flatten()):
        log_likelihoods.append(model.cost_func(test_theta, model.X, Y))
        
    current_likelihood = model.cost_func(model.theta, model.X, Y)
    log_likelihoods = np.asarray(log_likelihoods)
    ratios = 2*(log_likelihoods - current_likelihood).reshape(test_t1.shape)
    chi_squared = chi2(2)
    
    if field:
        return test_t1, test_t2, chi_squared.cdf(ratios)
    
    max_level = max(levels)
    
    max_contour = measure.find_contours(chi_squared.cdf(ratios), max_level)
    #print('\n')
    #print('-'*50)
    #print(counter)
    #print(len(max_contour))
    if len(max_contour) == 1:
        rms = np.mean(np.sqrt(np.sum(np.diff(max_contour[0], axis=0)**2, axis=1)))
        test_rms = np.sqrt((max_contour[0][-1,0] - max_contour[0][0,0])**2 + 
                           (max_contour[0][-1,1] - max_contour[0][0,1])**2)
        #print('rms: {}'.format(rms))
        #print('test_rms: {}'.format(test_rms))
        if test_rms > 2* rms:
            if counter >= max_iter:
                return 0, {level:get_contours(chi_squared.cdf(ratios), level, bounds, n) for level in levels}
            t1, t2 = None, None
            if np.abs(np.min(max_contour[0][:,1]) - 0) < 1:
                t1 = 'lower'
            elif np.abs(np.max(max_contour[0][:,1]) - (n[0]-1)) < 1:
                t1 = 'upper'
            if np.abs(np.min(max_contour[0][:,0]) - 0) < 1:
                t2 = 'lower'
            elif np.abs(np.max(max_contour[0][:,0]) - (n[1]-1)) < 1:
                t2 = 'upper'
                
            if t1 is None and t2 is None:
                t1, t2 = 'both', 'both'
                
            if t1 is not None:
                n[0] = n[0] * 2
            if t2 is not None:
                n[1] = n[1] * 2

            new_bounds = increase_bounds(bounds, t1, t2)
            #print('rms fail: increasing bounds')
            return map_likelihood_ratio(model, new_bounds, n, levels, 
                                        max_iter, counter=counter+1)
        else:
            #print('max_contour success: now exiting')
            return 1, {level:get_contours(chi_squared.cdf(ratios), level, bounds, n) for level in levels}
            
    else:
        if counter >= max_iter:
            return 0, {level:get_contours(chi_squared.cdf(ratios), level, bounds, n) for level in levels}
        #print('0 or >1 max contours: increasing bounds')
        new_bounds = increase_bounds(bounds, 'both', 'both')
        return map_likelihood_ratio(model, new_bounds, np.array(n)*2, levels,
                                    max_iter, counter=counter+1)
    
def convert_levels(asked):
    return np.interp(asked, [0, .5, .8, .9, .95, .98, .99,  1], 
                            [0, .2033, .5602, .7416, .8536, .9333, .9638, 1])


def contour_walk(model, pts, bounds, n, levels, max_iter):
    levels = convert_levels(levels)
    flag, contours = map_likelihood_ratio(model, bounds, n, levels, max_iter)
    lb = []
    ub = []
    if flag:
        for level in levels:
            if len(contours[level]) > 1:
                raise Exception("""Likelihood ratio contours not fully closed.
                                 Try increasing 'max_iters' under predict_probability method.
                                 Otherwise, data may be ill defined or choose another confidence method.""")
            theta = contours[level][0].T
            preds = model.pred(pts, theta, model.inverted)
            lb.append(np.min(preds, axis=1))
            ub.append(np.max(preds, axis=1))
        lb = np.atleast_2d(np.array(lb)).T
        ub = np.atleast_2d(np.array(ub)).T
        
        return lb, ub
    else:
        raise Exception("""Likelihood ratio contours not fully closed.
                        Try increasing 'max_iters' under predict_probability method.
                        Otherwise, data may be ill defined or choose another confidence method.""")
            
    
