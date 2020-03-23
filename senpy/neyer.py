# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:32:38 2020

@author: alexc
"""

import numpy as np
from scipy.optimize import minimize, brute, fmin
from .confidence import (parametric_bootstrap, nonparametric_bootstrap,
                        delta, contour_walk, increase_bounds,
                        HomogeneousResult)
from .plotting import plot_probability as pp, plot_confidence_region as pcr
import copy
from .utils import (custom_log, _round, check_bounds, check_diff, 
                   check_success, check_fail)

class Neyer():
    """
    Implements the maximum likelihood estimates of parameters theta given a set of data.
    Optimization algorithms used are L-BFGS-G (default), TNC, and SLSQP.
    """
    
    available_opt_methods = ('L-BFGS-B', 'SLSQP', 'TNC')

    def __init__(self, latent='normal', inverted=False,
                 method='L-BFGS-B', num_restarts=3,
                 t1_min=None, t1_max=None, t2_guess=None,
                 precision=8, resolution=None, 
                 lower_bound=None, upper_bound=None, 
                 hist=False, log_file=None):
        
        self.inverted = inverted
        self.theta = None
        self.latent = latent
        self.method = method
        self.num_restarts = num_restarts
        
        if self.num_restarts < 1:
            print('Number of restarts must be greater than or eqaul to 1.')
            print('Defaulting to 3.')
            self.num_restarts = 3
        
        if self.method not in self.available_opt_methods:
            print("""method '{}' not understood.
                 Defaulting to L-BFGS-B.
                 Please choose from {}""".format(self.method, 
                                                 self.available_opt_methods))
            self.method = 'L-BFGS-B'
        
        if latent == 'normal':
            from .norm_funcs import function_dictionary
        elif latent == 'logistic':
            from .logit_funcs import function_dictionary
        elif latent == 'log-logistic':
            from .log_logit_funcs import function_dictionary
        else: 
            raise ValueError("""Value for "latent" not understood.
            Must be "normal", "logistic", or "log-logistic".""")
            
        self.pred = function_dictionary['pred']
        self.opt_config = function_dictionary['opt_config']
        self.cost_func = function_dictionary['cost']
        self.cost_deriv = function_dictionary['cost_deriv']
        self.est_names = function_dictionary['estimate_names']
        self.Hessian = function_dictionary['Hessian']
        self.cdf_deriv = function_dictionary['cdf_deriv']
        self.info = function_dictionary['info']
        
        self.precision = precision
        self.start = True
        self.binary = True
        self.overlap = True
        self.mle = True
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.hist = hist

        if isinstance(log_file, str):
            self.log_file = log_file
            file_obj = open(log_file, 'w')
            file_obj.close()

        if resolution != None:
            self.resolution = resolution

        if self.hist == True:
            self.det_vals = []
            self.det_res = []
            self.x_pts = []

        self.t1_min = t1_min
        self.t1_max = t1_max
        self.t2_guess = t2_guess

        self.X = np.asarray([]).reshape((-1,1))
        self.Y = np.asarray([]).reshape((-1,1))
        self.theta = np.array([np.nan, np.nan])
        self.observed_info = np.empty((2,2))

        self.updated = -1
            
    def fit(self, X, Y):
        if X.ndim != 2:
            raise ValueError("X must be of shape [n_examples, 1]")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("""input and output must have the same number of rows!
            shapes {} and {} do not match.""".format(X.shape, Y.shape))
    
        Y = Y.reshape((-1,1))
        self.Y = Y.copy()
        self.X = X
        
        if self.inverted:
            Y = np.logical_not(Y).astype(int)
            
        if check_success(Y) or check_fail(Y):
            raise HomogeneousResult('Need to have positive AND negative responses present in the data in order to call fit.')
    
        thetas = []
        costs = []
        t1_low, t1_high, t2_low, t2_high, bounds = self.opt_config(self.X)
    
        for i in range(self.num_restarts):
            theta_0 = [np.random.uniform(t1_low, t1_high),
                       np.random.uniform(t2_low, t2_high)]
            theta_0 = np.array(theta_0)
    
            res = minimize(self.cost_func, theta_0, 
                           args = (self.X, Y), 
                           method=self.method, 
                           jac=self.cost_deriv,
                           bounds=bounds)
            thetas.append(res.x)
            costs.append(res.fun)
            
        thetas = np.asarray(thetas)
        costs = np.asarray(costs)
    
        best_run = np.argmin(costs)
        
        self.theta = thetas[best_run]
        self.cost = costs[best_run]
        
        return self
    
    def get_estimators(self):
        if self.theta is not None:
            if check_diff(self.X, self.Y, self.inverted) > 0:
                raise Exception('Not enough data to estimate theta.')
            return self.theta
        else:
            raise Exception('Model not yet trained!')

    def print_estimators(self, cost=False):
        if self.theta is not None:
            if check_diff(self.X, self.Y, self.inverted) > 0:
                raise Exception('Not enough data to estimate theta.')
            t1n, t2n = self.est_names()
            t1, t2 = self.theta
            print('{}: {}\n{}: {}'.format(t1n, t1, t2n, t2))
            if cost:
                print('cost: {}'.format(self.cost))
        else:
            raise Exception('Model not yet trained!')
    
    def predict_probability(self, pts=None, confidence=None, 
                            CI_level = [.5, .8, .9, .95], 
                            num_samples=1000, max_iter=5):
        if self.theta is None:
            raise Exception('Model not yet trained!')
        if check_diff(self.X, self.Y, self.inverted) > 0:
            raise Exception('Not enough data to make a prediction.')
            
        if pts is None:
            xmin = np.min(self.X)
            xmax = np.max(self.X)
            xint = xmax-xmin
            xstart = xmin - xint*.05
            xend = xmax + xint*.05
            pts = np.linspace(xstart, xend, 100)
            
        pts = np.array(pts).reshape((-1,1))
            
        p = self.pred(pts, self.theta, self.inverted)

        if confidence is None:
            return pts, p
        elif confidence == 'parametric-bootstrap':
            current_model = copy.deepcopy(self)
            lb, ub = parametric_bootstrap(current_model, 
                                             pts,
                                             num_samples, 
                                             CI_level)
            return pts, p, lb, ub
        elif confidence == 'nonparametric-bootstrap':
            current_model = copy.deepcopy(self)
            lb, ub = nonparametric_bootstrap(current_model, 
                                             pts,
                                             num_samples, 
                                             CI_level)
            return pts, p, lb, ub
        elif confidence == 'likelihood-ratio':
            new_bounds = increase_bounds(self.opt_config(self.X), 
                                         'both', 'both')
            lb, ub = contour_walk(self, pts, new_bounds, [100], 
                                  CI_level, max_iter)
            return pts, p, lb, ub
        elif confidence == 'delta' or confidence == 'perturbation':
            lb, ub = delta(self, 
                           pts,
                           num_samples, 
                           CI_level, p)
            return pts, p, lb, ub
        else:
            ci_methods = [None, 'parametric-bootstrap', 
                          'nonparametric-bootstrap', 'likelihood-ratio',
                          'delta', 'perturbation']
            raise ValueError("confidence '{}' not understood.\nPlease choose from {}".format(confidence, ci_methods))
            
    def plot_probability(self, include_data=True, xlabel=None, ylabel=None,
                         alpha=1.0, save_dst=None, show=False, **kwargs):
        pp(self, include_data, xlabel, ylabel,
           alpha, save_dst, show, **kwargs)
        
    def plot_confidence_region(self, limits, n, CI_levels=10, 
                               save_dst=None, show=False):
        if self.theta is None:
            raise Exception('Model not yet trained!')
        if check_diff(self.X, self.Y, self.inverted) > 0:
            raise Exception('Not enough data to make a prediction.')
        pcr(self, limits, n, CI_levels, save_dst, show)
        
    def __prompt_input(self):
        t1n, t2n = self.est_names()
        self.t1_min = float(input('Lower bound guess for {}: '.format(t1n)))
        self.t1_max = float(input('Upper bound guess for {}: '.format(t1n)))
        self.t2_guess = float(input('Initial guess for {}: '.format(t2n)))

    def __max_info(self, theta):
        def det(level):
            X_test = np.vstack((self.X, level))
            info = self.info(X_test, theta[0], theta[1])
            return -1*(info[0][0] * info[1][1] - info[0][1] * info[1][0])

        ranges = self.max_s - self.min_s

        if self.lower_bound == None and self.upper_bound == None:
            res = brute(det, ((self.min_s - .5*ranges, self.max_s + .5*ranges),),
                    Ns=100, finish=fmin)
        else:
            if self.lower_bound == None:
                lb = self.min_s - ranges
            else: lb = self.lower_bound

            if self.upper_bound == None:
                ub = self.min_s + ranges
            else: ub = self.upper_bound
            res = brute(det, ((lb, ub),),
                    Ns=100, finish=fmin)

        if self.hist:
            if self.lower_bound == None:
                x_pts = np.linspace(self.min_s - 2.5*ranges,
                                    self.max_s + 2.5*ranges,
                                    500)
            else:
                x_pts = np.linspace(self.lower_bound - .1 * ranges,
                                    self.upper_bound + .1 * ranges,
                                    500)
            self.x_pts.append(x_pts)
            d_res = []
            for i in x_pts:
                d_res.append(-1*det(np.asarray(i)))
            self.det_vals.append(d_res)
            self.det_res.append(float(res))

        return float(res)
    
    def __check_initial_theta(self):
        if self.t1_max <= self.t1_min:
            raise ValueError('t1_max cannot be less than t1_min!')
        elif self.t2_guess <= 0:
            raise ValueError('t2_guess must be positive!')
    
    def next_pt(self):
        Y = self.Y.copy().astype(bool)
        if self.inverted:
            Y = np.logical_not(Y)
                
        if self.start:
            self.start = False
            if self.X.size == 0:
                custom_log(self, 'Starting Sequential Algorithm with No Data', True)
                if (self.t1_min == None) or (self.t1_max == None) or (self.t2_guess == None): 
                    self.__prompt_input()
                    self.__check_initial_theta()
                self.nx = _round(self, (self.t1_min + self.t1_max) / 2.)
                check_bounds(self, self.nx)
                custom_log(self, 'Next Point Requested: {}'.format(self.nx))
                self.updated = 0
                return self.nx
            else:
                diff = check_diff(self.X, self.Y, self.inverted)
                if diff > 0:
                    if (self.t1_min == None) or (self.t1_max == None) or (self.t2_guess == None): 
                        print("""Even though data has been provided, overlap has not been achieved.
                              In this case it is necessary to provide parameters for t1_min, t1_max, and t2_guess.
                              """)
                        self.__prompt_input()
                        self.__check_initial_theta()
                    return self.next_pt()
                else:
                    self.binary = False
                    self.overlap = False
                    return self.next_pt()
        else:
            if self.X.size > self.updated:
                self.updated = self.X.size
            else:
                return self.nx

        if self.binary:
            self.max_s = np.max(self.X)
            self.min_s = np.min(self.X)
            custom_log(self, 'In Binary Search Section', True)
            custom_log(self, 'Min Stimlus: {}'.format(self.min_s))
            custom_log(self, 'Max Stimulus: {}'.format(self.max_s))

            # all success case
            if Y.size == np.sum(Y):
                custom_log(self, 'In All Success Section', True)
                t1 = (self.t1_min + self.min_s) / 2.
                t2 = self.min_s - 2. * self.t2_guess
                t3 = 2. * self.min_s - self.max_s
                self.nx = _round(self, min(t1, t2, t2))
                check_bounds(self, self.nx)
                custom_log(self, 'Next Point Requested: {}'.format(self.nx))

                return self.nx

            # all failure case
            if np.sum(Y) == 0:
                custom_log(self, 'In All Failure Section', True)
                t1 = (self.t1_max + self.max_s) / 2.
                t2 = self.max_s + 2. * self.t2_guess
                t3 = 2. * self.max_s - self.min_s
                self.nx = _round(self, max(t1, t2, t3))
                check_bounds(self, self.nx)
                custom_log(self, 'Next Point Requested: {}'.format(self.nx))
                return self.nx

            self.min_go = np.min(self.X[Y])
            self.max_no = np.max(self.X[np.logical_not(Y)])
            self.diff = round(self.min_go - self.max_no, self.precision)
            custom_log(self, 'Min Go: {}'.format(self.min_go))
            custom_log(self, 'Max No-Go: {}'.format(self.max_no))
            custom_log(self, 'Difference: {}'.format(self.diff))
            custom_log(self, 'Theta 2 guess: {}'.format(self.t2_guess))

            if self.diff > self.t2_guess:
                self.nx = _round(self, (self.max_no + self.min_go) / 2.)
                check_bounds(self, self.nx)
                custom_log(self, 'Next Point Requested: {}'.format(self.nx))
                return self.nx
            else:
                self.binary = False

        if self.overlap:
            custom_log(self, 'In Overlap Search Section', True)
            self.min_go = np.min(self.X[Y])
            self.max_no = np.max(self.X[np.logical_not(Y)])
            self.diff = round(self.min_go - self.max_no, self.precision)
            custom_log(self, 'Min Go: {}'.format(self.min_go))
            custom_log(self, 'Max No-Go: {}'.format(self.max_no))
            custom_log(self, 'Difference: {}'.format(self.diff))
            custom_log(self, 'Theta 2 guess: {}'.format(self.t2_guess))

            if self.diff > self.t2_guess:
                custom_log(self, 'Reverting Back to Binary Search', True)
                self.binary = True
                self.updated = -1
                return self.next_pt()

            if self.diff < 0:
                custom_log(self, '--- Overlap Achieved! ---', True)
                self.overlap = False

            else:
                self.theta[0] = (self.max_no + self.min_go) / 2.
                self.theta[1] = self.t2_guess
                custom_log(self, 'Maximize Determinate With...')
                t1n, t2n = self.est_names()
                custom_log(self, '{}: {}'.format(t1n, self.theta[0]))
                custom_log(self, '{}: {}'.format(t2n, self.theta[1]))

                self.nx = _round(self, self.__max_info(self.theta))
                self.t2_guess *= 0.8
                check_bounds(self, self.nx)
                custom_log(self, 'Next Point Requested: {}'.format(self.nx))
                return self.nx

        if self.mle:
            custom_log(self, 'In Maximum Liklihood Section', True)
            self.max_s = max(self.X)
            self.min_s = min(self.X)
            custom_log(self, 'Min Stimlus: {}'.format(self.min_s))
            custom_log(self, 'Max Stimulus: {}'.format(self.max_s))

            self.fit(self.X, self.Y)
            t1n, t2n = self.est_names()
            custom_log(self, 'Estimated {}: {}'.format(t1n, self.theta[0]))
            custom_log(self, 'Estimated {}: {}'.format(t2n, self.theta[1]))

            self.theta[0] = max(self.min_s, min(self.theta[0], self.max_s))
            self.theta[1] = min(self.theta[1], self.max_s - self.min_s)
            custom_log(self, 'Bounded Estimated {}: {}'.format(t1n, self.theta[0]))
            custom_log(self, 'Bounded Estimated {}: {}'.format(t2n, self.theta[1]))

            self.nx = _round(self, self.__max_info(self.theta))
            check_bounds(self, self.nx)
            custom_log(self, 'Next Point Requested: {}'.format(self.nx))

            return self.nx
        
    def post_test_outcome(self, res, pt):
        if isinstance(res, bool) or (res == 0) or (res == 1):
            self.X = np.vstack((self.X, pt))
            custom_log(self, 'Tested Points: \n {}'.format(self.X.flatten()))
            self.Y = np.vstack((self.Y, int(res)))
            custom_log(self, 'Test Results: \n {}'.format(self.Y.flatten()))
        else:
            raise ValueError('Result must be \{0, 1\} or \{True, False\}!')
            
    def loop(self, iterations=1000000):
        """
        This method suggests new test levels and accepts user input to calculate maximum likelihood estimates. That is, this method constitutes a loop. Loop will continue indefinitely until 'end' is received as user input during the either the test level or result input queries. Alternatively, if a set number of specimens is to be used then the number of loops can be specified with the 'iter' keyword argument.
        """

        print('-'*50)
        print("""If the level at which the test is performed is the same as the
        suggested level, then the user can simply press enter (no need for input)
        when queried about the test level.""")
        print('\n')
        print("""When the user does not wish to test any more levels,
        input "end" (without quotes) when queried abou the next test.""")
        print('-'*50)
        print('\n')

        for _ in range(iterations):
            nx = self.next_pt()
            print('Specimen number: {}'.format(self.X.size + 1))
            print('The next suggested test point is: {}'.format(nx))
            pt = input('Please input the level at which the test was performed: ')
            pt = "".join(pt.split()).lower()
            if pt == 'end':
                break
            elif pt == '':
                pt = nx
            else:
                try:
                    pt = float(pt)
                except:
                    print("Input level '{}' not understood. Try again. Type 'end' to terminate loop.".format(pt))
                    continue

            res = input('Please input the result: ')
            res = "".join(res.split()).lower()
            print('\n')
            if res == 'true' or res == '1':
                self.post_test_outcome(1, pt)
            elif res == 'false' or res == '0':
                self.post_test_outcome(0, pt)
            elif res == '':
                pass
            elif res == 'end':
                break
            else:
                print("Result value '{}' not understood. Input must be 0 or False for a negative response and 1 or True for a positive response. Boolean inputs are not case sensitive. Try again. Type 'end' during input query to terminate loop.".format(res))
