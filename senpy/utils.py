# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:14:24 2020

@author: alexc
"""

import numpy as np

def custom_log(model, string, br=False):
    if hasattr(model, 'log_file'):
        file_obj = open(model.log_file, 'a')
        if br:
            file_obj.write('\n' + '-'*70 + '\n')
            file_obj.write(string + '\n')
            file_obj.write('-'*70 + '\n')
        else:
            file_obj.write(string + '\n')
        file_obj.close()

def _round(model, val):
    if hasattr(model, 'resolution'):
        return (model.resolution * (np.array(val) / 
                model.resolution).round()).round(model.precision)
    else:
        return round(val, model.precision)

def check_bounds(model, level):
    if model.lower_bound != None:
        if level < model.lower_bound:
            str_out = '\n!!! The algorithm would like to suggest testing at level {}, however this exceeds the prescribed lower bound. If the test apparatus can be adjusted, test at level {}; otherwise test at level {}.\n'.format(level, level, model.lower_bound)
            print(str_out)
            custom_log(str_out)
    if model.upper_bound != None:
        if level > model.upper_bound:
            str_out = '\n!!! The algorithm would like to suggest testing at level {}, however this exceeds the prescribed upper bound. If the test apparatus can be adjusted, test at level {}; otherwise test at level {}.\n'.format(level, level, model.upper_bound)
            print(str_out)
            custom_log(str_out)
            
def check_success(Y):
    return Y.size == np.sum(Y)

def check_fail(Y):
    return np.sum(Y) == 0

def check_diff(X, Y, inverted):
    Yc = Y.copy().astype(bool)
    if inverted:
        Yc = np.logical_not(Yc)
    if check_success(Yc) or check_fail(Yc):
        return 1
    min_go = np.min(X[Yc])
    max_no = np.max(X[np.logical_not(Yc)])
    diff = min_go - max_no
    return diff
